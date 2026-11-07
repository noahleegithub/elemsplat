#!/usr/bin/env python
"""
semseg_train.py -- train one semantic network on one fold and predict its validation and
test clouds (paper Section 3.2 and 4.4).

    python semseg_train.py --data $BRPCD_DIR --model pointnet2 --fold 3 --out $RESULTS_DIR/trackA/semseg
    python semseg_train.py ... --fold all          # network for RQ3 (trained on every cloud)

Writes <out>/<model>/fold_<k>/{best.pt, train_log.csv, metrics_semseg.json, pred/<bridge>.npz}.
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_cfg, load_json, log, save_json, set_seed  # noqa: E402
from metrics import confusion, semantic_iou  # noqa: E402
from semseg_data import N_FEAT, BlockSampler, class_counts, class_weights, inference_blocks, load_clouds  # noqa: E402
from semseg_models import build_model  # noqa: E402


@torch.no_grad()
def predict_cloud(model, cloud, sc, device, n_cls, batch=16):
    """Sliding blocks; class probabilities averaged per point; argmax."""
    model.eval()
    prob = np.zeros((len(cloud), n_cls), np.float32)
    hits = np.zeros(len(cloud), np.int32)
    rng = np.random.default_rng(0)
    N = sc["block_points"]
    chunks, owners = [], []
    for cx, cy in inference_blocks(cloud, sc["block_size"], sc["infer_stride"]):
        idx = cloud.block_indices(cx, cy, sc["block_size"])
        if len(idx) == 0:
            continue
        idx = idx[rng.permutation(len(idx))]
        for s in range(0, len(idx), N):
            part = idx[s:s + N]
            pad = rng.choice(idx, N - len(part), replace=True) if len(part) < N else np.zeros(0, int)
            sel = np.concatenate([part, pad])
            chunks.append(cloud.features(sel, cx, cy).T)
            owners.append(part)
            if len(chunks) == batch:
                _flush(model, chunks, owners, prob, hits, device)
                chunks, owners = [], []
    if chunks:
        _flush(model, chunks, owners, prob, hits, device)
    miss = hits == 0
    pred = prob.argmax(1)
    if miss.any() and (~miss).any():                      # never covered: copy the nearest covered point
        from scipy.spatial import cKDTree
        _, j = cKDTree(cloud.xyz[~miss]).query(cloud.xyz[miss])
        pred[miss] = pred[~miss][j]
    return pred.astype(np.int8)


def _flush(model, chunks, owners, prob, hits, device):
    x = torch.from_numpy(np.stack(chunks)).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        p = F.softmax(model(x).float(), 1).cpu().numpy()
    for b, part in enumerate(owners):
        prob[part] += p[b, :, :len(part)].T
        hits[part] += 1


def evaluate(model, clouds, sc, device, n_cls):
    cm = np.zeros((n_cls, n_cls), np.int64)
    preds = {}
    for c in clouds:
        pr = predict_cloud(model, c, sc, device, n_cls)
        preds[c.id] = pr
        cm += confusion(c.cls, pr.astype(np.int64), n_cls)
    return semantic_iou(None, None, n_cls, cm=cm), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, choices=["pointnet", "pointnet2"])
    ap.add_argument("--fold", required=True, help="fold index or 'all'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--iters", type=int, default=None, help="override semseg.iterations (smoke test)")
    ap.add_argument("--block-points", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    sc = dict(cfg["semseg"])
    if a.block_points:
        sc["block_points"] = a.block_points
    iters = a.iters or sc["iterations"]
    n_cls = len(cfg["brpcd"]["classes"])
    set_seed(cfg["seed"])
    split = load_json(os.path.join(a.data, "split.json"))
    if a.fold == "all":
        fold = {"fold": "all", "train": split["all"], "val": [], "test": []}
    else:
        fold = split["folds"][int(a.fold)]
    out = os.path.join(a.out, a.model, f"fold_{a.fold}")
    os.makedirs(os.path.join(out, "pred"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cell = sc["block_size"] / 4.0
    log(f"model={a.model} fold={a.fold} device={device} train={len(fold['train'])} val={len(fold['val'])} "
        f"test={len(fold['test'])} iters={iters}")
    train = load_clouds(a.data, fold["train"], cell)
    val = load_clouds(a.data, fold["val"], cell)
    test = load_clouds(a.data, fold["test"], cell)
    w = class_weights(class_counts(train, n_cls), sc["class_weight_power"])
    log(f"class weights {np.round(w, 3).tolist()}")
    model = build_model(a.model, N_FEAT, n_cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=sc["lr"], weight_decay=sc["weight_decay"])
    warm = sc.get("warmup_iters", 500)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(it, iters) / iters)))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and sc.get("amp", True))
    loader = DataLoader(BlockSampler(train, sc["block_size"], sc["block_points"], sc.get("augment"), cfg["seed"]),
                        batch_size=sc["batch_size"], num_workers=a.workers if a.workers is not None else sc["num_workers"],
                        pin_memory=device.type == "cuda")
    wt = torch.from_numpy(w).to(device)
    best, best_it, log_rows = -1.0, -1, []
    t0 = time.time()
    it = 0
    run_loss = 0.0
    for x, y in loader:
        if it >= iters:
            break
        model.train()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda" and sc.get("amp", True)):
            loss = F.cross_entropy(model(x).float(), y, weight=wt)      # Eq. (3)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run_loss += float(loss)
        it += 1
        if it % 100 == 0 or it == iters:
            log(f"it {it:6d}/{iters} loss {run_loss / min(it, 100):.4f} lr {sched.get_last_lr()[0]:.2e} "
                f"({(time.time() - t0) / it:.3f} s/it)")
            run_loss = 0.0
        if val and (it % sc["val_every"] == 0 or it == iters):
            res, _ = evaluate(model, val, sc, device, n_cls)
            log_rows.append({"iter": it, "val_miou": res["miou"]})
            log(f"   val mIoU {res['miou']:.4f}")
            if res["miou"] > best:
                best, best_it = res["miou"], it
                torch.save(model.state_dict(), os.path.join(out, "best.pt"))
    train_s = time.time() - t0
    if not val or best_it < 0:
        torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        best_it = it
    with open(os.path.join(out, "train_log.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=["iter", "val_miou"])
        wri.writeheader()
        wri.writerows(log_rows)
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device, weights_only=True))
    result = {"model": a.model, "fold": a.fold, "best_iter": best_it, "best_val_miou": best, "train_seconds": train_s,
              "class_weights": w.tolist(), "per_cloud": {}}
    for name, group in (("val", val), ("test", test)):
        if not group:
            continue
        res, preds = evaluate(model, group, sc, device, n_cls)
        result[f"{name}_pooled"] = res
        for c in group:
            np.savez_compressed(os.path.join(out, "pred", f"{c.id}.npz"), pred=preds[c.id])
            if name == "test":
                result["per_cloud"][c.id] = semantic_iou(c.cls, preds[c.id].astype(np.int64), n_cls)
        log(f"{name}: mIoU {res['miou']:.4f}  IoU {np.round(res['iou'], 3).tolist()}")
    save_json(result, os.path.join(out, "metrics_semseg.json"))
    log(f"done: {out}")


if __name__ == "__main__":
    main()
