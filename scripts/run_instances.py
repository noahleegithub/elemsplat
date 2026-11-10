#!/usr/bin/env python
"""
run_instances.py -- M1 and M2 on one fold (paper Sections 3.3 and 4.4).

Classes come from a semantic network (--classes pred --semseg-model pointnet2) or from the
ground truth (--classes true, the control that isolates the instance step). M1 is tuned per
class and M2 on its grid, both on the fold's validation bridges, then frozen and applied to
the test bridges.

    python run_instances.py --data $BRPCD_DIR --fold 3 --classes pred --semseg-model pointnet2 \
        --semseg-root $RESULTS_DIR/trackA/semseg --out $RESULTS_DIR/trackA/instances
Writes <out>/<source>/fold_<k>/<bridge>.json (+ tuned.json), source = model name or "true".
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_m1 import labels_from_prepared, prepare_thinning, run_m1  # noqa: E402
from common import load_cfg, load_json, log, m1_default, save_json  # noqa: E402
from layout_m2 import run_m2  # noqa: E402
from metrics import panoptic_quality  # noqa: E402


def load(data, bridge, classes, semseg_dir):
    z = np.load(os.path.join(data, "clouds", f"{bridge}.npz"))
    meta = load_json(os.path.join(data, "clouds", f"{bridge}.json"))
    xyz = z["xyz"].astype(np.float64)
    gt_cls, gt_inst = z["cls"].astype(np.int64), z["inst"].astype(np.int64)
    if classes == "true":
        cls = gt_cls
    else:
        p = os.path.join(semseg_dir, "pred", f"{bridge}.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} missing -- run 03a_semseg for this fold first")
        cls = np.load(p)["pred"].astype(np.int64)
    return {"id": bridge, "xyz": xyz, "cls": cls, "gt_cls": gt_cls, "gt_inst": gt_inst, "meta": meta}


def pq_of(c, inst, classes, mcfg, only=None):
    cl = classes if only is None else [only]
    return panoptic_quality(c["gt_cls"], c["gt_inst"], c["cls"], inst, cl, mcfg["pq_iou"], mcfg["min_segment_points"])


def tune_m1(val, cfg, n_cls, voxel):
    """Grid search per class, ordered eps-outer so the thinning is built once per (class, eps).

    Building it per grid cell instead re-queried every point of every validation cloud 24 times,
    which is what stalled the largest cable-stayed fold past 12 h with no output.
    """
    grid = list(itertools.product(cfg["m1"]["eps_grid"], cfg["m1"]["min_pts_grid"]))
    n_cells = len(grid)
    best = {}
    for c in range(n_cls):
        clouds = [v for v in val if (v["gt_cls"] == c).any()]
        if not clouds:
            best[c] = m1_default(cfg)
            log(f"  tune M1 class {c}: no validation cloud labels it, defaults kept")
            continue
        scores, done = [], 0
        for eps in sorted({g[0] for g in grid}):
            preps = [prepare_thinning(v["xyz"][v["cls"] == c], eps, voxel) for v in clouds]
            for mp in sorted({g[1] for g in grid if g[0] == eps}):
                vals = []
                for v, prep in zip(clouds, preps):
                    inst = np.full(len(v["xyz"]), -1, np.int64)
                    m = v["cls"] == c
                    lab = labels_from_prepared(v["xyz"][m], prep, eps, mp)
                    good = lab >= 0
                    if good.any():
                        u, inv = np.unique(lab[good], return_inverse=True)
                        inst[np.flatnonzero(m)[good]] = np.asarray(inv).ravel()
                    r = pq_of(v, inst, list(range(n_cls)), cfg["metrics"], only=c)
                    if np.isfinite(r["PQ"]):
                        vals.append(r["PQ"])
                scores.append((np.mean(vals) if vals else -1, eps, mp))
                done += 1
                log(f"  tune M1 class {c} cell {done}/{n_cells} eps={eps} min_pts={mp} "
                    f"PQ={scores[-1][0]:.4f}")
        scores.sort(key=lambda t: -t[0])
        best[c] = (scores[0][1], scores[0][2]) if scores[0][0] >= 0 else m1_default(cfg)
    return best


def tune_m2(val, cfg, ci, n_cls):
    base = dict(cfg["m2"])
    tune = base.pop("tune", {}) or {}
    keys = list(tune)
    combos = list(itertools.product(*[tune[k] for k in keys])) or [()]
    best, best_s = base, -1.0
    for combo in combos:
        p = dict(base, **dict(zip(keys, combo)))
        vals = []
        for v in val:
            try:
                inst, _, _ = run_m2(v["xyz"], v["cls"], ci, p, v["meta"]["units"])
            except ValueError:
                continue
            r = pq_of(v, inst, list(range(n_cls)), cfg["metrics"])
            if np.isfinite(r["PQ"]):
                vals.append(r["PQ"])
        s = float(np.mean(vals)) if vals else -1.0
        if s > best_s:
            best, best_s = p, s
    return best, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--fold", required=True, type=int)
    ap.add_argument("--classes", choices=["pred", "true"], required=True)
    ap.add_argument("--semseg-model", default="pointnet2")
    ap.add_argument("--semseg-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--no-tune", action="store_true", help="use the defaults of experiment.yaml (smoke test)")
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    classes = cfg["brpcd"]["classes"]
    ci = {c: i for i, c in enumerate(classes)}
    n_cls = len(classes)
    split = load_json(os.path.join(a.data, "split.json"))
    fold = split["folds"][a.fold]
    src = "true" if a.classes == "true" else a.semseg_model
    semseg_dir = os.path.join(a.semseg_root or "", a.semseg_model, f"fold_{a.fold}") if a.classes == "pred" else None
    out = os.path.join(a.out, src, f"fold_{a.fold}")
    os.makedirs(out, exist_ok=True)
    voxel = load_json(os.path.join(a.data, "convert_summary.json"))["voxel"]
    t0 = time.time()
    val = [load(a.data, b, a.classes, semseg_dir) for b in fold["val"]]
    test = [load(a.data, b, a.classes, semseg_dir) for b in fold["test"]]
    if a.no_tune or not val:
        m1p = {c: m1_default(cfg) for c in range(n_cls)}
        m2p, m2s = {k: v for k, v in cfg["m2"].items() if k != "tune"}, float("nan")
    else:
        m1p = tune_m1(val, cfg, n_cls, voxel)
        m2p, m2s = tune_m2(val, cfg, ci, n_cls)
    save_json({"m1": {classes[c]: list(v) for c, v in m1p.items()},
               "m2": {k: m2p[k] for k in (cfg["m2"].get("tune") or {})} if not a.no_tune else {},
               "m2_val_pq": m2s, "tune_seconds": time.time() - t0}, os.path.join(out, "tuned.json"))
    log(f"fold {a.fold} [{src}] tuned in {time.time() - t0:.0f} s; M1 {m1p}; M2 val PQ {m2s:.3f}")
    for c in test:
        i1 = run_m1(c["xyz"], c["cls"], m1p, base_voxel=voxel)
        r1 = pq_of(c, i1, list(range(n_cls)), cfg["metrics"])
        try:
            i2, info, _ = run_m2(c["xyz"], c["cls"], ci, m2p, c["meta"]["units"])
            r2 = pq_of(c, i2, list(range(n_cls)), cfg["metrics"])
            r2["failed"] = False
        except ValueError as e:
            # A layout-fit failure is NOT a zero: leave the scores None so aggregate.py drops the
            # bridge from M2 instead of averaging a spurious 0 into the PQ curve and the H1 trend.
            i2, info = None, {"error": str(e)}
            r2 = {"PQ": None, "SQ": None, "RQ": None, "per_class": {}, "N": r1["N"], "failed": True}
            log(f"  WARNING {c['id']}: M2 fit failed ({e}) -- excluded from M2 means and from H1")
        rec = {"bridge": c["id"], "type": c["meta"]["type"], "is_real": c["meta"]["is_real"],
               "group": c["meta"]["source"], "fold": a.fold, "classes": src, "N": r1["N"],
               "units": c["meta"]["units"], "M1": r1, "M2": r2, "M2_layout": info}
        save_json(rec, os.path.join(out, f"{c['id']}.json"))
        np.savez_compressed(os.path.join(out, f"{c['id']}_inst.npz"), m1=i1.astype(np.int32),
                            m2=(i2 if i2 is not None else np.full(len(i1), -1)).astype(np.int32))
        log(f"  {c['id']:36s} N={r1['N']:4d}  M1 PQ {r1['PQ']:.3f}   M2 PQ "
            f"{'FAILED' if r2['PQ'] is None else format(r2['PQ'], '.3f')}")


if __name__ == "__main__":
    main()
