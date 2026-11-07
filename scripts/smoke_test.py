#!/usr/bin/env python
"""
smoke_test.py -- end-to-end test of the analysis code on SYNTHETIC bridges (CPU, ~10-20 min).

Runs every Python step that does not need the GPU renderers: BrPCD-like tree -> inspect ->
convert -> splits -> PointNet (a few iterations) -> M1/M2 on predicted and true classes ->
synthetic three-span bridge -> provisional scale -> measurement -> RQ3 merge -> fake VGGT
predictions (rendered depth maps in a random frame) -> alignment -> measurement -> merge ->
aggregate -> tables -> figures (stamped "SYNTHETIC SMOKE TEST"). The numbers say nothing about the
hypotheses; they only show that the pipeline runs and the outputs have the expected form.

    python smoke_test.py --out $SMOKE_DIR
"""
import argparse
import copy
import os
import subprocess
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synth_bridges as sb  # noqa: E402
from bridge_frame import provisional  # noqa: E402
from common import FT_TO_M, load_cfg, log, save_json  # noqa: E402

STAMP = "SYNTHETIC SMOKE TEST - NOT RESULTS"


def run(args, cfg_path):
    cmd = [sys.executable, os.path.join(HERE, args[0])] + args[1:] + ["--cfg", cfg_path]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-4:])
    log(f"{'OK  ' if r.returncode == 0 else 'FAIL'} {args[0]} ({time.time() - t0:.0f} s)\n{tail}")
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        sys.exit(f"smoke test failed at {args[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-real", type=int, default=6)
    ap.add_argument("--n-variants", type=int, default=3)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--paper-out", default=None, help="also write tables/figures here")
    a = ap.parse_args()
    t_all = time.time()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_cfg()
    sm = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
    sm["semseg"].update({"models": ["pointnet"], "block_points": 512, "batch_size": 8, "iterations": a.iters,
                         "val_every": a.iters, "warmup_iters": 5, "num_workers": 0})
    sm["m1"].update({"eps_grid": [0.3, 0.8], "min_pts_grid": [5, 10]})
    sm["m2"]["tune"] = {"kde_bw_offset": [0.15, 0.25], "peak_eta": [0.15]}
    sm["merge"]["semseg_model"] = "pointnet"
    sm["vggt"].update({"max_frames": 40, "max_points": 400000})
    sm["brpcd"]["class_patterns"] = os.path.join(os.path.dirname(HERE), "configs", "brpcd_classes.yaml")
    sm["measure"]["overrides"] = None
    cfg_path = os.path.join(a.out, "experiment_smoke.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(sm, f, sort_keys=False)

    raw, data, res = (os.path.join(a.out, d) for d in ("brpcd_raw", "brpcd", "results"))
    if not os.path.isdir(raw):
        names = sb.write_brpcd_tree(raw, n_real=a.n_real, n_variants=a.n_variants, seed=0)
        log(f"synthetic BrPCD-like tree: {len(names)} clouds")
    run(["brpcd_inspect.py", "--raw", raw, "--out", data, "--read-limit", "1"], cfg_path)
    run(["brpcd_convert.py", "--raw", raw, "--out", data, "--workers", "2"], cfg_path)
    run(["make_splits.py", "--data", data], cfg_path)
    split = yaml.safe_load(open(os.path.join(data, "split.json")))
    folds = list(range(len(split["folds"])))
    for k in folds + ["all"]:
        if os.path.exists(os.path.join(res, "trackA", "semseg", "pointnet", f"fold_{k}", "metrics_semseg.json")):
            log(f"skip semseg fold {k} (done)")
            continue
        run(["semseg_train.py", "--data", data, "--model", "pointnet", "--fold", str(k),
             "--out", os.path.join(res, "trackA", "semseg"), "--workers", "0"], cfg_path)
    for k in folds:
        run(["run_instances.py", "--data", data, "--fold", str(k), "--classes", "pred", "--semseg-model", "pointnet",
             "--semseg-root", os.path.join(res, "trackA", "semseg"), "--out", os.path.join(res, "trackA", "instances")], cfg_path)
        run(["run_instances.py", "--data", data, "--fold", str(k), "--classes", "true",
             "--out", os.path.join(res, "trackA", "instances")], cfg_path)

    # Track B on a synthetic three-span bridge (two noise levels, NOT 3DGS/2DGS)
    tb = os.path.join(res, "trackB")
    os.makedirs(tb, exist_ok=True)
    xyz, cc, cv, truth, cls, inst = sb.vtti_like_scene(seed=3, scale=0.37)
    rng = np.random.default_rng(0)
    prov = provisional(xyz[rng.permutation(len(xyz))[:30000]], cv, sum(sm["vtti"]["spans_ft"]) * FT_TO_M, sm["measure"])
    save_json(prov, os.path.join(tb, "scale0.json"))
    np.savez(os.path.join(tb, "views.npz"), centres=cc, dirs=cv)
    np.savez(os.path.join(tb, "labels.npz"), cls=cls, inst=inst)
    noisy = xyz + rng.normal(0, 0.03 * truth["scale"], xyz.shape)
    for name, label, pts in (("synthA", "synthetic, 1 cm noise", xyz), ("synthB", "synthetic, 3 cm noise", noisy)):
        np.savez(os.path.join(tb, f"pts_{name}.npz"), xyz=pts.astype(np.float32))
        run(["measure_vtti.py", "--points", os.path.join(tb, f"pts_{name}.npz"), "--method", name, "--label", label,
             "--viewdirs", os.path.join(tb, "views.npz"), "--trackb", tb], cfg_path)
    run(["merge_vtti.py", "--trackb", tb, "--method", "synthA", "--brpcd", data,
         "--semseg-root", os.path.join(res, "trackA", "semseg"), "--instances-root", os.path.join(res, "trackA", "instances"),
         "--out", os.path.join(res, "rq3"), "--true-labels", os.path.join(tb, "labels.npz")], cfg_path)

    # fake VGGT: depth maps rendered from the clean cloud in a random frame -> vggt_points.py (no GPU, --predictions)
    names = [f"img_{k:03d}.jpg" for k in range(len(cc))]
    sb.write_colmap_txt_model(os.path.join(tb, "fake_scene", "sparse", "0"), names, cc, cv)
    sel = np.round(np.linspace(0, len(cc) - 1, sm["vggt"]["max_frames"])).astype(int)
    fake = sb.fake_vggt_predictions(xyz, [names[i] for i in sel], cc[sel], cv[sel], seed=1)
    os.makedirs(os.path.join(tb, "vggt"), exist_ok=True)
    np.savez_compressed(os.path.join(tb, "vggt", "predictions.npz"), **{k: v for k, v in fake.items() if k != "truth"})
    run(["vggt_points.py", "--scene", os.path.join(tb, "fake_scene"), "--out", os.path.join(tb, "vggt"),
         "--predictions", os.path.join(tb, "vggt", "predictions.npz"), "--points-out", os.path.join(tb, "pts_vggt.npz")], cfg_path)
    run(["measure_vtti.py", "--points", os.path.join(tb, "pts_vggt.npz"), "--method", "vggt", "--label", "fake VGGT (synthetic)",
         "--viewdirs", os.path.join(tb, "views.npz"), "--trackb", tb], cfg_path)
    from scipy.spatial import cKDTree
    pv = np.load(os.path.join(tb, "pts_vggt.npz"))["xyz"]
    _, nn = cKDTree(xyz).query(pv)                       # true labels for the unprojected pixels: nearest clean point
    np.savez(os.path.join(tb, "labels_vggt.npz"), cls=cls[nn], inst=inst[nn])
    run(["merge_vtti.py", "--trackb", tb, "--method", "vggt", "--brpcd", data,
         "--semseg-root", os.path.join(res, "trackA", "semseg"), "--instances-root", os.path.join(res, "trackA", "instances"),
         "--out", os.path.join(res, "rq3_vggt"), "--true-labels", os.path.join(tb, "labels_vggt.npz")], cfg_path)
    run(["aggregate.py", "--results", res, "--data", data], cfg_path)
    outs = [os.path.join(a.out, "paper_assets")] + ([a.paper_out] if a.paper_out else [])
    for o in outs:
        run(["make_tables.py", "--summary", os.path.join(res, "summary", "summary.json"), "--out", os.path.join(o, "tables")], cfg_path)
        run(["make_figures.py", "--results", res, "--data", data, "--out", os.path.join(o, "figures"), "--stamp", STAMP,
             "--example-source", "true"], cfg_path)
    log(f"SMOKE TEST PASSED in {(time.time() - t_all) / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
