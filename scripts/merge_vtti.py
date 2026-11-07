#!/usr/bin/env python
"""
merge_vtti.py -- RQ3: member instances on the scaled 2DGS points of VTTI (paper Section 3.4);
--method vggt repeats it on the feed-forward points.

The metric cloud from measure_vtti.py (s along the axis, t across, h up) is cropped to the
bridge, put in the BrPCD convention (x = axis, z up, min z = 0), downsampled to the BrPCD
voxel, labelled by the Track A network trained on every BrPCD cloud, and split into members
by M1 and M2 with the parameters tuned in Track A (per-class median over folds for M1, the
most frequent combination for M2). Outputs girder and pier counts, span lengths and girder
spacing, a coloured PLY for CloudCompare and the npz used by make_figures.py.

    python merge_vtti.py --trackb $RESULTS_DIR/trackB --brpcd $BRPCD_DIR \
        --semseg-root $RESULTS_DIR/trackA/semseg --instances-root $RESULTS_DIR/trackA/instances \
        --out $RESULTS_DIR/rq3
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster_m1 import run_m1  # noqa: E402
from common import FT_TO_M, load_cfg, load_json, load_yaml, log, m1_default, pkg_path, save_json  # noqa: E402
from geometry import estimate_normals, voxel_downsample_idx  # noqa: E402
from layout_m2 import run_m2  # noqa: E402


def tuned_params(inst_root, model, cfg):
    files = sorted(glob.glob(os.path.join(inst_root, model, "fold_*", "tuned.json")))
    classes = cfg["brpcd"]["classes"]
    m1 = {i: m1_default(cfg) for i in range(len(classes))}
    m2 = {k: v for k, v in cfg["m2"].items() if k != "tune"}
    if not files:
        log("no tuned.json found -> experiment.yaml defaults")
        return m1, m2, 0
    T = [load_json(f) for f in files]
    for i, c in enumerate(classes):
        vals = [t["m1"][c] for t in T if c in t.get("m1", {})]
        if vals:
            m1[i] = (float(np.median([v[0] for v in vals])), int(np.median([v[1] for v in vals])))
    combos = collections.Counter(tuple(sorted(t["m2"].items())) for t in T if t.get("m2"))
    if combos:
        m2.update(dict(combos.most_common(1)[0][0]))
    return m1, m2, len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackb", required=True)
    ap.add_argument("--method", default="2dgs")
    ap.add_argument("--brpcd", required=True)
    ap.add_argument("--semseg-root", required=True)
    ap.add_argument("--instances-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--true-labels", default=None, help="npz with per-point cls (synthetic test only)")
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    mg, mc = cfg["merge"], cfg["measure"]
    classes = cfg["brpcd"]["classes"]
    ci = {c: i for i, c in enumerate(classes)}
    os.makedirs(a.out, exist_ok=True)
    z = np.load(os.path.join(a.trackb, "measure", f"aligned_{a.method}.npz"))
    sth = z["stha"].astype(np.float64)
    st, hw, gh = z["stations_m"], float(z["half_width_m"]), float(z["ground_h_m"])
    m = ((sth[:, 0] > st[0] - mg["crop_margin_m"]) & (sth[:, 0] < st[-1] + mg["crop_margin_m"]) &
         (np.abs(sth[:, 1]) < hw + mg["crop_margin_m"]) & (sth[:, 2] > gh + mc["ground_clearance_m"]))
    xyz = sth[m].copy()
    xyz[:, 2] -= xyz[:, 2].min()
    voxel = load_json(os.path.join(a.brpcd, "convert_summary.json"))["voxel"]
    keep = voxel_downsample_idx(xyz, voxel)
    xyz = xyz[keep]
    nrm = estimate_normals(xyz, cfg["brpcd"].get("normals_k", 16))
    log(f"cropped {m.sum():,} -> {len(xyz):,} points at voxel {voxel} m")
    cloud_path = os.path.join(a.out, "vtti_cloud.npz")
    np.savez_compressed(cloud_path, xyz=xyz.astype(np.float32), normal=nrm, cls=np.zeros(len(xyz), np.int8),
                        inst=np.zeros(len(xyz), np.int32))
    if a.true_labels:                                     # synthetic test: skip the network
        tl = np.load(a.true_labels)["cls"][z["idx"]][m][keep]
        pred = tl.astype(np.int64)
        src = "true labels (synthetic test)"
    else:
        import torch
        from semseg_data import N_FEAT, Cloud
        from semseg_models import build_model
        from semseg_train import predict_cloud
        sc = cfg["semseg"]
        ck = os.path.join(a.semseg_root, mg["semseg_model"], f"fold_{mg['checkpoint']}", "best.pt")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = build_model(mg["semseg_model"], N_FEAT, len(classes)).to(dev)
        net.load_state_dict(torch.load(ck, map_location=dev, weights_only=True))
        pred = predict_cloud(net, Cloud(cloud_path, sc["block_size"] / 4.0), sc, dev, len(classes)).astype(np.int64)
        src = ck
    m1p, m2p, n_t = tuned_params(a.instances_root, mg["semseg_model"], cfg)
    i1 = run_m1(xyz, pred, m1p, base_voxel=voxel)
    units = {"girder": "whole", "deck": "whole", "pier": "bent", "main_cable": "side"}
    i2, info, layout = run_m2(xyz, pred, ci, m2p, units)
    ms = cfg["metrics"]["min_segment_points"]

    def count(inst, c):
        ids, cnt = np.unique(inst[(pred == ci[c]) & (inst >= 0)], return_counts=True)
        return int((cnt >= ms).sum())

    ov = load_yaml(a.overrides or pkg_path(mc["overrides"])) if (a.overrides or mc.get("overrides")) else {}
    ref_g = ov.get("girder_count_ref") or mg["ref_counts"].get("girders")
    spans_ref = [v * FT_TO_M for v in cfg["vtti"]["spans_ft"]]
    stn = np.array(info["stations"])
    spans = np.diff(stn).tolist() if len(stn) == 4 else None
    res = {"method": a.method, "labels_from": src, "n_points": int(len(xyz)), "tuned_from_folds": n_t,
           "class_share": {c: float((pred == i).mean()) for c, i in ci.items()},
           "M2": {"girders": info["n_girder_lines"], "piers": info["n_piers"], "stations_m": stn.tolist(),
                  "spans_m": spans,
                  "rel_err": None if spans is None else [abs(spans[k] - spans_ref[k]) / spans_ref[k] for k in range(3)],
                  "girder_spacing_m": np.diff(info["offsets"]).tolist(), "girder_offsets_m": info["offsets"]},
           "M1": {"girders": count(i1, "girder"), "piers": count(i1, "pier")},
           "reference": {"girders": ref_g, "piers": mg["ref_counts"]["piers"], "spans_m": spans_ref},
           "m1_params": {classes[k]: list(v) for k, v in m1p.items()},
           "m2_params": {k: m2p[k] for k in (cfg["m2"].get("tune") or {}) if k in m2p}}
    save_json(res, os.path.join(a.out, "merge.json"))
    np.savez_compressed(os.path.join(a.out, "vtti_instances.npz"), xyz=xyz.astype(np.float32), cls=pred.astype(np.int8),
                        m1=i1.astype(np.int32), m2=i2.astype(np.int32))
    try:
        from pcio import write_ply
        rng = np.random.default_rng(1)
        pal = rng.integers(40, 230, size=(max(int(i2.max()) + 2, 2), 3)).astype(np.uint8)
        col = np.where((i2 >= 0)[:, None], pal[np.maximum(i2, 0)], np.uint8(160))
        write_ply(os.path.join(a.out, "vtti_m2_instances.ply"), xyz, col.astype(np.uint8),
                  {"cls": pred.astype(np.int32), "m2": i2.astype(np.int32), "m1": i1.astype(np.int32)})
    except Exception as e:  # noqa: BLE001
        log(f"PLY export skipped: {e}")
    log(f"RQ3: M2 girders {res['M2']['girders']} piers {res['M2']['piers']} spans {np.round(spans, 2).tolist() if spans else None}"
        f" | M1 girders {res['M1']['girders']} piers {res['M1']['piers']} | reference girders {ref_g} piers {res['reference']['piers']}")


if __name__ == "__main__":
    main()
