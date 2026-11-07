#!/usr/bin/env python
"""
brpcd_convert.py -- BrPCD member files -> one cloud per bridge (paper Section 3.2, "Data and
bridge frame").

For every bridge: read the member files (one file = one instance), map folder/file names to
the six classes, scale to metres, voxel-downsample (one random point per voxel keeps exact
labels), rotate the leading horizontal principal direction onto x, estimate normals and
infer the instance unit of each class from the ground truth. Output per bridge:
    <out>/clouds/<bridge>.npz   xyz float32, normal float32, cls int8, inst int32
    <out>/clouds/<bridge>.json  meta (type, real/variant, source scan, counts, units, transform)
and <out>/index.csv over all bridges.

    python brpcd_convert.py --raw $BRPCD_RAW --out $BRPCD_DIR --workers 8
"""
import argparse
import csv
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brpcd_io import bridge_type, class_of, find_bridge_dirs, member_files  # noqa: E402
from common import load_cfg, load_json, load_yaml, log, pkg_path, save_json  # noqa: E402
from geometry import estimate_normals, median_spacing, voxel_downsample_idx  # noqa: E402
from layout_m2 import infer_units  # noqa: E402
from pcio import read_points  # noqa: E402

CFG = None


def _auto_voxel(bridges, cfg):
    bc = cfg["brpcd"]
    meds = []
    for b in bridges:
        if not b["is_real"]:
            continue
        for f in b["members"][:3]:
            try:
                x = read_points(f)["xyz"] * bc["unit_scale"]
                if len(x) > 50:
                    meds.append(median_spacing(x))
            except Exception as e:  # noqa: BLE001
                log(f"  voxel probe failed on {f}: {e}")
    if not meds:
        return bc.get("voxel_size_min", 0.02)
    v = round(2.0 * float(np.median(meds)), 2)
    return float(np.clip(v, bc.get("voxel_size_min", 0.02), bc.get("voxel_size_max", 0.2)))


def _guess_sources(bridges):
    """Map every variant to the real scan it was generated from (same type, closest name)."""
    reals = [b for b in bridges if b["is_real"]]
    norm = lambda n: re.sub(r"[-_]?f[-_]", "-", n.lower())
    for b in bridges:
        if b["is_real"]:
            b["source"], b["source_conf"] = b["id"], "self"
            continue
        cands = [r for r in reals if r["type"] == b["type"]] or reals
        nb = norm(b["id"])
        digits = re.findall(r"\d+", nb)

        def score(r):
            nr = norm(r["id"])
            pre = len(os.path.commonprefix([nr, nb]))
            dig = 5 * len(set(re.findall(r"\d+", nr)) & set(digits))
            return pre + dig
        ranked = sorted(cands, key=score, reverse=True)
        b["source"] = ranked[0]["id"] if ranked else b["id"]
        b["source_conf"] = "unique" if len(cands) == 1 else ("name" if len(ranked) < 2 or score(ranked[0]) > score(ranked[1]) else "tie")
    return bridges


def _convert_one(b):
    cfg = CFG
    bc, ci = cfg["brpcd"], {c: i for i, c in enumerate(cfg["brpcd"]["classes"])}
    out_npz = os.path.join(b["out"], "clouds", b["id"] + ".npz")
    if os.path.exists(out_npz) and not b.get("force"):
        return load_json(out_npz.replace(".npz", ".json"))
    xs, cs, iss, dropped, skipped = [], [], [], [], []
    for k, f in enumerate(b["members"]):
        c = b["classes"][k]
        if c in (None, "__ignore__"):
            skipped.append(os.path.relpath(f, b["dir"]))
            continue
        try:
            x = read_points(f)["xyz"] * bc["unit_scale"]
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{os.path.relpath(f, b['dir'])} ({e})")
            continue
        xs.append(x)
        cs.append(np.full(len(x), ci[c], np.int8))
        iss.append(np.full(len(x), k, np.int32))
    if not xs:
        return {"id": b["id"], "error": "no readable member files", "skipped": skipped}
    xyz, cls, inst = np.concatenate(xs), np.concatenate(cs), np.concatenate(iss)
    n_raw = len(xyz)
    keep = voxel_downsample_idx(xyz, b["voxel"], seed=0)
    xyz, cls, inst = xyz[keep], cls[keep], inst[keep]
    ids, cnt = np.unique(inst, return_counts=True)
    small = ids[cnt < bc.get("min_instance_points", 30)]
    if len(small):
        dropped = [os.path.relpath(b["members"][int(i)], b["dir"]) for i in small]
        m = ~np.isin(inst, small)
        xyz, cls, inst = xyz[m], cls[m], inst[m]
    # horizontal PCA: leading direction of the xy coordinates onto +x, centroid to 0, min z to 0
    sub = xyz[:: max(1, len(xyz) // 200000), :2]
    c2 = sub.mean(0)
    _, _, vt = np.linalg.svd(sub - c2, full_matrices=False)
    ux = vt[0] / np.linalg.norm(vt[0])
    R = np.array([[ux[0], ux[1], 0.0], [-ux[1], ux[0], 0.0], [0.0, 0.0, 1.0]])
    t = np.array([c2[0], c2[1], 0.0])
    xyz = (xyz - t) @ R.T
    zmin = float(xyz[:, 2].min())
    xyz[:, 2] -= zmin
    _, inst = np.unique(inst, return_inverse=True)
    inst = np.asarray(inst).ravel().astype(np.int32)
    nrm = estimate_normals(xyz, bc.get("normals_k", 16))
    units = infer_units(xyz, cls.astype(np.int64), inst.astype(np.int64), ci, cfg["m2"])
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, xyz=xyz.astype(np.float32), normal=nrm.astype(np.float32),
                        cls=cls, inst=inst)
    per_class = {cn: int(len(np.unique(inst[cls == i]))) for cn, i in ci.items()}
    meta = {"id": b["id"], "type": b["type"], "is_real": b["is_real"], "source": b["source"],
            "source_conf": b["source_conf"], "n_raw": int(n_raw), "n_points": int(len(xyz)),
            "voxel": b["voxel"], "n_instances": int(inst.max() + 1) if len(inst) else 0,
            "instances_per_class": per_class, "units": units,
            "transform": {"R": R.tolist(), "t": t.tolist(), "z_shift": -zmin, "unit_scale": bc["unit_scale"]},
            "extent": (xyz.max(0) - xyz.min(0)).round(2).tolist(),
            "dropped_small_members": dropped, "skipped_files": skipped}
    save_json(meta, out_npz.replace(".npz", ".json"))
    return meta


def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    CFG = load_cfg(a.cfg)
    bc = CFG["brpcd"]
    cp = load_yaml(pkg_path(bc["class_patterns"]))
    real_re = re.compile(bc["real_scan_regex"])
    bridges = []
    for bd in find_bridge_dirs(a.raw, bc["point_extensions"]):
        members, _ = member_files(bd, bc["point_extensions"])
        classes = [class_of(os.path.relpath(f, bd), cp["patterns"], cp["order"], cp.get("ignore", [])) for f in members]
        name = os.path.basename(bd)
        bridges.append({"id": name, "dir": bd, "type": bridge_type(bd, a.raw), "is_real": bool(real_re.search(name)),
                        "members": members, "classes": classes, "out": a.out, "force": a.force})
    ids = [b["id"] for b in bridges]
    for b in bridges:                                  # make ids unique across type folders
        if ids.count(b["id"]) > 1:
            b["id"] = f"{b['type']}_{b['id']}"
    _guess_sources(bridges)
    override = pkg_path("configs/brpcd_groups.csv")
    if override.exists():                              # optional manual variant -> source table
        with open(override) as f:
            table = {r["bridge"]: r["source"] for r in csv.DictReader(f)}
        for b in bridges:
            if b["id"] in table:
                b["source"], b["source_conf"] = table[b["id"]], "manual"
    vox = bc["voxel_size"]
    vox = _auto_voxel(bridges, CFG) if vox == "auto" else float(vox)
    log(f"{len(bridges)} bridges, voxel = {vox:.3f} m")
    for b in bridges:
        b["voxel"] = vox
    with Pool(max(1, a.workers)) as pool:
        metas = pool.map(_convert_one, bridges, chunksize=1)
    rows = []
    for m in metas:
        if "error" in m:
            log(f"  FAILED {m['id']}: {m['error']}")
            continue
        rows.append({"bridge": m["id"], "type": m["type"], "is_real": int(m["is_real"]), "source": m["source"],
                     "source_conf": m["source_conf"], "n_points": m["n_points"], "n_instances": m["n_instances"],
                     **{f"n_{k}": v for k, v in m["instances_per_class"].items()},
                     **{f"unit_{k}": v for k, v in m["units"].items()}})
        log(f"  {m['id']:40s} {m['type']:12s} real={m['is_real']!s:5s} pts={m['n_points']:>9,d} "
            f"inst={m['n_instances']:4d} src={m['source']} ({m['source_conf']})")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_json({"voxel": vox, "n_bridges": len(rows), "n_real": sum(r["is_real"] for r in rows)},
              os.path.join(a.out, "convert_summary.json"))
    ties = [r["bridge"] for r in rows if r["source_conf"] == "tie"]
    if ties:
        log(f"WARNING: source scan of {len(ties)} variants is ambiguous: {ties[:6]} ... "
            "write configs/brpcd_groups.csv (columns bridge,source) and rerun with --force")
    log(f"index written: {os.path.join(a.out, 'index.csv')}")


if __name__ == "__main__":
    main()
