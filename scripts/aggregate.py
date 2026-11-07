#!/usr/bin/env python
"""
aggregate.py -- collect every result file into summary/summary.json (+ instances_rows.csv).

Track A: semantic IoU pooled over the test clouds of all folds (real scans and variants kept
apart), PQ/SQ/RQ per method and class source, PQ-vs-log N slopes per bridge type and the
bridge-level bootstrap interval of slope(M2) - slope(M1) (H1). Track B: image metrics, run
info and measurements per reconstruction. RQ3: merge.json.

    python aggregate.py --results $RESULTS_DIR --data $BRPCD_DIR
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_cfg, load_json, log, save_json  # noqa: E402
from metrics import bootstrap_slope_diff, semantic_iou, trend_slope  # noqa: E402


def _index(data):
    p = os.path.join(data, "index.csv")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return {r["bridge"]: r for r in csv.DictReader(f)}


def semseg_summary(root, index, n_cls):
    out = {}
    for mdir in sorted(glob.glob(os.path.join(root, "trackA", "semseg", "*"))):
        model = os.path.basename(mdir)
        cm = {"real": np.zeros((n_cls, n_cls), np.int64), "variant": np.zeros((n_cls, n_cls), np.int64)}
        n_folds, secs = 0, []
        for f in sorted(glob.glob(os.path.join(mdir, "fold_*", "metrics_semseg.json"))):
            if f.endswith(os.path.join("fold_all", "metrics_semseg.json")):
                continue
            r = load_json(f)
            n_folds += 1
            secs.append(r.get("train_seconds") or 0)
            for b, v in r.get("per_cloud", {}).items():
                key = "real" if index.get(b, {}).get("is_real") == "1" else "variant"
                cm[key] += np.array(v["confusion"], np.int64)
        if n_folds:
            out[model] = {k: semantic_iou(None, None, n_cls, cm=c) for k, c in cm.items() if c.sum()}
            out[model]["n_folds"] = n_folds
            out[model]["train_hours_mean"] = float(np.mean(secs) / 3600) if secs else None
    return out


def instance_rows(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "trackA", "instances", "*", "fold_*", "*.json"))):
        if f.endswith("tuned.json"):
            continue
        r = load_json(f)
        for meth in ("M1", "M2"):
            m = r[meth]
            rows.append({"bridge": r["bridge"], "type": r["type"], "is_real": int(bool(r["is_real"])), "group": r["group"],
                         "fold": r["fold"], "classes": r["classes"], "method": meth, "N": r["N"],
                         "PQ": m["PQ"], "SQ": m["SQ"], "RQ": m["RQ"], "failed": int(m.get("PQ") is None),
                         **{f"PQ_c{c}": (v.get("PQ") if isinstance(v, dict) else None)
                            for c, v in m.get("per_class", {}).items()}})
    return rows


def instance_summary(rows):
    out = {}
    for src in sorted({r["classes"] for r in rows}):
        S = [r for r in rows if r["classes"] == src]
        out[src] = {}
        for meth in ("M1", "M2"):
            R = [r for r in S if r["method"] == meth]
            real = [r for r in R if r["is_real"]]
            var = [r for r in R if not r["is_real"]]
            agg = lambda L, k: float(np.mean([x[k] for x in L if x[k] is not None])) if L else None
            d = {"real": {k: agg(real, k) for k in ("PQ", "SQ", "RQ")}, "variant": {k: agg(var, k) for k in ("PQ", "SQ", "RQ")},
                 "n_real": len(real), "n_variant": len(var),
                 # fits that raised: counted, never averaged in as a zero
                 "n_failed": int(sum(x.get("failed", 0) for x in R)), "by_type": {}}
            for t in sorted({r["type"] for r in R}):
                T = [r for r in R if r["type"] == t]
                b1, b0 = trend_slope([r["N"] for r in T], [r["PQ"] for r in T])
                d["by_type"][t] = {"PQ_mean": agg(T, "PQ"), "slope": b1, "intercept": b0, "n": len(T),
                                   "N_range": [min(r["N"] for r in T), max(r["N"] for r in T)]}
            b1, b0 = trend_slope([r["N"] for r in R], [r["PQ"] for r in R])
            d["slope_all"], d["intercept_all"] = b1, b0
            out[src][meth] = d
        pair, n_pair_dropped = {}, 0
        for r in S:
            pair.setdefault(r["bridge"], {"group": r["group"], "N": r["N"], "type": r["type"]})[f"pq_{r['method'].lower()}"] = r["PQ"]
        # H1 pairs a bridge only when both methods produced a score: a failed M2 fit would
        # otherwise enter the slope difference as a zero and bias the bootstrap interval.
        P = []
        for v in pair.values():
            if "pq_m1" not in v or "pq_m2" not in v:
                continue
            if v["pq_m1"] is None or v["pq_m2"] is None:
                n_pair_dropped += 1
                continue
            P.append(v)
        out[src]["H1"] = {"all": bootstrap_slope_diff(P) if len(P) >= 3 else None,
                          "n_dropped": n_pair_dropped,
                          "by_type": {t: bootstrap_slope_diff([p for p in P if p["type"] == t])
                                      for t in sorted({p["type"] for p in P}) if sum(p["type"] == t for p in P) >= 3}}
    return out


def trackb_summary(root):
    tb = os.path.join(root, "trackB")
    out = {"prepare": load_json(os.path.join(tb, "vtti_prepare.json")) if os.path.exists(os.path.join(tb, "vtti_prepare.json")) else None,
           "render": {}, "measure": {}}
    for meth in ("gs3d", "gs2d"):
        d = {}
        for name in ("metrics_images.json", "run_info.json"):
            p = os.path.join(tb, meth, name)
            if os.path.exists(p):
                d[name.split(".")[0]] = load_json(p)
        if d:
            out["render"][meth] = d
    for p in sorted(glob.glob(os.path.join(tb, "measure", "measure_*.json"))):
        r = load_json(p)
        out["measure"][r["method"]] = r
    p = os.path.join(tb, "vggt", "run_info.json")
    out["vggt"] = load_json(p) if os.path.exists(p) else None
    return out


def rq3_extra(root):
    """{method: merge.json} for the optional repeats of RQ3 on other point clouds (rq3_<method>/)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "rq3_*"))):
        p = os.path.join(d, "merge.json")
        if os.path.exists(p):
            out[os.path.basename(d)[4:]] = load_json(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", default="")
    ap.add_argument("--cfg", default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    classes = cfg["brpcd"]["classes"]
    index = _index(a.data) if a.data else {}
    rows = instance_rows(a.results)
    summ = {"classes": classes, "semseg": semseg_summary(a.results, index, len(classes)),
            "instances": instance_summary(rows) if rows else {}, "trackB": trackb_summary(a.results),
            "rq3": load_json(os.path.join(a.results, "rq3", "merge.json"))
            if os.path.exists(os.path.join(a.results, "rq3", "merge.json")) else None,
            "rq3_extra": rq3_extra(a.results)}
    os.makedirs(os.path.join(a.results, "summary"), exist_ok=True)
    save_json(summ, os.path.join(a.results, "summary", "summary.json"))
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith("PQ_c"), k))
        with open(os.path.join(a.results, "summary", "instances_rows.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    log(f"summary written: {os.path.join(a.results, 'summary', 'summary.json')} "
        f"(semseg models {list(summ['semseg'])}, instance sources {list(summ['instances'])}, "
        f"measured {list(summ['trackB']['measure'])}, rq3 {'yes' if summ['rq3'] else 'no'})")


if __name__ == "__main__":
    main()
