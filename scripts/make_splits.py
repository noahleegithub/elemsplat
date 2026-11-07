#!/usr/bin/env python
"""
make_splits.py -- bridge-level splits (paper Section 4.1, "Splits").

Variants always follow their source scan, so a test bridge never leaks into training.
mode loo   : one fold per real scan; validation = another real scan, of the same type when possible
mode fixed3: a single fold whose test scans are splits.fixed_test
Also writes the "all" entry used to train the network for RQ3.

    python make_splits.py --data $BRPCD_DIR
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_cfg, log, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cfg", default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    with open(os.path.join(a.data, "index.csv")) as f:
        rows = list(csv.DictReader(f))
    reals = [r for r in rows if r["is_real"] == "1"]
    group = {r["bridge"]: r["source"] for r in rows}
    members = lambda src: sorted(b for b, s in group.items() if s == src)
    mode, sp = cfg["splits"]["mode"], cfg["splits"]
    folds = []
    if mode == "loo":
        for k, r in enumerate(reals):
            others = [o for o in reals if o["bridge"] != r["bridge"]]
            same = [o for o in others if o["type"] == r["type"]]
            pool = same or others
            val_src = [pool[(k + j) % len(pool)]["bridge"] for j in range(min(sp.get("val_per_fold", 1), len(pool)))]
            test = members(r["bridge"])
            val = sorted(b for v in val_src for b in members(v))
            train = sorted(b for b in group if b not in test and b not in val)
            folds.append({"fold": k, "test_scan": r["bridge"], "type": r["type"], "test": test, "val": val, "train": train})
    elif mode == "fixed3":
        test_src = sp["fixed_test"]
        if not test_src:
            sys.exit("splits.fixed_test is empty -- list one real scan per bridge type in experiment.yaml")
        others = [o["bridge"] for o in reals if o["bridge"] not in test_src]
        val_src = others[: sp.get("val_per_fold", 1)]
        test = sorted(b for s in test_src for b in members(s))
        val = sorted(b for s in val_src for b in members(s))
        folds.append({"fold": 0, "test_scan": ",".join(test_src), "type": "mixed", "test": test, "val": val,
                      "train": sorted(b for b in group if b not in test and b not in val)})
    else:
        sys.exit(f"unknown splits.mode {mode}")
    for fo in folds:
        assert not set(fo["test"]) & set(fo["train"]), "leak: test bridge in training"
        assert not set(fo["val"]) & set(fo["train"]), "leak: validation bridge in training"
    split = {"mode": mode, "folds": folds, "all": sorted(group), "groups": group}
    save_json(split, os.path.join(a.data, "split.json"))
    for fo in folds:
        log(f"fold {fo['fold']:2d} test={fo['test_scan']:28s} ({len(fo['test'])} clouds)  val={len(fo['val'])}  "
            f"train={len(fo['train'])}")
    log(f"written {os.path.join(a.data, 'split.json')}")


if __name__ == "__main__":
    main()
