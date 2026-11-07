#!/usr/bin/env python
"""
brpcd_inspect.py -- "checks after download" for BrPCD (paper Section 4.1).

Reports the folder tree, file formats, column counts, point counts, coordinate extents
(to confirm the unit), whether points carry colour, and how every member file maps to a
class. Nothing is converted. Read inspect_report.txt before running brpcd_convert.py.

    python brpcd_inspect.py --raw $BRPCD_RAW --out $BRPCD_DIR
"""
import argparse
import collections
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brpcd_io import bridge_type, class_of, count_lines, find_bridge_dirs, member_files  # noqa: E402
from common import load_cfg, load_yaml, pkg_path, save_json  # noqa: E402
from pcio import read_points  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--read-limit", type=int, default=3, help="member files per bridge read fully for extents")
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    bc = cfg["brpcd"]
    cp = load_yaml(pkg_path(bc["class_patterns"]))
    exts = bc["point_extensions"]
    os.makedirs(a.out, exist_ok=True)
    lines, report = [], {"bridges": []}
    say = lambda s="": (lines.append(s), print(s))

    bdirs = find_bridge_dirs(a.raw, exts)
    say(f"raw folder : {a.raw}")
    say(f"bridges    : {len(bdirs)} folders detected")
    real_re = re.compile(bc["real_scan_regex"])
    ext_count = collections.Counter()
    unmatched = collections.Counter()
    for bd in bdirs:
        name = os.path.basename(bd)
        btype = bridge_type(bd, a.raw)
        members, skipped = member_files(bd, exts)
        cls_count = collections.Counter()
        n_lines_total = 0
        for f in members:
            rel = os.path.relpath(f, bd)
            c = class_of(rel, cp["patterns"], cp["order"], cp.get("ignore", []))
            if c is None:
                unmatched[rel.split(os.sep)[0] if os.sep in rel else os.path.splitext(rel)[0]] += 1
            cls_count[c or "UNMATCHED"] += 1
            ext_count[os.path.splitext(f)[1].lower()] += 1
            if os.path.splitext(f)[1].lower() in (".txt", ".xyz", ".pts", ".csv", ".asc"):
                n_lines_total += count_lines(f)
        sample = []
        for f in members[: a.read_limit]:
            try:
                d = read_points(f)
                x = d["xyz"]
                sample.append({"file": os.path.relpath(f, bd), "format": d["format"], "n_cols": d["n_cols"],
                               "n": int(len(x)), "rgb": d["rgb"] is not None,
                               "min": x.min(0).round(3).tolist(), "max": x.max(0).round(3).tolist()})
            except Exception as e:  # noqa: BLE001
                sample.append({"file": os.path.relpath(f, bd), "error": str(e)})
        rec = {"bridge": name, "dir": bd, "type": btype, "is_real": bool(real_re.search(name)),
               "n_member_files": len(members), "skipped_complete_clouds": [os.path.basename(s) for s in skipped],
               "classes": dict(cls_count), "ascii_lines": n_lines_total, "sample": sample}
        report["bridges"].append(rec)
        say("")
        say(f"[{'REAL' if rec['is_real'] else 'var '}] {name:40s} type={btype:12s} members={len(members):4d} "
            f"lines~{n_lines_total:,}")
        say(f"        classes: {dict(cls_count)}")
        if skipped:
            say(f"        skipped complete-cloud files: {rec['skipped_complete_clouds']}")
        for s in sample:
            say(f"        sample {s}")
    say("")
    say(f"file extensions : {dict(ext_count)}")
    say(f"real scans      : {sum(b['is_real'] for b in report['bridges'])} (expected 10)")
    say(f"variants        : {sum(not b['is_real'] for b in report['bridges'])} (expected 88)")
    if unmatched:
        say(f"UNMATCHED names : {dict(unmatched)}  -> add patterns to configs/brpcd_classes.yaml")
    ext = [s for b in report["bridges"] for s in b["sample"] if "max" in s]
    if ext:
        span = np.max([np.subtract(s["max"], s["min"]) for s in ext], 0)
        say(f"largest member extent in the samples: {np.round(span, 2).tolist()}  "
            "(tens to hundreds = metres; tens of thousands = millimetres -> set brpcd.unit_scale: 0.001)")
    report["unmatched"] = dict(unmatched)
    save_json(report, os.path.join(a.out, "inspect.json"))
    with open(os.path.join(a.out, "inspect_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwritten: {os.path.join(a.out, 'inspect_report.txt')}")


if __name__ == "__main__":
    main()
