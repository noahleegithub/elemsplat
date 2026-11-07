"""
brpcd_io.py -- discovery of the BrPCD folder tree (shared by brpcd_inspect.py and brpcd_convert.py).

Expected layout (README of github.com/wxiong12/SEU-Digital-Bridge):
    <raw>/<bridge type>/<bridge>/[<label folder>/]<member files>
A file directly inside a bridge folder that sits next to label sub-folders is taken as the
complete cloud and skipped; every file below the label folders is one member (= instance).
If a bridge folder holds only files, each file is a member and its name gives the class.
"""
import os
import re

TYPE_RULES = [("suspension", r"susp|^ss[-_]|\bss\b"), ("cable-stayed", r"cable|stay|^cs[-_]"),
              ("girder", r"girder|beam|continuous|simply|^c[-_]|^s[-_]")]


def list_point_files(root, exts):
    exts = {e.lower() for e in exts}
    out = []
    for d, _, files in os.walk(root):
        if "__MACOSX" in d:
            continue
        for f in files:
            if not f.startswith("._") and os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(d, f))
    return sorted(out)


def find_bridge_dirs(raw, exts):
    """Folders that look like one bridge: chosen depth = the one whose names most often contain 'bridge'."""
    files = list_point_files(raw, exts)
    if not files:
        return []
    rels = [os.path.relpath(f, raw).split(os.sep) for f in files]
    best_depth, best_score = 1, -1
    for depth in range(1, max(len(r) for r in rels)):
        names = {r[depth - 1] for r in rels if len(r) > depth}
        score = sum("bridge" in n.lower() for n in names) + 0.01 * (depth == 2)
        if score > best_score:
            best_depth, best_score = depth, score
    dirs = sorted({os.path.join(raw, *r[:best_depth]) for r in rels if len(r) > best_depth})
    return dirs


def bridge_type(bridge_dir, raw):
    rel = os.path.relpath(bridge_dir, raw).lower()
    name = os.path.basename(bridge_dir).lower()
    for t, pat in TYPE_RULES:
        if re.search(pat, rel.split(os.sep)[0]) or re.search(pat, name):
            return t
    return "unknown"


def member_files(bridge_dir, exts):
    """(members, skipped_complete_clouds)."""
    files = list_point_files(bridge_dir, exts)
    top = [f for f in files if os.path.dirname(f) == bridge_dir]
    deeper = [f for f in files if os.path.dirname(f) != bridge_dir]
    if deeper:
        return deeper, top
    return top, []


def class_of(rel_path, patterns, order, ignore):
    low = rel_path.lower().replace("\\", "/")
    for pat in ignore:
        if re.search(pat, low):
            return "__ignore__"
    for c in order:
        for pat in patterns.get(c, []):
            if re.search(pat, low, flags=re.IGNORECASE):
                return c
    return None


def count_lines(path, max_bytes=None):
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            n += b.count(b"\n")
    return n
