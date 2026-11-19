#!/usr/bin/env python3
"""Make PGSR accept the standard COLMAP layout (sparse/0/...) as well as sparse/*.bin.

PGSR's readColmapSceneInfo hardcodes `sparse/images.bin`, i.e. it expects the model files
directly under `sparse/`. Our VTTI scene (and every 3DGS/2DGS dataset) puts them in
`sparse/0/`, so the loader dies with FileNotFoundError before training starts.

The patch resolves the model directory once and uses it for the camera files and the
points3D conversion. Idempotent: running it twice is a no-op.

    python patch_pgsr.py /path/to/PGSR
"""
import re
import sys
import os

root = sys.argv[1] if len(sys.argv) > 1 else "."
f = os.path.join(root, "scene", "dataset_readers.py")
src = open(f).read()

if "model_dir = _resolve_sparse" in src:
    print("already patched")
    sys.exit(0)

helper = '''
def _resolve_sparse(path):
    """Return the directory that actually holds the COLMAP model.

    PGSR assumes `sparse/images.bin`; standard 3DGS/2DGS datasets (and ours) use `sparse/0/`.
    """
    sparse = os.path.join(path, "sparse")
    for cand in ("0", ""):
        d = os.path.join(sparse, cand) if cand else sparse
        if os.path.exists(os.path.join(d, "images.bin")) or os.path.exists(os.path.join(d, "images.txt")):
            return d
    return sparse


'''

# insert the helper before readColmapSceneInfo
anchor = "def readColmapSceneInfo(path, images, eval, llffhold=8):"
assert anchor in src, "readColmapSceneInfo not found"
src = src.replace(anchor, helper.lstrip("\n") + anchor, 1)

# resolve the model dir at the top of that function
head = anchor + "\n"
assert head in src
src = src.replace(head, head + "    model_dir = _resolve_sparse(path)\n", 1)

# point every sparse path at the resolved directory
src = src.replace('os.path.join(path, "sparse", "images.bin")', 'os.path.join(model_dir, "images.bin")')
src = src.replace('os.path.join(path, "sparse", "cameras.bin")', 'os.path.join(model_dir, "cameras.bin")')
src = src.replace('os.path.join(path, "sparse", "images.txt")', 'os.path.join(model_dir, "images.txt")')
src = src.replace('os.path.join(path, "sparse", "cameras.txt")', 'os.path.join(model_dir, "cameras.txt")')
src = src.replace('os.path.join(path, "sparse/points3D.ply")', 'os.path.join(model_dir, "points3D.ply")')
src = src.replace('os.path.join(path, "sparse/points3D.bin")', 'os.path.join(model_dir, "points3D.bin")')
src = src.replace('os.path.join(path, "sparse/points3D.txt")', 'os.path.join(model_dir, "points3D.txt")')

open(f, "w").write(src)
left = len(re.findall(r'os\.path\.join\(path, "sparse', src))
print(f"patched {f}; remaining hardcoded sparse paths: {left}")
