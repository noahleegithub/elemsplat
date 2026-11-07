#!/usr/bin/env python
"""
gs_points.py -- point clouds from the two reconstructions (paper Section 3.4).

    3DGS : centres of the Gaussians whose opacity exceeds points.gs3d_min_opacity
    2DGS : the TSDF mesh sampled uniformly by area (points.mesh_samples points)

    python gs_points.py --gs3d <model_dir> --iteration 30000 --out pts_3dgs.npz
    python gs_points.py --mesh <model_dir>/train/ours_30000/fuse_post.ply --out pts_2dgs.npz
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_cfg, log  # noqa: E402
from pcio import read_gaussians_ply, read_ply_mesh  # noqa: E402


def sample_mesh(verts, faces, n, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    ok = area > 0
    a, b, c, area = a[ok], b[ok], c[ok], area[ok]
    f = rng.choice(len(area), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[f] + u[:, None] * (b[f] - a[f]) + v[:, None] * (c[f] - a[f]), float(area.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs3d", default=None, help="3DGS model folder")
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--mesh", default=None, help="2DGS TSDF mesh (.ply) or a glob")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    pc = cfg["points"]
    if a.gs3d:
        ply = os.path.join(a.gs3d, "point_cloud", f"iteration_{a.iteration}", "point_cloud.ply")
        xyz, op = read_gaussians_ply(ply)
        keep = op > pc["gs3d_min_opacity"]
        np.savez_compressed(a.out, xyz=xyz[keep].astype(np.float32), n_total=len(xyz), opacity_thr=pc["gs3d_min_opacity"])
        log(f"3DGS: {keep.sum():,} of {len(xyz):,} Gaussians kept (opacity > {pc['gs3d_min_opacity']}) -> {a.out}")
    elif a.mesh:
        paths = sorted(glob.glob(a.mesh))
        if not paths:
            sys.exit(f"no mesh matches {a.mesh}")
        verts, faces = read_ply_mesh(paths[0])
        if not len(faces):
            sys.exit(f"{paths[0]} has no faces")
        pts, area = sample_mesh(verts, faces, pc["mesh_samples"])
        np.savez_compressed(a.out, xyz=pts.astype(np.float32), n_faces=len(faces), area_sfm=area)
        log(f"2DGS mesh {paths[0]}: {len(faces):,} faces -> {len(pts):,} samples -> {a.out}")
    else:
        sys.exit("give --gs3d or --mesh")


if __name__ == "__main__":
    main()
