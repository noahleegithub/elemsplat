#!/usr/bin/env python3
"""Primitive geometry statistics per method: are the disks degenerating?

3DGS stores isotropic-ish 3D covariance; the 2DGS and PGSR forks constrain one axis to ~0 so each
primitive is a flat disk. That is the intended design, but it has a testable consequence: a disk
can represent a surface but not a volume, so foliage and other volumetric structure must be
approximated by many edge-on disks that contribute almost nothing to the rendered image. The
signature would be a pile of degenerate primitives (min scale ~ 0) with no compensating increase
in coverage - a much larger primitive count than 3DGS at a lower PSNR, which is exactly what we
observe.

Reports, over the primitives actually used for rendering (opacity > 0.5):
  - median of the three scales (world units)
  - median thickness = smallest scale
  - median aspect = largest / smallest
  - share of primitives with thickness < 1 mm
  - share with thickness < 1 cm

    python diag_primitive_scales.py name=path/to/point_cloud.ply [...]
"""
import sys

import numpy as np


def read_ply(path):
    try:
        from plyfile import PlyData
    except ImportError:
        sys.exit("plyfile missing: pip install plyfile")
    v = PlyData.read(path)["vertex"]
    names = v.properties
    props = [n.name if hasattr(n, "name") else str(n) for n in names]
    cols = [np.asarray(v[props[i]]) for i in range(len(props))]
    idx = {n: i for i, n in enumerate(props)}
    return cols, idx


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(f"{'method':<10}{'used':>10}{'kept %':>8}{'thick med':>12}{'largest med':>13}"
          f"{'aspect':>8}{'<1mm':>8}{'<1cm':>8}")
    print("=" * 80)
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        cols, idx = read_ply(path)
        def col(name):
            return cols[idx[name]].astype(np.float64)
        scales = np.stack([np.exp(col(f"scale_{i}")) for i in range(3)], 1)
        opac = 1.0 / (1.0 + np.exp(-col("opacity")))
        keep = opac > 0.5
        s = scales[keep]
        s_min = s.min(1)
        s_max = s.max(1)
        print(f"{label:<10}{keep.sum():>10,}{100 * keep.mean():>7.1f}%"
              f"{np.median(s_min) * 1000:>11.2f}mm{np.median(s_max) * 1000:>12.1f}mm"
              f"{np.median(s_max / np.maximum(s_min, 1e-12)):>8.0f}"
              f"{100 * (s_min < 1e-3).mean():>7.1f}%{100 * (s_min < 1e-2).mean():>7.1f}%")


if __name__ == "__main__":
    main()
