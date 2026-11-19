#!/usr/bin/env python3
"""Is a method's PSNR deficit geometry, or a global colour/exposure offset?

The per-tile decomposition (diag_tile_psnr.py) shows the deficit is roughly uniform across the
image. Geometry errors are not uniform: melting hits textured surfaces, thin structure and
grazing angles harder than flat sky. A uniform deficit points instead at tone: a per-channel
gain/offset difference between render and photograph.

This measures how much PSNR each method recovers if you remove (a) a per-channel mean offset and
(b) a per-channel affine gain+offset, fitted per view on that same view. The fitted correction is
diagnostic only - it is not a legitimate reportable score, since it is fitted on the test frames -
but the *size* of the recovery says whether the error is photometric or geometric.

    python diag_color_psnr.py --results $RESULTS_DIR
"""
import argparse
import os

import numpy as np
from PIL import Image

IT = 30000
METHODS = [("3DGS", "gs3d"), ("2DGS", "gs2d"), ("PGSR", "gs_pgsr")]
INK = "\033[0m"


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse <= 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def render_map(model_dir, base_names):
    d = os.path.join(model_dir, "test", f"ours_{IT}", "renders")
    if not os.path.isdir(d):
        return {}
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".png"))
    names = [os.path.basename(f) for f in files]
    if set(names) & set(base_names):
        return {n: p for n, p in zip(names, files) if n in base_names}
    if len(files) == len(base_names):
        return dict(zip(base_names, files))
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    tb = os.path.join(a.results, "trackB")
    gt_dir = os.path.join(tb, "gs3d", "test", f"ours_{IT}", "gt")
    base = sorted(f for f in os.listdir(gt_dir) if f.lower().endswith(".png"))

    rows = []
    for label, sub in METHODS:
        rm = render_map(os.path.join(tb, sub), base)
        raw, shifted, affine, biases = [], [], [], []
        for name in base:
            rp = rm.get(name)
            if not rp or not os.path.exists(rp):
                continue
            g = load(os.path.join(gt_dir, name))
            r = load(rp)
            if r.shape != g.shape:
                continue
            raw.append(psnr(r, g))
            # (a) per-channel mean offset
            bias = np.array([(r[..., c] - g[..., c]).mean() for c in range(3)])
            biases.append(bias)
            r_s = np.clip(r - bias, 0, 255)
            shifted.append(psnr(r_s, g))
            # (b) per-channel affine gain+offset, least squares
            r_a = r.copy()
            for c in range(3):
                x, y = r[..., c].ravel(), g[..., c].ravel()
                A = np.stack([x, np.ones_like(x)], 1)
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                r_a[..., c] = np.clip(r[..., c] * coef[0] + coef[1], 0, 255)
            affine.append(psnr(r_a, g))
        if not raw:
            continue
        b = np.mean(biases, axis=0)
        rows.append((label, np.mean(raw), np.mean(shifted), np.mean(affine), b))

    print(f"{'method':<7}{'PSNR':>8}{'+offset':>10}{'+affine':>10}{'recovered':>11}   mean render-GT bias (RGB)")
    print("=" * 78)
    for label, raw, sh, af, b in rows:
        print(f"{label:<7}{raw:>8.2f}{sh:>10.2f}{af:>10.2f}{af - raw:>+11.2f}   "
              f"{b[0]:+6.1f} {b[1]:+6.1f} {b[2]:+6.1f}")
    if len(rows) > 1:
        gap = rows[0][1] - rows[1][1]
        gap_aff = rows[0][3] - rows[1][3]
        print(f"\n{rows[0][0]} - {rows[1][0]}: {gap:.2f} dB raw -> {gap_aff:.2f} dB after a per-view "
              f"affine colour fit ({(1 - gap_aff / gap) * 100:.0f}% of the gap is photometric)")


if __name__ == "__main__":
    main()
