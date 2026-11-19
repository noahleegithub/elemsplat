#!/usr/bin/env python3
"""Where does each reconstruction lose its PSNR? Per-tile decomposition against the photograph.

A single mean PSNR cannot distinguish two very different failure modes:
  (a) the method cannot model the *bridge* (planar deck, thin girders, railings), or
  (b) the method cannot model the *far field* (hundreds of metres of vegetation and terrain),
      which in our UAV frames occupies most of the pixels.

Tiling the image and differencing each method against the best one separates them: a deficit
concentrated in the upper rows is far-field background, a deficit concentrated on the deck is
geometry. Nothing is segmented; the spatial pattern is the evidence.

    python diag_tile_psnr.py --results $RESULTS_DIR --out <dir> [--grid 8x6]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

INK = "#1b1b1b"
METHODS = [("3DGS", "gs3d"), ("2DGS", "gs2d"), ("PGSR", "gs_pgsr")]
IT = 30000


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
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="8x6", help="columns x rows of tiles")
    a = ap.parse_args()
    cols, rows = (int(v) for v in a.grid.lower().split("x"))
    tb = os.path.join(a.results, "trackB")

    gt_dir = os.path.join(tb, "gs3d", "test", f"ours_{IT}", "gt")
    base_names = sorted(f for f in os.listdir(gt_dir) if f.lower().endswith(".png"))

    per_tile = {label: [] for label, _ in METHODS}
    overall = {label: [] for label, _ in METHODS}
    for name in base_names:
        g = load(os.path.join(gt_dir, name))
        H, W = g.shape[:2]
        for label, sub in METHODS:
            rp = render_map(os.path.join(tb, sub), base_names).get(name)
            if not rp or not os.path.exists(rp):
                continue
            r = load(rp)
            if r.shape != g.shape:
                continue
            overall[label].append(psnr(r, g))
            tile = np.zeros((rows, cols))
            for i in range(rows):
                for j in range(cols):
                    sl = (slice(i * H // rows, (i + 1) * H // rows), slice(j * W // cols, (j + 1) * W // cols))
                    tile[i, j] = psnr(r[sl], g[sl])
            per_tile[label].append(tile)

    print(f"views used: {len(overall['3DGS'])}  grid {cols}x{rows}")
    for label, _ in METHODS:
        v = np.array(overall[label])
        print(f"  {label:5s} whole-image PSNR mean {v.mean():6.2f} dB")

    ref = "3DGS"
    ref_tile = np.mean(per_tile[ref], axis=0)
    print(f"\nper-tile deficit against {ref} (negative = worse), averaged over views:")
    head = "".join(f"{'col ' + str(j):>9}" for j in range(cols))
    print(f"{'row':<6}{head}")
    deltas = {}
    for label, _ in METHODS:
        if label == ref:
            continue
        t = np.mean(per_tile[label], axis=0)
        deltas[label] = t - ref_tile
        for i in range(rows):
            print(f"r{i:<5}" + "".join(f"{deltas[label][i, j]:>9.2f}" for j in range(cols)))
        band = [deltas[label][: rows // 3].mean(), deltas[label][rows // 3: 2 * rows // 3].mean(),
                deltas[label][2 * rows // 3:].mean()]
        print(f"  {label}: top third {band[0]:+.2f} dB | middle {band[1]:+.2f} | bottom {band[2]:+.2f}\n")

    os.makedirs(a.out, exist_ok=True)
    fig, axes = plt.subplots(1, len(deltas), figsize=(3.2 * len(deltas) + 0.6, 2.4), squeeze=False)
    for ax, (label, d) in zip(axes[0], deltas.items()):
        vmax = float(np.abs(d).max())
        im = ax.imshow(d, cmap="RdBu", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{label} - {ref}", fontsize=8, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for i in range(rows):
            for j in range(cols):
                ax.text(j, i, f"{d[i, j]:.0f}", ha="center", va="center", fontsize=4.6,
                        color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=5)
    fig.suptitle("PSNR deficit per image tile (dB); top rows are far-field vegetation/sky",
                 fontsize=7.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fp = os.path.join(a.out, "fig_tile_psnr.png")
    fig.savefig(fp, dpi=250, bbox_inches="tight")
    print("wrote", fp)
    with open(os.path.join(a.out, "tile_psnr.json"), "w") as fh:
        json.dump({k: v.tolist() for k, v in deltas.items()}, fh, indent=2)


if __name__ == "__main__":
    main()
