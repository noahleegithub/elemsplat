#!/usr/bin/env python3
"""Contact sheet of held-out VTTI renders: photograph vs 3DGS vs 2DGS vs PGSR.

fig_render in make_figures.py shows two views, which is enough for a caption but not enough to
judge a method. This draws N views picked by rank (best, quartiles, worst) so the spread is
visible instead of the mean hiding it, with each cell labelled by its own PSNR.

The three directions do not agree on render file names: 3DGS/2DGS write 00028.png by position in
the sorted test list, PGSR writes DJI_0619.png by image name. Views are therefore matched by name
where the sets intersect and by position otherwise, and the counts are checked.

    python fig_render_sheet.py --results $RESULTS_DIR --out <dir> [--rows 6]
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

METHODS = [("3DGS", "gs3d"), ("2DGS", "gs2d"), ("PGSR", "gs_pgsr")]
INK = "#1b1b1b"
INK2 = "#6b6b6b"
ACCENT = {"3DGS": "#eb6834", "2DGS": "#2a78d6", "PGSR": "#1baf7a"}


def load_json(p):
    with open(p) as fh:
        return json.load(fh)


def view_map(model_dir, it, base_names):
    """Return {base_view_name: render_path} for one method."""
    d = os.path.join(model_dir, "test", f"ours_{it}", "renders")
    files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".png"))
    names = [os.path.basename(f) for f in files]
    if set(names) & set(base_names):
        return {n: p for n, p in zip(names, files) if n in base_names}
    assert len(files) == len(base_names), f"{model_dir}: {len(files)} renders vs {len(base_names)} views"
    return dict(zip(base_names, files))


def metric_map(model_dir, base_names):
    """Return {base_view_name: {psnr, ssim, lpips}} for one method.

    metrics_images.json inherits the method's own naming, so PGSR's per_view is keyed by
    DJI_0619.png while the sheet indexes 00025.png. Re-key by position when the names differ,
    otherwise every PGSR cell prints nan.
    """
    p = os.path.join(model_dir, "metrics_images.json")
    if not os.path.exists(p):
        return {}
    d = load_json(p).get("per_view") or {}
    if set(d) & set(base_names):
        return {k: v for k, v in d.items() if k in base_names}
    keys = sorted(d)
    if len(keys) == len(base_names):
        return dict(zip(base_names, [d[k] for k in keys]))
    print(f"WARNING: {model_dir}: {len(keys)} metric rows vs {len(base_names)} views, PSNR suppressed")
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--rows", type=int, default=6)
    a = ap.parse_args()
    tb = os.path.join(a.results, "trackB")
    os.makedirs(a.out, exist_ok=True)

    # base view list + ground truth from the first method that has renders
    base, gt = None, None
    for label, sub in METHODS:
        d = os.path.join(tb, sub, "test", f"ours_{a.iterations}")
        if os.path.isdir(os.path.join(d, "renders")):
            base = sorted(f for f in os.listdir(os.path.join(d, "renders")) if f.lower().endswith(".png"))
            gt = os.path.join(d, "gt")
            break
    if base is None:
        sys.exit("no renders found under trackB/*/test/ours_*/renders")

    paths = {label: view_map(os.path.join(tb, sub), a.iterations, base) for label, sub in METHODS}
    psnr = {label: metric_map(os.path.join(tb, sub), base) for label, sub in METHODS}

    # rank by the mean of the three methods' PSNR so the sample reflects the methods, not one of them
    def rank_key(n):
        vals = [psnr[l].get(n, {}).get("psnr") for l in psnr]
        vals = [v for v in vals if v]
        return float(np.mean(vals)) if vals else 0.0

    ordered = sorted(paths["3DGS"].keys() or base, key=rank_key)
    n = len(ordered)
    wanted = min(a.rows, n)
    picks = []
    for i in range(wanted):
        frac = i / max(1, wanted - 1) if wanted > 1 else 0.0
        picks.append(ordered[min(n - 1, int(round(frac * (n - 1))))])
    # best row at the top reads better than worst-first
    picks = picks[::-1]

    cols = ["photo"] + [l for l, _ in METHODS]
    fig, axes = plt.subplots(wanted, len(cols), figsize=(9.0, 1.34 * wanted + 0.5), squeeze=False)
    for r, name in enumerate(picks):
        cell = [(os.path.join(gt, name), None)]
        for label, _ in METHODS:
            cell.append((paths[label].get(name), psnr[label].get(name, {}).get("psnr")))
        for c, (path, score) in enumerate(cell):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if path and os.path.exists(path):
                ax.imshow(Image.open(path))
            if r == 0:
                ttl = "photograph" if c == 0 else cols[c]
                ax.set_title(ttl, fontsize=8, color=INK if c == 0 else ACCENT[cols[c]], pad=4,
                             fontweight="bold")
            if score:
                ax.text(0.985, 0.02, f"{score:.2f} dB", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=6.6, color="white",
                        bbox=dict(boxstyle="round,pad=0.18", fc=INK, ec="none", alpha=0.72))
        # row label: which held-out view, and how the three methods rank on it
        best = max((l for l, _ in METHODS), key=lambda l: psnr[l].get(name, {}).get("psnr", -1))
        axes[r][0].set_ylabel(f"{name.replace('.png', '')}\nbest: {best}", fontsize=6.4, color=INK2,
                              rotation=0, ha="right", va="center", labelpad=34)
    fig.suptitle("Held-out VTTI views (every 8th image, 35 in total) — rows sampled by mean PSNR: "
                 "best at the top, worst at the bottom", fontsize=7.4, color=INK, y=0.995)
    fig.tight_layout(rect=(0.045, 0, 1, 0.975))
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {})):
        fp = os.path.join(a.out, f"fig_render_sheet.{ext}")
        fig.savefig(fp, bbox_inches="tight", **kw)
        print("wrote", fp)
    plt.close(fig)

    # per-method spread, so the sheet is not read as "one number per method"
    print("\nview                      " + "".join(f"{l:>9}" for l, _ in METHODS))
    for name in picks:
        row = "".join(f"{psnr[l].get(name, {}).get('psnr', float('nan')):>9.2f}" for l, _ in METHODS)
        print(f"{name:<25}{row}")
    for l, _ in METHODS:
        v = [psnr[l][k]["psnr"] for k in psnr[l] if "psnr" in psnr[l][k]]
        print(f"{l} over {len(v)} views: mean {np.mean(v):.2f}  median {np.median(v):.2f}  "
              f"min {np.min(v):.2f}  max {np.max(v):.2f}")


if __name__ == "__main__":
    main()
