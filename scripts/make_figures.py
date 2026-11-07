#!/usr/bin/env python
"""
make_figures.py -- the result figures of the paper (PDF with embedded TrueType fonts, as
NeurIPS requires, plus PNG for slides).

    fig_pq_vs_n         Fig. 2  PQ against N for M1 and M2, one panel per bridge type
    fig_semantic_iou            per-class IoU of the semantic networks (real test scans)
    fig_brpcd_examples          GT members vs M1 vs M2 on one test bridge per type
    fig_render          Fig. 3  held-out view: photo, 3DGS, 2DGS (+ 2DGS normals)
    fig_measure                 side-span errors and deck patch residuals, 3DGS vs 2DGS
    fig_vtti_instances  Fig. 4  VTTI 2DGS points coloured by class and by M2 member

    python make_figures.py --results $RESULTS_DIR --data $BRPCD_DIR --out <paper>/figures/results
    add  --stamp "SYNTHETIC SMOKE TEST - NOT RESULTS"  for the smoke test
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_cfg, load_json, log  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"   # categorical slots 1, 2, 3, 7
INK, INK2, GRID, MUTE = "#0b0b0b", "#52514e", "#e6e6e3", "#b9b8b3"
W = 5.5                                                   # NeurIPS text width (in)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200,
    "axes.axisbelow": True})
TYPES = ["girder", "cable-stayed", "suspension"]
STAMP = None


def save(fig, out, name):
    if STAMP:
        fig.text(0.0, -0.01, STAMP, ha="left", va="top", fontsize=7, color="#b00020", fontweight="bold")
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    log(f"  {name}.pdf/.png")


def placeholder(out, name, text, h=1.6):
    fig = plt.figure(figsize=(W, h))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.01, 0.05), 0.98, 0.9, transform=ax.transAxes, fill=False, ec=MUTE, ls="--"))
    import textwrap
    ax.text(0.5, 0.5, textwrap.fill(text, 90), ha="center", va="center", color=INK2, fontsize=7.5,
            transform=ax.transAxes)
    save(fig, out, name)


def palette(n, seed=3):
    """Distinct colours for many members: the categorical hues at three lightness steps, shuffled."""
    base = np.array([[42, 120, 214], [235, 104, 52], [27, 175, 122], [237, 161, 0], [232, 123, 164],
                     [0, 131, 0], [98, 80, 214], [227, 73, 72]], float) / 255
    cols = np.concatenate([base, 0.55 * base + 0.45, 0.6 * base])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(np.resize(np.arange(len(cols)), max(n, 1)))
    return cols[idx]


# ----------------------------------------------------------------------------- Track A
def fig_pq_vs_n(results, cfg, out):
    p = os.path.join(results, "summary", "instances_rows.csv")
    if not os.path.exists(p):
        return placeholder(out, "fig_pq_vs_n", "PQ against N: waiting for Track A (03b_instances)")
    with open(p) as f:
        rows = list(csv.DictReader(f))
    model = cfg["merge"]["semseg_model"]
    types = [t for t in TYPES if any(r["type"] == t for r in rows)] or sorted({r["type"] for r in rows})
    fig, axes = plt.subplots(1, len(types), figsize=(W, 1.95), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], types):
        for src, filled, ls in ((model, True, "-"), ("true", False, "--")):
            for meth, col in (("M2", BLUE), ("M1", ORANGE)):
                R = [r for r in rows if r["type"] == t and r["classes"] == src and r["method"] == meth and r["PQ"]]
                if not R:
                    continue
                N = np.array([float(r["N"]) for r in R])
                pq = 100 * np.array([float(r["PQ"]) for r in R])
                real = np.array([r["is_real"] == "1" for r in R])
                ax.scatter(N[~real], pq[~real], s=12, marker="o", facecolors=col if filled else "none",
                           edgecolors=col, linewidths=0.8, alpha=0.55, zorder=3)
                ax.scatter(N[real], pq[real], s=22, marker="D", facecolors=col if filled else "none",
                           edgecolors=col, linewidths=0.9, zorder=4)
                if len(N) >= 3 and np.ptp(np.log(N)) > 0:
                    b1, b0 = np.polyfit(np.log(N), pq, 1)
                    xx = np.geomspace(N.min(), N.max(), 30)
                    ax.plot(xx, b0 + b1 * np.log(xx), color=col, ls=ls, lw=1.0, zorder=2)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(t, color=INK)
        ax.set_xlabel("members $N$")
        ax.set_ylim(0, 102)
    axes[0][0].set_ylabel("PQ (%)")
    h = [plt.Line2D([], [], color=BLUE, marker="o", ls="-", label="M2, predicted classes"),
         plt.Line2D([], [], color=ORANGE, marker="o", ls="-", label="M1, predicted classes"),
         plt.Line2D([], [], color=BLUE, marker="o", mfc="none", ls="--", label="M2, true classes"),
         plt.Line2D([], [], color=ORANGE, marker="o", mfc="none", ls="--", label="M1, true classes"),
         plt.Line2D([], [], color=INK2, marker="D", ls="", mfc="none", label="real scan (circles: variants)")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    save(fig, out, "fig_pq_vs_n")


def fig_semantic_iou(summary, cfg, out):
    S = (summary or {}).get("semseg") or {}
    models = [m for m in cfg["semseg"]["models"] if m in S]
    if not models:
        return placeholder(out, "fig_semantic_iou", "Per-class IoU: waiting for Track A (03a_semseg)")
    classes = cfg["brpcd"]["classes"]
    names = ["girder", "deck", "pier", "main cable", "suspender", "stay"]
    fig, ax = plt.subplots(figsize=(W, 1.7))
    x = np.arange(len(classes))
    wbar = 0.8 / len(models)
    cols = {"pointnet2": BLUE, "pointnet": AQUA}
    for j, m in enumerate(models):
        iou = np.array([np.nan if v is None else 100 * v for v in S[m].get("real", {}).get("iou", [None] * len(classes))])
        ax.bar(x - 0.4 + wbar * (j + 0.5), np.nan_to_num(iou), width=wbar * 0.92, color=cols.get(m, VIOLET),
               label=f"{ {'pointnet2': 'PointNet++', 'pointnet': 'PointNet'}.get(m, m)} (mIoU {100 * S[m]['real']['miou']:.1f})")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("IoU (%), real test scans")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=len(models), loc="upper right")
    fig.tight_layout()
    save(fig, out, "fig_semantic_iou")


def fig_brpcd_examples(results, data, cfg, out, source=None):
    model = source or cfg["merge"]["semseg_model"]
    files = sorted(glob.glob(os.path.join(results, "trackA", "instances", model, "fold_*", "*.json")))
    files = [f for f in files if not f.endswith("tuned.json")]
    picks = {}
    for f in files:
        r = load_json(f)
        if r["is_real"] and (r["type"] not in picks or r["N"] > picks[r["type"]][1]["N"]):
            picks[r["type"]] = (f, r)
    if not picks or not data:
        return placeholder(out, "fig_brpcd_examples", "Example bridges: waiting for Track A (03b_instances)")
    types = [t for t in TYPES if t in picks]
    fig, axes = plt.subplots(len(types), 3, figsize=(W, 1.35 * len(types) + 0.3), squeeze=False)
    for row, t in zip(axes, types):
        f, r = picks[t]
        z = np.load(os.path.join(data, "clouds", f"{r['bridge']}.npz"))
        ins = np.load(f.replace(".json", "_inst.npz"))
        xyz, gt, gcls = z["xyz"], z["inst"].astype(np.int64), z["cls"].astype(int)
        keep = np.random.default_rng(0).permutation(len(xyz))[:120000]
        view = (0, 1) if t == "girder" else (0, 2)
        if t == "girder":                                    # plan view: hide the deck so the girder lines show
            keep = keep[gcls[keep] != cfg["brpcd"]["classes"].index("deck")]
        pal = palette(int(gt.max()) + 1)
        gcol = pal[gt]
        for ax, (lab, inst) in zip(row, (("true members", None), ("M1", ins["m1"]), ("M2", ins["m2"]))):
            if inst is None:
                col = gcol
            else:
                inst = inst.astype(np.int64)
                col = np.tile(np.array(matplotlib.colors.to_rgb(MUTE)), (len(inst), 1))
                for i in np.unique(inst[inst >= 0]):        # colour a predicted member like its matching true member
                    m = inst == i
                    g, c = np.unique(gt[m], return_counts=True)
                    gi = g[np.argmax(c)]
                    if c.max() / (m.sum() + (gt == gi).sum() - c.max()) > 0.5:
                        col[m] = pal[gi]
            ax.scatter(xyz[keep, view[0]], xyz[keep, view[1]], c=col[keep], s=0.15, linewidths=0, rasterized=True)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            pq = "" if inst is None else f"  PQ {100 * r[lab]['PQ']:.1f}"
            ax.set_title(f"{lab}{pq}", fontsize=7)
        row[0].set_ylabel(f"{t}\n{r['bridge']}, N={r['N']}\n" + ("plan, deck hidden" if t == "girder" else "elevation"),
                          fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig_brpcd_examples")


# ----------------------------------------------------------------------------- Track B
def _views(model_dir, it):
    d = os.path.join(model_dir, "test", f"ours_{it}")
    return sorted(glob.glob(os.path.join(d, "renders", "*.png"))), d


def fig_render(results, cfg, out):
    tb = os.path.join(results, "trackB")
    it3, it2 = cfg["gs3d"]["iterations"], cfg["gs2d"]["iterations"]
    r3, d3 = _views(os.path.join(tb, "gs3d"), it3)
    r2, d2 = _views(os.path.join(tb, "gs2d"), it2)
    if not r3 or not r2:
        return placeholder(out, "fig_render", "Held-out views (photo | 3DGS | 2DGS | 2DGS normals): waiting for 03c/03d", 2.0)
    from PIL import Image
    m2 = os.path.join(tb, "gs2d", "metrics_images.json")
    m3 = os.path.join(tb, "gs3d", "metrics_images.json")
    per2 = load_json(m2)["per_view"] if os.path.exists(m2) else {}
    per3 = load_json(m3)["per_view"] if os.path.exists(m3) else {}
    names = [os.path.basename(p) for p in r2 if os.path.exists(os.path.join(d3, "renders", os.path.basename(p)))]
    if per2:
        names = sorted(names, key=lambda n: per2.get(n, {}).get("psnr", 0))
    pick = [names[len(names) // 2], names[0]] if len(names) > 1 else names
    vis = os.path.join(d2, "vis")
    cols = 4 if os.path.isdir(vis) else 3
    fig, axes = plt.subplots(len(pick), cols, figsize=(W, 1.0 * len(pick) * 5.5 / cols * 0.75 + 0.3), squeeze=False)
    for row, n in zip(axes, pick):
        panels = [("photo", os.path.join(d2, "gt", n)), ("3DGS", os.path.join(d3, "renders", n)),
                  ("2DGS", os.path.join(d2, "renders", n))]
        if cols == 4:
            nn = sorted(glob.glob(os.path.join(vis, f"normal_{os.path.splitext(n)[0]}*"))) or \
                 sorted(glob.glob(os.path.join(vis, f"*normal*{os.path.splitext(n)[0]}*")))
            panels.append(("2DGS normals", nn[0] if nn else None))
        for ax, (lab, path) in zip(row, panels):
            ax.axis("off")
            if path and os.path.exists(path):
                ax.imshow(Image.open(path))
            sc = {"3DGS": per3.get(n, {}), "2DGS": per2.get(n, {})}.get(lab, {})
            ax.set_title(lab + (f"  {sc['psnr']:.2f} dB" if sc.get("psnr") else ""), fontsize=7)
    fig.tight_layout()
    save(fig, out, "fig_render")


def _wrap(label, width=14):
    """Break a tick label after a comma when there is one, else at word boundaries."""
    import textwrap
    if len(label) <= width:
        return label
    if ", " in label:
        return label.replace(", ", ",\n", 1)
    return textwrap.fill(label, width)


def fig_measure(summary, out):
    M = (((summary or {}).get("trackB") or {}).get("measure")) or {}
    M = {k: v for k, v in M.items() if v.get("rel_err")}
    if not M:
        return placeholder(out, "fig_measure", "Side-span errors and deck residuals: waiting for 03e_measure")
    order = [k for k in ("2dgs", "3dgs", "2dgs_mean") if k in M] + [k for k in M if k not in ("2dgs", "3dgs", "2dgs_mean")]
    col = {k: c for k, c in zip(order, [BLUE, ORANGE, AQUA, VIOLET])}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.9), gridspec_kw={"width_ratios": [1.2, 1]})
    x = np.arange(2)
    wb = 0.8 / len(order)
    for j, k in enumerate(order):
        e = [100 * M[k]["rel_err"][0], 100 * M[k]["rel_err"][2]]
        a1.bar(x - 0.4 + wb * (j + 0.5), e, width=wb * 0.9, color=col[k], label=M[k].get("label", k))
    a1.axhline(2.0, color=INK2, lw=0.8, ls=":")
    a1.text(1.45, 2.05, "2% target", fontsize=6.5, color=INK2, ha="right", va="bottom")
    a1.set_xticks(x)
    a1.set_xticklabels(["side span 1 ($e_1$)", "side span 3 ($e_3$)"])
    a1.set_ylabel("relative error (%)")
    ymax = max([2.0] + [100 * M[k]["rel_err"][i] for k in order for i in (0, 2)])
    a1.set_ylim(0, 1.35 * ymax)
    a1.legend(frameon=False, fontsize=6.5, loc="upper left", ncol=1)
    data = [np.array(M[k].get("deck_rms_m", {}).get("values") or [np.nan]) * 1000 for k in order]
    bp = a2.boxplot(data, patch_artist=True, widths=0.5, showfliers=False)  # vertical is the default
    for patch, k in zip(bp["boxes"], order):
        patch.set_facecolor(col[k])
        patch.set_alpha(0.35)
        patch.set_edgecolor(col[k])
    for med in bp["medians"]:
        med.set_color(INK)
    a2.set_xticks(range(1, len(order) + 1))
    a2.set_xticklabels([_wrap(M[k].get("label", k)) for k in order])
    a2.set_ylabel("deck patch RMS (mm)")
    fig.tight_layout()
    save(fig, out, "fig_measure")


def fig_cross_section(results, summary, out):
    """One cross-section slice (t, h) per measured point cloud at mid main span: the real version of the
    expected-geometry schematic. Points within +-0.5 m of the slice station, same window and the same
    metre scale for every method; the ground is left out."""
    M = (((summary or {}).get("trackB") or {}).get("measure")) or {}
    order = [k for k in ("2dgs", "3dgs", "2dgs_mean", "vggt") if k in M] + sorted(k for k in M if k not in ("2dgs", "3dgs", "2dgs_mean", "vggt"))
    slices = []
    for k in order:
        f = os.path.join(results, "trackB", "measure", f"aligned_{k}.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        sth = z["stha"].astype(np.float64)
        st = z["stations_m"]
        s_mid = 0.5 * (st[1] + st[2]) if len(st) >= 3 else float(np.median(sth[:, 0]))   # mid main span
        sel = sth[np.abs(sth[:, 0] - s_mid) < 0.5]
        sel = sel[(np.abs(sel[:, 1]) < float(z["half_width_m"]) + 2.0) & (sel[:, 2] > float(z["ground_h_m"]) + 1.0)]
        if len(sel):
            slices.append((k, sel))
    if not slices:
        return placeholder(out, "fig_cross_section", "Cross-sections of the measured point clouds: waiting for 03e_measure")
    fixed = {"2dgs": BLUE, "3dgs": ORANGE, "2dgs_mean": AQUA, "vggt": VIOLET}
    spare = iter([BLUE, ORANGE, AQUA, VIOLET, MUTE])
    col = {k: fixed.get(k) or next(spare) for k, _ in slices}
    allp = np.concatenate([p for _, p in slices])
    t0, t1 = allp[:, 1].min() - 0.5, allp[:, 1].max() + 0.5
    h0, h1 = allp[:, 2].min() - 0.5, allp[:, 2].max() + 0.5
    n = len(slices)
    panel_w = (W - 0.6) / n
    fig_h = min(3.2, max(1.5, panel_w * (h1 - h0) / (t1 - t0) + 0.8))
    fig, axes = plt.subplots(1, n, figsize=(W, fig_h), squeeze=False)
    for ax, (k, sel) in zip(axes[0], slices):
        ax.scatter(sel[:, 1], sel[:, 2], s=0.6, c=col[k], linewidths=0, rasterized=True)
        ax.axhline(0.0, color=INK2, lw=0.6, ls=":")
        ax.set_title(f"{M[k].get('label', k)} ({len(sel):,} pts)", fontsize=7)
        ax.set_xlabel("t (m)")
        ax.set_xlim(t0, t1)
        ax.set_ylim(h0, h1)
        ax.set_aspect("equal", adjustable="box")
    axes[0][0].set_ylabel("h (m), deck plane = 0")
    fig.tight_layout()
    save(fig, out, "fig_cross_section")


def fig_vtti_instances(results, cfg, out):
    p = os.path.join(results, "rq3", "vtti_instances.npz")
    if not os.path.exists(p):
        return placeholder(out, "fig_vtti_instances", "VTTI members (RQ3): waiting for 03f_merge", 2.0)
    z = np.load(p)
    xyz, cls, m2 = z["xyz"], z["cls"].astype(int), z["m2"].astype(np.int64)
    keep = np.random.default_rng(0).permutation(len(xyz))[:200000]
    ci = {c: i for i, c in enumerate(cfg["brpcd"]["classes"])}
    ccol = np.tile(np.array(matplotlib.colors.to_rgb(MUTE)), (len(xyz), 1))
    for name, c in (("girder", BLUE), ("deck", AQUA), ("pier", ORANGE)):
        ccol[cls == ci[name]] = matplotlib.colors.to_rgb(c)
    pal = palette(int(m2.max()) + 2)
    icol = np.where((m2 >= 0)[:, None], pal[np.maximum(m2, 0)], np.array(matplotlib.colors.to_rgb(MUTE)))
    no_deck = keep[cls[keep] != ci["deck"]]
    mj = os.path.join(results, "rq3", "merge.json")
    counts = ""
    if os.path.exists(mj):
        mm = load_json(mj)
        g2, p2, g1 = (mm.get("M2") or {}).get("girders"), (mm.get("M2") or {}).get("piers"), (mm.get("M1") or {}).get("girders")
        if g2 is not None:
            counts = f": {g2} girders, {p2} piers" + (f" (M1 finds {g1} girders)" if g1 is not None else "")
    # an elevation coloured by member is left out on purpose: parallel girders overlap in that view
    panels = [(ccol, keep, (0, 2), "elevation, coloured by predicted class"),
              (ccol, no_deck, (0, 1), "plan with the deck hidden, coloured by predicted class"),
              (icol, no_deck, (0, 1), "plan with the deck hidden, coloured by M2 member" + counts)]
    span = np.ptp(xyz[:, 0]) or 1.0
    hts = [max(0.35, 4.9 * np.ptp(xyz[:, v[1]]) / span) for _, _, v, _ in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(W, sum(hts) + 1.1), gridspec_kw={"height_ratios": hts})
    for ax, (col, idx, (vx, vy), title) in zip(axes, panels):
        ax.scatter(xyz[idx, vx], xyz[idx, vy], c=col[idx], s=0.12, linewidths=0, rasterized=True)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=7, loc="left")
    h = [plt.Line2D([], [], color=c, marker="s", ls="", label=l) for l, c in
         (("girder", BLUE), ("deck", AQUA), ("pier", ORANGE), ("other classes", MUTE))]
    fig.tight_layout()
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))  # above the strips
    save(fig, out, "fig_vtti_instances")


def main():
    global STAMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--example-source", default=None, help="class source for fig_brpcd_examples (default merge.semseg_model)")
    a = ap.parse_args()
    STAMP = a.stamp
    cfg = load_cfg(a.cfg)
    sp = os.path.join(a.results, "summary", "summary.json")
    summary = load_json(sp) if os.path.exists(sp) else None
    log(f"figures -> {a.out}")
    for fn in (lambda: fig_pq_vs_n(a.results, cfg, a.out), lambda: fig_semantic_iou(summary, cfg, a.out),
               lambda: fig_brpcd_examples(a.results, a.data, cfg, a.out, a.example_source), lambda: fig_render(a.results, cfg, a.out),
               lambda: fig_measure(summary, a.out), lambda: fig_vtti_instances(a.results, cfg, a.out),
               lambda: fig_cross_section(a.results, summary, a.out)):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            log(f"  figure failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
