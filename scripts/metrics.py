"""
metrics.py -- evaluation used in Section 4.3 of the paper.

    semantic_iou        Eq. (20): per-class IoU and mIoU over the classes present
    panoptic_quality    Eq. (21): PQ = SQ x RQ per class on point sets (IoU > 0.5 matching)
    trend_slope         Eq. (22): least-squares slope of PQ against log N
    bootstrap_slope_diff   95% interval of slope(M2) - slope(M1), resampling whole bridges
    psnr                Eq. (23)
"""
import numpy as np


# ----------------------------------------------------------------------------- semantics
def confusion(gt, pred, n_cls):
    m = (gt >= 0) & (gt < n_cls) & (pred >= 0) & (pred < n_cls)
    return np.bincount(n_cls * gt[m] + pred[m], minlength=n_cls * n_cls).reshape(n_cls, n_cls)


def semantic_iou(gt, pred, n_cls, cm=None):
    cm = confusion(gt, pred, n_cls) if cm is None else cm
    tp = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - np.diag(cm)
    present = cm.sum(1) > 0
    iou = np.full(n_cls, np.nan)
    iou[present] = tp[present] / np.maximum(union[present], 1)
    return {"iou": iou.tolist(), "miou": float(np.nanmean(iou)) if present.any() else float("nan"),
            "present": present.tolist(), "oa": float(tp.sum() / max(cm.sum(), 1)), "confusion": cm.tolist()}


# ----------------------------------------------------------------------------- panoptic
def _segments(cls, inst, c):
    m = (cls == c) & (inst >= 0)
    ids, cnt = np.unique(inst[m], return_counts=True)
    return m, dict(zip(ids.tolist(), cnt.tolist()))


def panoptic_quality(gt_cls, gt_inst, pr_cls, pr_inst, classes, iou_thr=0.5, min_seg=30):
    """
    Per class c: true segments = points with gt_cls == c grouped by gt_inst; predicted segments =
    points with pr_cls == c grouped by pr_inst (segments below `min_seg` points are dropped as
    noise). A pair matches when IoU > iou_thr (unique by construction). Returns per-class
    PQ/SQ/RQ/TP/FP/FN and their mean over the classes that have true segments.
    """
    out = {}
    for c in classes:
        gm, gseg = _segments(gt_cls, gt_inst, c)
        pm, pseg = _segments(pr_cls, pr_inst, c)
        pseg = {k: v for k, v in pseg.items() if v >= min_seg}
        if not gseg:
            if pseg:
                out[int(c)] = {"PQ": None, "SQ": None, "RQ": None, "TP": 0, "FP": len(pseg), "FN": 0, "n_true": 0}
            continue
        both = gm & pm
        pairs, inter = np.unique(np.stack([gt_inst[both], pr_inst[both]], 1), axis=0, return_counts=True) \
            if both.any() else (np.zeros((0, 2), int), np.zeros(0, int))
        tp_iou, matched_g, matched_p = [], set(), set()
        for (g, p_), n_int in zip(pairs.tolist(), inter.tolist()):
            if p_ not in pseg:
                continue
            iou = n_int / (gseg[g] + pseg[p_] - n_int)
            if iou > iou_thr:
                tp_iou.append(iou)
                matched_g.add(g)
                matched_p.add(p_)
        tp, fp, fn = len(tp_iou), len(pseg) - len(matched_p), len(gseg) - len(matched_g)
        sq = float(np.mean(tp_iou)) if tp else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0
        out[int(c)] = {"PQ": sq * rq, "SQ": sq, "RQ": rq, "TP": tp, "FP": fp, "FN": fn, "n_true": len(gseg)}
    valid = [v for v in out.values() if v["n_true"] > 0]
    mean = {k: float(np.mean([v[k] for v in valid])) if valid else float("nan") for k in ("PQ", "SQ", "RQ")}
    return {"per_class": out, **mean, "N": int(sum(v["n_true"] for v in valid))}


# ----------------------------------------------------------------------------- trends
def trend_slope(N, pq):
    """Slope beta_1 of PQ = beta_0 + beta_1 log N (Eq. 22); NaN for fewer than 3 points.

    None (a failed fit, not a score of zero) becomes NaN and is dropped by the finite mask.
    """
    N = np.array([np.nan if v is None else v for v in np.asarray(N, dtype=object)], float)
    pq = np.array([np.nan if v is None else v for v in np.asarray(pq, dtype=object)], float)
    ok = np.isfinite(N) & np.isfinite(pq) & (N > 0)
    if ok.sum() < 3 or np.ptp(np.log(N[ok])) == 0:
        return float("nan"), float("nan")
    b1, b0 = np.polyfit(np.log(N[ok]), pq[ok], 1)
    return float(b1), float(b0)


def bootstrap_slope_diff(rows, n_boot=2000, ci=0.95, seed=0):
    """
    rows: list of dicts with keys group, N, pq_m1, pq_m2 (one per test cloud). Resamples whole
    groups (a real scan together with its variants) and returns the slope difference
    beta1(M2) - beta1(M1) with its percentile interval.
    """
    rng = np.random.default_rng(seed)
    groups = sorted({r["group"] for r in rows})
    by_g = {g: [r for r in rows if r["group"] == g] for g in groups}

    def diff(sample):
        N = [r["N"] for r in sample]
        return trend_slope(N, [r["pq_m2"] for r in sample])[0] - trend_slope(N, [r["pq_m1"] for r in sample])[0]

    est = diff(rows)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(groups, size=len(groups), replace=True)
        d = diff([r for g in pick for r in by_g[g]])
        if np.isfinite(d):
            boots.append(d)
    if len(boots) < 10:
        return {"diff": est, "lo": float("nan"), "hi": float("nan"), "n_boot": len(boots)}
    a = (1 - ci) / 2
    return {"diff": est, "lo": float(np.quantile(boots, a)), "hi": float(np.quantile(boots, 1 - a)),
            "n_boot": len(boots)}


# ----------------------------------------------------------------------------- images
def psnr(img, ref):
    """Eq. (23) for images scaled to [0, 1]."""
    mse = float(np.mean((np.asarray(img, float) - np.asarray(ref, float)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
