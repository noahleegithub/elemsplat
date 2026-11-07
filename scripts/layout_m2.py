"""
layout_m2.py -- M2, layout-guided instance assignment (paper Section 3.3, Eqs. 1, 2, 5-7).

fit_layout() fits the structural layout T = (frame, stations sigma, girder offsets tau,
cable planes and lines) to points with (predicted or true) classes; assign() gives every
point the instance of the layout cell it falls in. The instance UNIT of each class follows
the BrPCD ground truth (see infer_units): girders and deck either whole or split at the
supports, piers per bent or per column, main cables per side or per side and span.
"""
import numpy as np

from geometry import edge_stations, frame_from_deck, kde_peaks, point_line_dist, sequential_lines

DEFAULT_UNITS = {"girder": "by_span", "deck": "by_span", "pier": "bent", "main_cable": "side"}


# ----------------------------------------------------------------------------- helpers
def _mask(cls, ci, name):
    return cls == ci[name] if name in ci else np.zeros(len(cls), bool)


def _frame(xyz, cls, ci, p, seed):
    deck = _mask(cls, ci, "deck")
    src = xyz[deck] if deck.sum() >= 200 else xyz[_mask(cls, ci, "girder") | deck]
    if len(src) < 200:
        src = xyz
    frame, _ = frame_from_deck(src, p["deck_ransac_thresh"], p.get("ransac_iters", 1000), seed,
                               ref_up=(0.0, 0.0, 1.0), max_angle_deg=30.0)
    return frame


def stations_from_piers(s_pier, s_deck, p):
    """Pier peaks (Eq. 5) plus the superstructure ends where no pier stands near the end.
    Ends = outermost 0.25 m bins of s_deck (girder points if any, else deck) holding at least
    0.2 x the median bin count; unlike percentiles this has no inward bias on long bridges."""
    bw, eta = p["kde_bw_station"], p["peak_eta"]
    min_sep = p.get("station_min_sep", 5.0)
    peaks = kde_peaks(s_pier, bw, eta, min_sep=min_sep)[0] if len(s_pier) >= 10 else np.zeros(0)
    if len(s_deck) >= 10:
        lo, hi = edge_stations(s_deck, p.get("end_edge_bin", 0.25), p.get("end_edge_frac", 0.2))
    elif len(peaks):
        lo, hi = peaks.min(), peaks.max()
    else:
        lo, hi = 0.0, 0.0
    end_margin = p.get("end_margin", 3.0)
    st = list(peaks)
    if not len(peaks) or peaks.min() - lo > end_margin:
        st.append(lo)
    if not len(peaks) or hi - peaks.max() > end_margin:
        st.append(hi)
    return np.array(sorted(st)), peaks


def span_index(s, stations):
    """k(i) = 1 + #{interior stations <= s}, clipped to 1..K (K = len(stations) - 1)."""
    K = max(1, len(stations) - 1)
    k = 1 + np.searchsorted(stations[1:-1], s, side="right") if len(stations) > 2 else np.ones(len(s), int)
    return np.clip(k, 1, K)


# ----------------------------------------------------------------------------- fit
def fit_layout(xyz, cls, ci, p, seed=0):
    frame = _frame(xyz, cls, ci, p, seed)
    s, t, h = frame.local(xyz)
    g = _mask(cls, ci, "girder")
    ends_src = g if g.sum() >= 50 else (_mask(cls, ci, "deck") | g)
    stations, pier_peaks = stations_from_piers(s[_mask(cls, ci, "pier")], s[ends_src], p)

    offsets = kde_peaks(t[g], p["kde_bw_offset"], p["peak_eta"], min_sep=p.get("offset_min_sep", 0.6),
                        valley_frac=p.get("offset_valley_frac", 0.15))[0] if g.sum() >= 10 else np.zeros(0)
    if g.sum() and not len(offsets):
        offsets = np.array([float(np.median(t[g]))])  # one girder line (e.g. a box girder)

    deck = _mask(cls, ci, "deck")
    t_centre = float(np.median(t[deck])) if deck.sum() else float(np.median(t))

    cables = _mask(cls, ci, "stay") | _mask(cls, ci, "suspender")
    planes, lines = [], []
    if cables.sum() >= p.get("cable_min_inliers", 40):
        planes = kde_peaks(t[cables], p["kde_bw_offset"], p["peak_eta"], min_sep=p.get("cable_plane_min_sep", 1.0))[0]
        if not len(planes):
            planes = np.array([float(np.median(t[cables]))])
        loc = np.stack([s, t, h], 1)
        plane_of = np.argmin(np.abs(t[cables, None] - planes[None, :]), 1)
        idx_c = np.flatnonzero(cables)
        for pi in range(len(planes)):
            sel = idx_c[plane_of == pi]
            if len(sel) < p.get("cable_min_inliers", 40):
                continue
            L, _ = sequential_lines(loc[sel], p["cable_thresh"], p.get("cable_min_inliers", 40),
                                    p.get("cable_max_lines", 600), seed=seed + pi,
                                    local_radius=p.get("cable_local_radius", 8.0))
            for (o, d) in L:
                lines.append({"plane": pi, "o": o, "d": d, "anchor": _anchor(o, d)})
        lines = _gap_check(loc, cables, lines, planes, p, seed)
    return {"frame": frame, "stations": stations, "pier_peaks": pier_peaks, "offsets": offsets,
            "t_centre": t_centre, "cable_planes": np.asarray(planes, float), "lines": lines}


def _anchor(o, d):
    """Station where the line meets the deck plane h = 0 (None for near-horizontal lines)."""
    if abs(d[2]) < 0.1:
        return None
    r = -o[2] / d[2]
    return float(o[0] + r * d[0])


def _gap_check(loc, cables, lines, planes, p, seed):
    """Anchors in one plane should be close to evenly spaced: refit inside large gaps."""
    factor = p.get("anchor_gap_factor", 1.5)
    out = list(lines)
    idx_c = np.flatnonzero(cables)
    for pi in range(len(planes)):
        anc = sorted(l["anchor"] for l in lines if l["plane"] == pi and l["anchor"] is not None)
        if len(anc) < 4:
            continue
        gaps = np.diff(anc)
        med = np.median(gaps)
        for j in np.flatnonzero(gaps > factor * med):
            lo, hi = anc[j] + 0.25 * med, anc[j + 1] - 0.25 * med
            pts = loc[idx_c]
            near_plane = np.abs(pts[:, 1] - planes[pi]) < 0.5 * p.get("cable_plane_min_sep", 1.0) + p["cable_thresh"]
            in_gap = (pts[:, 0] > lo) & (pts[:, 0] < hi) & (pts[:, 2] < 0.25 * (hi - lo) + 1.0) & near_plane
            if in_gap.sum() < p.get("cable_min_inliers", 40) // 2:
                continue
            L, _ = sequential_lines(pts[in_gap], 0.5 * p["cable_thresh"], max(5, p.get("cable_min_inliers", 40) // 2),
                                    max_lines=3, seed=seed + 1000 + j)
            for (o, d) in L:
                out.append({"plane": pi, "o": o, "d": d, "anchor": _anchor(o, d), "from_gap": True})
    return out


# ----------------------------------------------------------------------------- assign
def assign(xyz, cls, ci, layout, units=None, p=None):
    """Eq. (7). Returns int instance ids (-1 = unassigned) unique across classes, and the keys."""
    units = dict(DEFAULT_UNITS, **(units or {}))
    p = p or {}
    frame = layout["frame"]
    s, t, h = frame.local(xyz)
    st = layout["stations"]
    k = span_index(s, st)
    inst = np.full(len(xyz), -1, np.int64)
    keys = {}

    def put(mask, key_arrays, cid):
        """Give the points in `mask` one instance per distinct key tuple (class id first)."""
        if not mask.any():
            return
        kk = np.stack(key_arrays, 1) if len(key_arrays) > 1 else np.asarray(key_arrays[0])[:, None]
        uniq, inv = np.unique(kk, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
        base = len(keys)
        for j, u in enumerate(uniq):
            keys[base + j] = (int(cid),) + tuple(int(x) for x in u)
        inst[mask] = base + inv

    # girders: transverse line g(i) (x span k(i))
    if "girder" in ci:
        cname_id = ci["girder"]
        m = cls == cname_id
        if m.any():
            off = layout["offsets"] if len(layout["offsets"]) else np.array([0.0])
            g = np.argmin(np.abs(t[m, None] - off[None, :]), 1)
            put(m, [g, k[m]] if units["girder"] == "by_span" else [g], cname_id)
    if "deck" in ci:
        cname_id = ci["deck"]
        m = cls == cname_id
        put(m, [k[m]] if units["deck"] == "by_span" else [np.zeros(m.sum(), int)], cname_id)
    if "pier" in ci:
        cname_id = ci["pier"]
        m = cls == cname_id
        if m.any():
            peaks = layout["pier_peaks"] if len(layout["pier_peaks"]) else st
            b = np.argmin(np.abs(s[m, None] - peaks[None, :]), 1)
            if units["pier"] == "column":
                col = np.zeros(m.sum(), int)
                idx = np.flatnonzero(m)
                for bi in np.unique(b):
                    sel = b == bi
                    cpk = kde_peaks(t[idx[sel]], p.get("kde_bw_offset", 0.15) * 3, p.get("peak_eta", 0.15),
                                    min_sep=1.0)[0] if sel.sum() > 20 else np.zeros(0)
                    if len(cpk) > 1:
                        col[sel] = np.argmin(np.abs(t[idx[sel], None] - cpk[None, :]), 1)
                put(m, [b, col], cname_id)
            else:
                put(m, [b], cname_id)
    if "main_cable" in ci:
        cname_id = ci["main_cable"]
        m = cls == cname_id
        side = (t[m] > layout["t_centre"]).astype(int)
        put(m, [side, k[m]] if units["main_cable"] == "side_span" else [side], cname_id)
    for cname in ("stay", "suspender"):
        if cname not in ci:
            continue
        cname_id = ci[cname]
        m = cls == cname_id
        if not m.any():
            continue
        L = layout["lines"]
        if not L:
            continue
        loc = np.stack([s[m], t[m], h[m]], 1)
        D = np.stack([point_line_dist(loc, l["o"], l["d"]) for l in L], 1)
        j = np.argmin(D, 1)
        far = D[np.arange(len(j)), j] > 3.0 * p.get("cable_thresh", 0.15)
        j = j.astype(np.int64)
        idx = np.flatnonzero(m)
        good = ~far
        if good.any():
            put_mask = np.zeros(len(xyz), bool)
            put_mask[idx[good]] = True
            put(put_mask, [j[good]], cname_id)
    return inst, keys


def run_m2(xyz, cls, ci, p, units=None, seed=0):
    layout = fit_layout(xyz, cls, ci, p, seed)
    inst, keys = assign(xyz, cls, ci, layout, units, p)
    info = {"n_stations": int(len(layout["stations"])), "stations": layout["stations"].tolist(),
            "n_piers": int(len(layout["pier_peaks"])), "n_girder_lines": int(len(layout["offsets"])),
            "offsets": layout["offsets"].tolist(), "n_cable_planes": int(len(layout["cable_planes"])),
            "n_cable_lines": int(len(layout["lines"])), "frame": layout["frame"].to_dict()}
    return inst, info, layout


# ----------------------------------------------------------------------------- units
def infer_units(xyz, cls, inst, ci, p):
    """Instance unit of each class from the ground truth (does a member cross a support?)."""
    units = dict(DEFAULT_UNITS)
    try:
        frame = _frame(xyz, cls, ci, p, 0)
    except ValueError:
        return units
    s, t, _ = frame.local(xyz)
    deck_like = _mask(cls, ci, "deck") | _mask(cls, ci, "girder")
    st, pier_peaks = stations_from_piers(s[_mask(cls, ci, "pier")], s[deck_like], p)
    interior = st[1:-1]

    def crosses(name):
        m = _mask(cls, ci, name)
        ids = np.unique(inst[m])
        if not len(ids) or not len(interior):
            return None
        n_cross = 0
        for i in ids:
            sv = s[m & (inst == i)]
            if len(sv) < 10:
                continue
            a, b = np.percentile(sv, [5, 95])
            n_cross += int(((interior > a + 1.0) & (interior < b - 1.0)).any())
        return n_cross / max(1, len(ids))

    for name in ("girder", "deck"):
        f = crosses(name)
        if f is not None:
            units[name] = "whole" if f > 0.5 else "by_span"
    m = _mask(cls, ci, "pier")
    if m.any() and len(pier_peaks):
        ids = np.unique(inst[m])
        cent = np.array([np.median(s[m & (inst == i)]) for i in ids])
        per_station = np.bincount(np.argmin(np.abs(cent[:, None] - pier_peaks[None, :]), 1), minlength=len(pier_peaks))
        units["pier"] = "column" if per_station.mean() > 1.5 else "bent"
    m = _mask(cls, ci, "main_cable")
    if m.any():
        units["main_cable"] = "side_span" if len(np.unique(inst[m])) > 2 else "side"
    return units
