#!/usr/bin/env python
"""
measure_vtti.py -- metric checks on a VTTI reconstruction (paper Section 3.4, Eqs. 17-19).

    python measure_vtti.py --points pts_2dgs.npz --method 2dgs --label 2DGS \
        --sparse $VTTI_SCENE/sparse/0 --trackb $RESULTS_DIR/trackB

The first run (2DGS) fixes the shared bridge frame and the pier boxes (trackB/measure/frame.json);
later runs (3DGS) reuse them, so both reconstructions are measured with the same boxes.
Boxes and end stations can be overridden in configs/vtti_overrides.yaml (metres in the frame
with the provisional scale kappa0 written by vtti_prepare.py).
Outputs in trackB/measure/: measure_<method>.json, aligned_<method>.npz (metric s,t,h),
pier_boxes_auto.yaml and diag_<method>.png.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colmap_io as cio  # noqa: E402
from bridge_frame import deck_and_ground, frame_from_deck_plane, horizontal_planes, up_from_cameras  # noqa: E402
from common import FT_TO_M, load_cfg, load_json, load_yaml, log, pkg_path, save_json  # noqa: E402
from geometry import Frame, edge_stations, fit_plane_svd, kde_peaks, umeyama  # noqa: E402


def enu(lat, lon, alt):
    lat, lon, alt = map(np.asarray, (lat, lon, alt))
    R = 6378137.0
    la0, lo0 = np.deg2rad(lat.mean()), np.deg2rad(lon.mean())
    return np.stack([R * (np.deg2rad(lon) - lo0) * np.cos(la0), R * (np.deg2rad(lat) - la0), alt - alt.mean()], 1)


def gps_scale(names, centres, gps_csv, trim=10.0):
    """Metric scale from EXIF GPS, horizontally only.

    Fitting all three axes is dominated by altitude noise (EXIF altitude scatters by ~10 m on a
    UAV), which inflated the residual to ~7 m and hid whether the scale was trustworthy. The
    horizontal fit drops the worst `trim` percent once and reports the residual, so the caller can
    judge the estimate instead of trusting it blindly.
    """
    import csv as _csv
    if not os.path.exists(gps_csv):
        return None
    rows = {r["name"]: r for r in _csv.DictReader(open(gps_csv))}
    names = [n for n in names if n in rows]
    if len(names) < 3:
        return None
    lat = np.array([float(rows[n]["lat"]) for n in names])
    lon = np.array([float(rows[n]["lon"]) for n in names])
    R = 6378137.0
    la0, lo0 = np.deg2rad(lat.mean()), np.deg2rad(lon.mean())
    X = np.column_stack([R * (np.deg2rad(lon) - lo0) * np.cos(la0), R * (np.deg2rad(lat) - la0)])
    C = np.array([centres[names.index(n)] for n in names])[:, :2]
    E3 = np.column_stack([X, np.zeros(len(X))])
    C3 = np.column_stack([C, np.zeros(len(C))])
    s, Rm, t = umeyama(C3, E3)
    res = np.linalg.norm((s * C3 @ Rm.T + t)[:, :2] - X, axis=1)
    keep = res <= np.percentile(res, 100.0 - trim) if trim > 0 else np.ones(len(res), bool)
    if keep.sum() >= 4:
        s, Rm, t = umeyama(C3[keep], E3[keep])
        res = np.linalg.norm((s * C3[keep] @ Rm.T + t)[:, :2] - X[keep], axis=1)
    rms = float(np.sqrt(np.mean(res ** 2)))
    log(f"  GPS scale: {s:.4f} m per SfM unit from {int(keep.sum())}/{len(keep)} cameras, "
        f"horizontal residual RMS {rms:.2f} m")
    return {"kappa": float(s), "n": int(keep.sum()), "rms_m": rms,
            "rel_uncertainty": float(rms / max(np.ptp(X[:, 0]), 1e-9))}


def frame_from_trajectory(xyz, centres, vdirs, mc, k, seed=0):
    """Deck frame anchored on the flight path instead of on the largest plane.

    A RANSAC plane over an outdoor UAV cloud locks onto terrain or approach road (on VTTI it
    reported a 29 m deck half-width), which puts h = 0 in the wrong place and makes every
    height-based test - girder band, pier band, deck flatness - meaningless. The camera centres
    carry the same information without a plane fit: the UAV flies the corridor, so the axis is a
    2-D PCA of the trajectory, and the deck is the upper envelope of the cloud under that corridor.

    `k` is the metric scale (m per SfM unit): the cloud is in SfM units, so every length in the
    config has to be converted before it is compared with a coordinate.

    Returns (Frame with a height profile, diagnostics dict).
    """
    up = up_from_cameras(vdirs)
    C = centres - centres.mean(0)
    C_pl = C - np.outer(C @ up, up)                       # trajectory projected into plan
    _, _, vt = np.linalg.svd(C_pl, full_matrices=False)
    u = vt[0] - (vt[0] @ up) * up
    if np.linalg.norm(u) < 1e-9:
        raise ValueError("degenerate trajectory: cameras are collinear in plan")
    u = u / np.linalg.norm(u)
    w = np.cross(up, u)
    w = w / np.linalg.norm(w)

    q = xyz - centres.mean(0)
    s_all, t_all, z_all = q @ u, q @ w, xyz @ up
    cw = float(mc.get("corridor_halfwidth_m", 12.0)) / max(k, 1e-9)     # m -> SfM units
    corridor = np.abs(t_all) <= cw
    if corridor.sum() < 1000:
        raise ValueError(f"only {int(corridor.sum())} points within {cw * k:.0f} m of the flight path")

    # upper envelope: topmost surface under the corridor, per along-axis bin
    binw = float(mc.get("envelope_bin_m", 2.0)) / max(k, 1e-9)
    pct = float(mc.get("envelope_pct", 90.0))
    s_lo, s_hi = np.percentile(s_all[corridor], [1, 99])
    edges = np.arange(s_lo, s_hi + binw, binw)
    ks, kh = [], []
    for lo in edges[:-1]:
        sel = corridor & (s_all >= lo) & (s_all < lo + binw)
        if sel.sum() < 30:
            continue
        ks.append(0.5 * (lo + lo + binw))
        kh.append(float(np.percentile(z_all[sel], pct)))
    if len(ks) < 4:
        raise ValueError(f"envelope got only {len(ks)} usable bins")
    ks, kh = np.array(ks), np.array(kh)
    # light smoothing of the envelope (crown is smooth; single bins are not)
    if len(ks) >= 5:
        k = max(3, (len(ks) // 8) | 1)
        kh = np.convolve(kh, np.ones(k) / k, mode="same")
        kh[: k // 2], kh[-(k // 2):] = kh[k // 2], kh[-1 - k // 2]
    d = -float(np.median(kh))
    kh = kh + d                                             # h = 0 on the envelope
    origin = centres.mean(0) - (centres.mean(0) @ up) * up   # on the plan of the trajectory
    origin = origin + float(np.median(t_all[corridor])) * w  # keep the corridor centred at t = 0
    fr = Frame(origin, u, w, up, d, deck=(ks, kh))

    # diagnostics: measure the deck's own transverse extent and roughness, not the corridor cap
    s, t, h = fr.local(xyz)
    band = float(mc.get("deck_surface_band_m", 0.30)) / max(k, 1e-9)
    on_deck = (np.abs(h) <= band) & corridor
    hw = float(np.percentile(np.abs(t[on_deck]), 98)) * k if on_deck.sum() > 100 else float("nan")
    rms = float(np.sqrt(np.mean(h[on_deck] ** 2))) * k if on_deck.sum() > 100 else float("nan")
    diag = {"frame_source": "trajectory", "corridor_halfwidth_m": cw * k,
            "deck_halfwidth_m": hw, "deck_rms_m": rms,
            "n_deck_points": int(on_deck.sum()), "n_bins": int(len(ks)),
            "envelope_range_m": [float(ks.min()) * k, float(ks.max()) * k]}
    log(f"  trajectory frame: axis from {len(centres)} cameras, corridor +-{cw * k:.0f} m, "
        f"{len(ks)} envelope bins")
    log(f"  deck half width {hw:.2f} m, RMS {1e3 * rms:.0f} mm, "
        f"{int(on_deck.sum()):,} surface points, envelope spans {(ks.max() - ks.min()) * k:.1f} m")
    return fr, diag


def make_frame(xyz, vdirs, k0, mc, total_m, seed=0, gps_k=None):
    """
    Shared frame from the dense cloud.

    The scale comes from GPS when it is available: fitting kappa so that the girder band equals
    the known bridge length is circular (the band includes the approach roads at both ends, which
    made kappa 23 % too small on VTTI). Without GPS the girder-band fit is the only option, and
    the discrepancy it implies is reported rather than hidden.
    """
    up = up_from_cameras(vdirs)
    sub = xyz[:: max(1, len(xyz) // 1500000)]
    k = float(gps_k) if gps_k else k0
    if gps_k:
        log(f"  frame: scale fixed by GPS kappa {k:.4f} m per SfM unit (girder-band fit skipped)")
    for it in range(3):
        planes = horizontal_planes(sub, up, mc["deck_ransac_thresh_m"] / k, mc["deck_normal_max_deg"], seed=seed)
        if not planes:
            raise SystemExit("no near-horizontal plane found in the point cloud")
        deck, ground = deck_and_ground(planes)
        fr = frame_from_deck_plane(sub, deck)
        _, t, _ = fr.local(sub[deck["idx"]])
        fr.origin = fr.origin + float(np.median(t)) * fr.w       # deck centreline at t = 0
        s, t, h = fr.local(sub)
        hw = float(np.percentile(np.abs(fr.local(sub[deck["idx"]])[1]), 98))
        gb = (h * k >= mc["girder_band_m"][0]) & (h * k <= mc["girder_band_m"][1]) & (np.abs(t) * k <= mc.get("deck_halfwidth_max_m", 12.0))
        if gb.sum() < 50:
            break
        lo, hi = edge_stations(s[gb], mc.get("end_edge_bin_m", 0.25) / k, mc.get("end_edge_frac", 0.2))
        if gps_k:
            ext = (hi - lo) * k
            log(f"  frame: girder band spans {ext:.1f} m against a {total_m:.1f} m bridge "
                f"({100 * (ext / total_m - 1):+.1f}% -- approach road in the band is expected)")
            break
        k_new = total_m / max(hi - lo, 1e-9)
        log(f"  frame iteration {it}: provisional scale {k:.4f} -> {k_new:.4f} m per SfM unit (girder band)")
        if abs(k_new - k) / k < 0.01:
            k = k_new
            break
        k = k_new
    ground_h = float(np.median(fr.local(sub[ground["idx"]])[2])) if ground is not None else float(np.percentile(h, 1))
    return fr, hw, ground_h, k


def auto_boxes(s, t, h, k0, hw, ground_h, mc, sep_target_m=None, sep_tol=None):
    """Pier candidates: below the girders, above the ground, under the deck -> KDE peaks along s."""
    sep_tol = float(mc.get("pier_sep_tol", 0.12)) if sep_tol is None else sep_tol
    sm, tm, hm = s * k0, t * k0, h * k0
    gh_m, hw_m = ground_h * k0, hw * k0
    # Piers sit under the deck, so their transverse extent cannot exceed the bridge width. When
    # the fitted "deck" is terrain the reported half-width is tens of metres and the filter stops
    # filtering, so cap it at a plausible girder-bridge half width.
    t_limit = min(hw_m + 1.0, float(mc.get("deck_halfwidth_max_m", 12.0)))
    # deck_and_ground() calls the lowest large plane "ground", but a plane 1 m under the deck is
    # the slab soffit or a girder plane, not terrain: the pier band (h < -pier_below_deck_m and
    # h > ground + clearance) is then empty by construction and no pier is ever found. When the
    # detected ground cannot be under the piers, fall back to a low percentile of the cloud.
    if gh_m > -mc["pier_below_deck_m"]:
        pct = mc.get("ground_pct_fallback", 2.0)
        floor = float(np.percentile(hm, pct))
        log(f"  ground plane {gh_m:.2f} m is above the pier depth {-mc['pier_below_deck_m']:.2f} m "
            f"(soffit, not terrain) -> floor = {pct}th percentile {floor:.2f} m")
        gh_m = floor
    cand = (hm < -mc["pier_below_deck_m"]) & (hm > gh_m + mc["ground_clearance_m"]) & (np.abs(tm) < t_limit)
    if cand.sum() < 50:
        return [], cand
    pk, grid, dens = kde_peaks(sm[cand], mc["pier_kde_bw_m"], 0.15, min_sep=10.0)
    if len(pk) < 1:
        return [], cand
    if len(pk) > 1 and sep_target_m:
        # The pier separation is a PREDICTION, not a free parameter: kappa comes from GPS and the
        # main span is known, so two piers must sit sep_target_m apart. Among candidate peaks pick
        # the pair closest to that separation; "the two strongest peaks" otherwise turns terrain
        # features into two false piers. Tolerance is wide on purpose -- it selects a pair, it does
        # not tune one.
        tol = sep_tol * sep_target_m
        best, best_score = None, -1.0
        strength = np.interp(pk, grid, dens)
        for i in range(len(pk)):
            for j in range(i + 1, len(pk)):
                d = abs(pk[j] - pk[i])
                if abs(d - sep_target_m) <= tol:
                    score = strength[i] + strength[j]
                    if score > best_score:
                        best, best_score = (pk[i], pk[j]), score
        if best is None:
            log(f"  no peak pair within {100 * sep_tol:.0f}% of the predicted {sep_target_m:.1f} m pier "
                f"separation (peaks at {np.round(pk, 1).tolist()})")
            pk = np.sort(pk[np.argsort(strength)[-2:]]) if len(pk) > 2 else pk
        else:
            pk = np.array(sorted(best))
            log(f"  pier pair selected by predicted separation {sep_target_m:.1f} m: "
                f"{np.round(pk, 1).tolist()} ({abs(pk[1] - pk[0]):.1f} m apart)")
    elif len(pk) > 2:                                          # keep the two strongest peaks
        strength = np.interp(pk, grid, dens)
        pk = np.sort(pk[np.argsort(strength)[-2:]])
    hb = mc["pier_box_halfwidth_m"]
    boxes = [{"s": [float(p - hb), float(p + hb)], "t": [float(-hw_m - 1.0), float(hw_m + 1.0)],
              "h": [float(gh_m + mc["ground_clearance_m"]), float(-mc["pier_below_deck_m"])]} for p in pk]
    return boxes, cand


def in_box(sm, tm, hm, b):
    return ((sm >= b["s"][0]) & (sm <= b["s"][1]) & (tm >= b["t"][0]) & (tm <= b["t"][1]) &
            (hm >= b["h"][0]) & (hm <= b["h"][1]))


def deck_rms(s, t, h, lo, hi, hw, patch, min_pts, band=(-0.15, 0.15)):
    """
    Eq. (19): RMS of a least-squares plane per square deck patch (metric inputs). Only points
    within `band` of the deck plane count, which keeps the slab underside (about 0.2 m below)
    and the parapets out; the share of patch points inside the band is reported with it.
    """
    m = (h > band[0]) & (h < band[1]) & (np.abs(t) < hw - 0.5) & (s > lo) & (s < hi)
    P = np.stack([s[m], t[m], h[m]], 1)
    if len(P) < min_pts:
        return {"median": None, "n_patches": 0}
    ij = np.floor(P[:, :2] / patch).astype(np.int64)
    key = ij[:, 0] * 1000003 + ij[:, 1]
    order = np.argsort(key)
    key, P = key[order], P[order]
    cuts = np.flatnonzero(np.diff(key)) + 1
    rms = []
    for blk in np.split(P, cuts):
        if len(blk) >= min_pts:
            n, d = fit_plane_svd(blk)
            rms.append(float(np.sqrt(np.mean((blk @ n + d) ** 2))))
    if not rms:
        return {"median": None, "n_patches": 0}
    r = np.array(rms)
    return {"median": float(np.median(r)), "mean": float(r.mean()), "p25": float(np.percentile(r, 25)),
            "p75": float(np.percentile(r, 75)), "n_patches": int(len(r)), "values": r.round(5).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--sparse", default=None, help="scene/sparse/0 (cameras for the up direction)")
    ap.add_argument("--viewdirs", default=None, help="npz with 'centres','dirs' (synthetic tests)")
    ap.add_argument("--trackb", required=True)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--new-frame", action="store_true")
    ap.add_argument("--no-gps-scale", action="store_true",
                    help="ignore EXIF GPS and fall back to the girder-band kappa (for comparison)")
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    mc, vc = cfg["measure"], cfg["vtti"]
    out = os.path.join(a.trackb, "measure")
    os.makedirs(out, exist_ok=True)
    prov = load_json(os.path.join(a.trackb, "scale0.json"))
    k0 = prov["kappa0"]
    xyz = np.load(a.points)["xyz"].astype(np.float64)
    if a.viewdirs:
        z = np.load(a.viewdirs)
        centres, vdirs, names = z["centres"], z["dirs"], [str(i) for i in range(len(z["dirs"]))]
    else:
        _, ims, _ = cio.read_model(a.sparse)
        poses = cio.camera_poses(ims)
        names = list(poses)
        centres = np.array([poses[n][0] for n in names])
        vdirs = np.array([poses[n][1] for n in names])
    ov = load_yaml(a.overrides or pkg_path(mc["overrides"])) if (a.overrides or mc.get("overrides")) else {}

    fpath = os.path.join(out, "frame.json")
    total_m = sum(vc["spans_ft"]) * FT_TO_M
    spans_ref = [v * FT_TO_M for v in vc["spans_ft"]]
    # GPS scale first: it is the only estimate that does not depend on the bridge itself, so it
    # can anchor kappa and turn the pier separation into a prediction instead of a free parameter.
    gps = None if a.no_gps_scale else gps_scale(names, centres, os.path.join(a.trackb, "gps.csv"))
    k_gps = gps["kappa"] if gps else None
    sep_target = spans_ref[vc["main_span_index"]] if k_gps else None
    if os.path.exists(fpath) and not a.new_frame:
        F = load_json(fpath)
        fr = Frame.from_dict(F["frame"])
        hw, ground_h, boxes_auto, k0 = F["half_width_sfm"], F["ground_h_sfm"], F["pier_boxes_auto"], F["kappa_frame"]
        log(f"reusing frame, pier boxes and provisional scale from {fpath} (made from {F['made_from']})")
    else:
        k_use = float(k_gps) if k_gps else k0
        fdiag = {}
        if mc.get("frame_source", "trajectory") == "trajectory":
            try:
                fr, fdiag = frame_from_trajectory(xyz, centres, vdirs, mc, k_use)
                k0 = k_use
                hw = fdiag["deck_halfwidth_m"] / max(k0, 1e-9)          # back to SfM units
                ground_h = float(np.percentile(fr.local(xyz)[2], mc.get("ground_pct_fallback", 2.0)))
            except (ValueError, np.linalg.LinAlgError) as e:
                log(f"  trajectory frame failed ({e}) -- falling back to the RANSAC deck plane")
                fr, hw, ground_h, k0 = make_frame(xyz, vdirs, k0, mc, total_m, gps_k=k_gps)
        else:
            fr, hw, ground_h, k0 = make_frame(xyz, vdirs, k0, mc, total_m, gps_k=k_gps)
        s, t, h = fr.local(xyz)
        boxes_auto, _ = auto_boxes(s, t, h, k0, hw, ground_h, mc, sep_target_m=sep_target)
        save_json({"frame": fr.to_dict(), "half_width_sfm": hw, "ground_h_sfm": ground_h,
                   "pier_boxes_auto": boxes_auto, "kappa0_prepare": prov["kappa0"], "kappa_frame": k0,
                   "kappa_source": "gps" if k_gps else "girder_band", "gps_scale": gps,
                   "frame_diag": fdiag, "made_from": a.method}, fpath)
        import yaml
        with open(os.path.join(out, "pier_boxes_auto.yaml"), "w") as f:
            yaml.safe_dump({"pier_boxes": boxes_auto}, f, sort_keys=False)
        log(f"frame + {len(boxes_auto)} automatic pier boxes written ({fpath})")
    s, t, h = fr.local(xyz)
    sm, tm, hm = s * k0, t * k0, h * k0
    boxes = ov.get("pier_boxes") or boxes_auto
    box_src = "override" if ov.get("pier_boxes") else "auto"
    lo_p, hi_p = mc["station_pct"]
    piers = []
    for b in boxes:
        m = in_box(sm, tm, hm, b)
        if m.sum() >= 20:
            a5, a95 = np.percentile(s[m], [lo_p, hi_p])
            piers.append({"station_sfm": float(0.5 * (a5 + a95)), "n": int(m.sum()), "box": b})
    piers.sort(key=lambda p: p["station_sfm"])
    gb = (hm >= mc["girder_band_m"][0]) & (hm <= mc["girder_band_m"][1]) & (np.abs(tm) <= min(hw * k0, mc.get("deck_halfwidth_max_m", 12.0)))
    # Deck ends: the plan takes them at the abutment bearing lines, else at the deck end joints.
    # The deck surface itself is the closest measurable proxy; the girder band is kept as fallback.
    ds = (np.abs(hm) <= float(mc.get("deck_surface_band_m", 0.30))) & (np.abs(tm) <= min(hw * k0, mc.get("deck_halfwidth_max_m", 12.0)))
    if ov.get("end_stations"):
        ends = [v / k0 for v in ov["end_stations"]]
        end_src = ov.get("end_station_source") or "override"
    elif mc.get("end_source", "deck_surface") == "deck_surface" and ds.sum() > 200:
        ends = list(edge_stations(s[ds], mc.get("end_edge_bin_m", 0.25) / k0, mc.get("end_edge_frac", 0.2)))
        end_src = "deck surface"
    elif gb.sum() > 50:
        ends = list(edge_stations(s[gb], mc.get("end_edge_bin_m", 0.25) / k0, mc.get("end_edge_frac", 0.2)))
        end_src = "girder band"
    else:
        ends = [float(np.percentile(s, 1)), float(np.percentile(s, 99))]
        end_src = "cloud extent"
    if end_src in ("deck surface", "girder band"):
        log(f"  ends from the {end_src}: {((ends[1] - ends[0]) * k0):.1f} m apart "
            f"(bridge is {total_m:.1f} m)")
    res = {"method": a.method, "label": a.label or a.method, "n_points": int(len(xyz)), "kappa0": k0,
           "boxes": box_src, "end_stations_from": end_src, "n_piers_found": len(piers)}
    spans_ref = [v * FT_TO_M for v in vc["spans_ft"]]
    if len(piers) >= 2:
        st = np.array([ends[0]] + [p["station_sfm"] for p in piers[:2]] + [ends[1]])
        mi = vc["main_span_index"]
        kappa_pier = spans_ref[mi] / abs(st[mi + 1] - st[mi])              # Eq. (17), circular by design
        # With GPS the scale is independent, so the main span is a genuine check as well and all
        # three errors are reported. Without GPS, Eq. (17) defines kappa and its error is zero by
        # construction -- that is a definition, not a result.
        kappa = float(k_gps) if k_gps else float(kappa_pier)
        spans = [float(kappa * abs(st[k + 1] - st[k])) for k in range(3)]
        if k_gps:
            errs = [abs(spans[k] - spans_ref[k]) / spans_ref[k] for k in range(3)]
        else:
            errs = [None if k == mi else abs(spans[k] - spans_ref[k]) / spans_ref[k] for k in range(3)]  # Eq. (18)
        res.update({"kappa": float(kappa), "kappa_source": "gps" if k_gps else "main_span",
                    "kappa_pier": float(kappa_pier),
                    "kappa_gps": (gps or {}).get("kappa"), "gps_rms_m": (gps or {}).get("rms_m"),
                    "sep_target_m": sep_target,
                    "stations_sfm": st.tolist(), "stations_m": (st * kappa).tolist(),
                    "spans_m": spans, "spans_ref_m": spans_ref, "rel_err": errs,
                    "rel_err_side_mean": float(np.mean([e for k, e in enumerate(errs)
                                                        if e is not None and k != mi]))
                    if any(e is not None and k != mi for k, e in enumerate(errs)) else None})
        S, T, H = s * kappa, t * kappa, h * kappa
        res["deck_rms_m"] = deck_rms(S, T, H, st[0] * kappa, st[-1] * kappa, hw * kappa,
                                     mc["deck_patch_m"], mc["patch_min_points"],
                                     tuple(mc.get("deck_band_m", (-0.15, 0.15))))       # Eq. (19)
        gps_csv = os.path.join(a.trackb, "gps.csv")
        if os.path.exists(gps_csv) and not a.viewdirs:
            with open(gps_csv) as f:
                rows = {r["name"]: r for r in csv.DictReader(f)}
            common_n = [n for n in names if n in rows]
            if len(common_n) >= 3:
                # cross-check only: the 3-D fit is dominated by EXIF altitude noise and must not
                # replace the horizontal scale that anchors kappa
                E = enu([float(rows[n]["lat"]) for n in common_n], [float(rows[n]["lon"]) for n in common_n],
                        [float(rows[n]["alt"]) for n in common_n])
                C = np.array([centres[names.index(n)] for n in common_n])
                k_3d, _, _ = umeyama(C, E)
                sp_gps = [float(k_3d * abs(st[k + 1] - st[k])) for k in range(3)]
                res["gps"] = {"kappa_3d": float(k_3d), "kappa_horizontal": (gps or {}).get("kappa"),
                              "n_cameras": len(common_n), "rms_horizontal_m": (gps or {}).get("rms_m"),
                              "spans_m_3d": sp_gps,
                              "rel_err_3d": [abs(sp_gps[k] - spans_ref[k]) / spans_ref[k] for k in range(3)]}
        keep = np.random.default_rng(0).permutation(len(S))[:2000000]
        np.savez_compressed(os.path.join(out, f"aligned_{a.method}.npz"),
                            stha=np.stack([S[keep], T[keep], H[keep]], 1).astype(np.float32), idx=keep,
                            kappa=kappa, half_width_m=hw * kappa, ground_h_m=ground_h * kappa,
                            stations_m=st * kappa)
        log(f"{a.method}: kappa {kappa:.4f} m/unit  spans {np.round(spans, 3).tolist()} m  "
            f"(ref {np.round(spans_ref, 3).tolist()})  side errors {[None if e is None else round(100 * e, 2) for e in errs]} %  "
            f"deck RMS median {res['deck_rms_m'].get('median')}")
    else:
        log(f"WARNING {a.method}: {len(piers)} pier stations found (need 2) -- check diag_{a.method}.png and set "
            "pier_boxes in configs/vtti_overrides.yaml")
    res["piers"] = piers
    save_json(res, os.path.join(out, f"measure_{a.method}.json"))
    try:
        _diag(sm, tm, hm, boxes, piers, ends, k0, os.path.join(out, f"diag_{a.method}.png"), a.label or a.method)
    except Exception as e:  # noqa: BLE001
        log(f"diagnostic plot failed: {e}")


def _diag(sm, tm, hm, boxes, piers, ends, k0, path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    k = rng.permutation(len(sm))[:400000]
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    ax[0].hexbin(sm[k], hm[k], gridsize=(220, 60), bins="log", cmap="Greys", mincnt=1)
    ax[1].hexbin(sm[k], tm[k], gridsize=(220, 40), bins="log", cmap="Greys", mincnt=1)
    for b in boxes:
        for axi, key in ((ax[0], "h"), (ax[1], "t")):
            axi.add_patch(plt.Rectangle((b["s"][0], b[key][0]), b["s"][1] - b["s"][0], b[key][1] - b[key][0],
                                        fill=False, ec="#1f5ea0", lw=1.2))
    for p in piers:
        for axi in ax:
            axi.axvline(p["station_sfm"] * k0, color="#1f5ea0", lw=0.8, ls="--")
    for e in ends:
        for axi in ax:
            axi.axvline(e * k0, color="#b5651d", lw=0.8, ls=":")
    ax[0].set_ylabel("h above deck plane (m, provisional)")
    ax[1].set_ylabel("t across axis (m)")
    ax[1].set_xlabel("s along axis (m, provisional scale)")
    ax[0].set_title(f"{label}: elevation and plan; boxes = pier crops, dashed = pier stations, dotted = ends", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
