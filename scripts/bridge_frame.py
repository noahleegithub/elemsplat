"""
bridge_frame.py -- bridge frame for an SfM reconstruction (VTTI, Track B and RQ3).

SfM leaves the scene in an arbitrary similarity frame. The vertical is first taken from the
cameras (minus the mean viewing direction: the UAV nadir grid looks down), then refined by
the deck plane = the HIGHEST large plane whose normal is within deck_normal_max_deg of that
vertical (the ground and the rail corridor lie lower). The axis is the principal direction of
the deck inliers in the plane (Eqs. 1 and 2 of the paper).
"""
import numpy as np

from geometry import Frame, principal_axis_in_plane, ransac_plane


def up_from_cameras(view_dirs):
    v = -np.asarray(view_dirs, float).mean(0)
    return v / np.linalg.norm(v)


def horizontal_planes(xyz, up, thresh, max_deg, n_planes=4, min_frac=0.01, iters=800, seed=0):
    """Up to n_planes near-horizontal planes found one after another (inliers removed)."""
    remaining = np.ones(len(xyz), bool)
    planes = []
    for k in range(n_planes):
        idx = np.flatnonzero(remaining)
        if len(idx) < 500:
            break
        try:
            n, d, inl = ransac_plane(xyz[idx], thresh, iters, seed + k, ref_normal=up, max_angle_deg=max_deg)
        except ValueError:
            break
        cnt = int(inl.sum())
        if cnt < min_frac * len(xyz):
            break
        sel = idx[inl]
        planes.append({"n": n, "d": float(d), "count": cnt, "elev": float(np.median(xyz[sel] @ up)), "idx": sel})
        remaining[sel] = False
    return planes


def deck_and_ground(planes, min_rel=0.3):
    big = [p for p in planes if p["count"] >= min_rel * max(q["count"] for q in planes)] or planes
    deck = max(big, key=lambda p: p["elev"])
    lower = [p for p in planes if p["elev"] < deck["elev"] and p is not deck]
    ground = min(lower, key=lambda p: p["elev"]) if lower else None
    return deck, ground


def frame_from_deck_plane(xyz, deck):
    n, d = np.asarray(deck["n"], float), float(deck["d"])
    P = xyz[deck["idx"]]
    u = principal_axis_in_plane(P, n)
    w = np.cross(n, u)
    origin = P.mean(0)
    origin = origin - (origin @ n + d) * n
    return Frame(origin, u, w, n, d)


def relative_scale(xyz):
    """Robust scene size (95th percentile distance to the median point)."""
    c = np.median(xyz, 0)
    return float(np.percentile(np.linalg.norm(xyz - c, axis=1), 95))


def provisional(xyz, view_dirs, total_length_m, cfg_measure, seed=0):
    """
    Frame + provisional scale kappa0 = known total length / deck extent along the axis, with a
    plane threshold relative to the scene size (no metric threshold exists before kappa0).
    Used only to convert metric thresholds and TSDF voxel sizes into SfM units.
    """
    up = up_from_cameras(view_dirs)
    rs = relative_scale(xyz)
    planes = horizontal_planes(xyz, up, 0.002 * rs, cfg_measure["deck_normal_max_deg"], seed=seed)
    if not planes:
        raise ValueError("no near-horizontal plane found -- check the camera up direction")
    deck, ground = deck_and_ground(planes)
    fr = frame_from_deck_plane(xyz, deck)
    s, _, _ = fr.local(xyz[deck["idx"]])
    lo, hi = np.percentile(s, [1, 99])
    kappa0 = total_length_m / max(hi - lo, 1e-9)
    return {"frame": fr.to_dict(), "up_cameras": up.tolist(), "kappa0": float(kappa0),
            "deck_extent_sfm": float(hi - lo), "n_planes": len(planes),
            "ground_elev_sfm": None if ground is None else ground["elev"], "deck_elev_sfm": deck["elev"]}
