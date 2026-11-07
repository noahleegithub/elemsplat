"""
synth_bridges.py -- synthetic bridges for the smoke test (NOT data for the paper).

    girder_bridge()        multi-girder bridge with cross-frames, piers (bents), deck
    cable_stayed_bridge()  box girder + deck, H towers, two planes of fan stays
    suspension_bridge()    main cables, suspenders, towers, box girder + deck
    vtti_like_scene()      three-span steel-girder bridge (90 + 150 + 90 ft) with ground, approach
                           roads and UAV-like cameras, under a random similarity transform
    write_brpcd_tree()     writes clouds in a BrPCD-like folder tree (type/bridge/class/member.txt)

Every generator returns xyz (N,3), cls (N,) class ids in the BrPCD order
[girder, deck, pier, main_cable, suspender, stay] and inst (N,) member ids.
"""
import os

import numpy as np

GIRDER, DECK, PIER, MAIN_CABLE, SUSPENDER, STAY = range(6)
CLASS_NAMES = ["girder", "deck", "pier", "main_cable", "suspender", "stay"]


class _Acc:
    def __init__(self, rng, density, noise):
        self.rng, self.density, self.noise = rng, density, noise
        self.xyz, self.cls, self.inst = [], [], []
        self.next_id = 0

    def new_id(self):
        self.next_id += 1
        return self.next_id - 1

    def add(self, pts, c, i):
        if len(pts):
            pts = pts + self.rng.normal(0, self.noise, pts.shape)
            self.xyz.append(pts)
            self.cls.append(np.full(len(pts), c, np.int64))
            self.inst.append(np.full(len(pts), i, np.int64))

    def rect(self, origin, e1, e2, c, i):
        """Points on the parallelogram origin + a e1 + b e2, a, b in [0, 1]."""
        e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
        area = np.linalg.norm(np.cross(e1, e2))
        n = self.rng.poisson(max(area * self.density, 1))
        ab = self.rng.random((n, 2))
        self.add(np.asarray(origin, float) + ab[:, :1] * e1 + ab[:, 1:] * e2, c, i)

    def box(self, lo, hi, c, i):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        d = hi - lo
        ex, ey, ez = np.array([d[0], 0, 0]), np.array([0, d[1], 0]), np.array([0, 0, d[2]])
        for o, a, b in ((lo, ex, ey), (lo + ez, ex, ey), (lo, ex, ez), (lo + ey, ex, ez), (lo, ey, ez), (lo + ex, ey, ez)):
            self.rect(o, a, b, c, i)

    def cylinder(self, p0, p1, r, c, i, dens_scale=1.0):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        ax = p1 - p0
        L = np.linalg.norm(ax)
        ax = ax / L
        tmp = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
        b1 = np.cross(ax, tmp)
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(ax, b1)
        n = self.rng.poisson(max(2 * np.pi * max(r, 0.05) * L * self.density * dens_scale, 2))
        a = self.rng.random(n) * L
        th = self.rng.random(n) * 2 * np.pi
        pts = p0 + a[:, None] * ax + r * (np.cos(th)[:, None] * b1 + np.sin(th)[:, None] * b2)
        self.add(pts, c, i)

    def done(self, dropout=0.0):
        xyz, cls, inst = np.concatenate(self.xyz), np.concatenate(self.cls), np.concatenate(self.inst)
        if dropout > 0:                                         # occlusion: drop random 2 m cells
            cell = np.floor(xyz / 2.0).astype(np.int64)
            key = cell[:, 0] * 73856093 ^ cell[:, 1] * 19349663 ^ cell[:, 2] * 83492791
            u, inv = np.unique(key, return_inverse=True)
            keep_cell = self.rng.random(len(u)) > dropout
            keep = keep_cell[np.asarray(inv).ravel()]
            xyz, cls, inst = xyz[keep], cls[keep], inst[keep]
        _, inst = np.unique(inst, return_inverse=True)       # compact ids
        return xyz, cls, np.asarray(inst).ravel().astype(np.int64)


# ----------------------------------------------------------------------------- girder bridge
def girder_bridge(spans=(30.0, 30.0), n_girders=5, spacing=2.4, continuous=False, columns=3,
                  girder_depth=1.6, deck_t=0.25, clearance=7.0, crossframe_every=7.5,
                  density=25.0, noise=0.005, dropout=0.05, seed=0, pier_unit="bent"):
    rng = np.random.default_rng(seed)
    A = _Acc(rng, density, noise)
    st = np.concatenate([[0.0], np.cumsum(spans)])
    W = (n_girders - 1) * spacing + 2.0
    yg = (np.arange(n_girders) - (n_girders - 1) / 2) * spacing
    # deck (top + bottom + edges), whole or per span
    deck_ids = [A.new_id()] * len(spans) if continuous else [A.new_id() for _ in spans]
    for k in range(len(spans)):
        A.box((st[k], -W / 2, -deck_t), (st[k + 1], W / 2, 0.0), DECK, deck_ids[k])
    # girders: web + flanges, whole or one member per span
    gids = {}
    for g in range(n_girders):
        whole = A.new_id()
        for k in range(len(spans)):
            gids[(g, k)] = whole if continuous else A.new_id()
    z_top, z_bot = -deck_t, -deck_t - girder_depth
    for g, y in enumerate(yg):
        for k in range(len(spans)):
            gid = gids[(g, k)]
            x0, x1 = st[k] + 0.3, st[k + 1] - 0.3
            A.rect((x0, y, z_bot), (x1 - x0, 0, 0), (0, 0, girder_depth), GIRDER, gid)
            A.rect((x0, y - 0.2, z_bot), (x1 - x0, 0, 0), (0, 0.4, 0), GIRDER, gid)
            A.rect((x0, y - 0.15, z_top), (x1 - x0, 0, 0), (0, 0.3, 0), GIRDER, gid)
    # cross-frames between neighbouring girders; each half belongs to the nearer girder, so the
    # girders of one span touch each other (this is what makes plain clustering merge them)
    for k in range(len(spans)):
        x0, x1 = st[k] + 0.3, st[k + 1] - 0.3
        for g in range(n_girders - 1):
            ym = (yg[g] + yg[g + 1]) / 2
            for xc in np.arange(x0 + crossframe_every / 2, x1, crossframe_every):
                for ya, yb, owner in ((yg[g], ym, gids[(g, k)]), (ym, yg[g + 1], gids[(g + 1, k)])):
                    n = rng.poisson(max(density * abs(yb - ya) * 1.2, 2))
                    a = rng.random(n)
                    za = z_bot + 0.2 + a * (girder_depth - 0.4)
                    A.add(np.stack([np.full(n, xc), ya + a * (yb - ya), za], 1), GIRDER, owner)
    # piers: bents at interior supports (columns + cap)
    for k in range(1, len(spans)):
        bent = A.new_id()
        x = st[k]
        cap_lo, cap_hi = -deck_t - girder_depth - 1.2, -deck_t - girder_depth
        A.box((x - 0.75, -W / 2 + 0.8, cap_lo), (x + 0.75, W / 2 - 0.8, cap_hi), PIER, bent)
        ys = np.linspace(-W / 2 + 1.6, W / 2 - 1.6, columns)
        for yc in ys:
            cid = bent if pier_unit == "bent" else A.new_id()
            A.cylinder((x, yc, -clearance - girder_depth), (x, yc, cap_lo), 0.5, PIER, cid)
    return A.done(dropout)


# ----------------------------------------------------------------------------- cable-stayed
def cable_stayed_bridge(main=120.0, side=50.0, stays_per_side=10, tower_h=40.0, deck_w=14.0,
                        density=20.0, noise=0.005, dropout=0.03, seed=0):
    rng = np.random.default_rng(seed)
    A = _Acc(rng, density, noise)
    L = side * 2 + main
    deck_id, gird_id = A.new_id(), A.new_id()
    A.box((0, -deck_w / 2, -0.3), (L, deck_w / 2, 0.0), DECK, deck_id)
    A.box((0, -deck_w / 2 + 1.0, -2.8), (L, deck_w / 2 - 1.0, -0.3), GIRDER, gird_id)
    towers = (side, side + main)
    for xt in towers:
        tid = A.new_id()
        for yl in (-deck_w / 2 - 0.8, deck_w / 2 + 0.8):
            A.cylinder((xt, yl, -20.0), (xt, yl, tower_h), 1.0, PIER, tid)
        A.box((xt - 0.8, -deck_w / 2 - 0.8, tower_h - 2), (xt + 0.8, deck_w / 2 + 0.8, tower_h), PIER, tid)
        for yp in (-deck_w / 2 + 0.5, deck_w / 2 - 0.5):
            for sgn in (-1, 1):
                reach = side - 5 if sgn < 0 else main / 2 - 5
                for j in range(stays_per_side):
                    xa = xt + sgn * (8 + j * (reach - 8) / max(1, stays_per_side - 1))
                    za = tower_h - 3 - j * 1.2
                    sid = A.new_id()
                    A.cylinder((xa, yp, 0.0), (xt, yp * 1.08, za), 0.08, STAY, sid, dens_scale=6.0)
    return A.done(dropout)


# ----------------------------------------------------------------------------- suspension
def suspension_bridge(main=200.0, side=60.0, hanger_every=10.0, tower_h=45.0, deck_w=16.0,
                      density=15.0, noise=0.005, dropout=0.03, seed=0):
    rng = np.random.default_rng(seed)
    A = _Acc(rng, density, noise)
    L = side * 2 + main
    A.box((0, -deck_w / 2, -0.3), (L, deck_w / 2, 0.0), DECK, A.new_id())
    A.box((0, -deck_w / 2 + 1.0, -3.0), (L, deck_w / 2 - 1.0, -0.3), GIRDER, A.new_id())
    towers = (side, side + main)
    for xt in towers:
        tid = A.new_id()
        for yl in (-deck_w / 2 - 1.0, deck_w / 2 + 1.0):
            A.cylinder((xt, yl, -20.0), (xt, yl, tower_h), 1.2, PIER, tid)
    sag = tower_h - 5.0
    for yp in (-deck_w / 2 - 1.0, deck_w / 2 + 1.0):
        cid = A.new_id()
        xs = np.linspace(0, L, 400)
        zc = np.where((xs > towers[0]) & (xs < towers[1]),
                      tower_h - sag * (1 - ((xs - (towers[0] + main / 2)) / (main / 2)) ** 2),
                      tower_h * np.where(xs <= towers[0], xs / side, (L - xs) / side))
        for a in range(len(xs) - 1):
            A.cylinder((xs[a], yp, zc[a]), (xs[a + 1], yp, zc[a + 1]), 0.3, MAIN_CABLE, cid)
        for xh in np.arange(towers[0] + hanger_every, towers[1] - 1, hanger_every):
            zt = tower_h - sag * (1 - ((xh - (towers[0] + main / 2)) / (main / 2)) ** 2)
            A.cylinder((xh, yp, 0.0), (xh, yp, zt), 0.05, SUSPENDER, A.new_id(), dens_scale=8.0)
    return A.done(dropout)


# ----------------------------------------------------------------------------- VTTI-like
def vtti_like_scene(seed=0, scale=0.37, density=60.0, noise=0.01):
    """
    Three spans 27.432 + 45.72 + 27.432 m (90 + 150 + 90 ft), 5 steel girders, 2 bents, ground
    8 m below the girders, approach roads at deck level, cameras of a nadir grid and a side pass.
    Returns (xyz, cams_centre, cams_viewdir, truth) in an arbitrary similarity frame.
    """
    rng = np.random.default_rng(seed)
    spans = (27.432, 45.72, 27.432)
    xyz, cls, inst = girder_bridge(spans, n_girders=5, spacing=2.4, continuous=True, columns=3,
                                   girder_depth=1.8, clearance=7.0, density=density, noise=noise,
                                   dropout=0.02, seed=seed)
    A = _Acc(rng, density / 6, noise * 2)
    L = sum(spans)
    A.rect((-60, -40, -0.25 - 1.8 - 7.0), (L + 120, 0, 0), (0, 80, 0), 9, 0)      # ground
    for x0, x1 in ((-60, 0), (L, L + 60)):                                      # approach roads
        A.rect((x0, -6, 0.0), (x1 - x0, 0, 0), (0, 12, 0), 9, 1)
    gx, gc, gi = A.done()
    pts = np.concatenate([xyz, gx])
    # cameras: nadir grid 30 m above the deck + side pass at girder height, looking at the bridge
    cx, cy = np.meshgrid(np.linspace(-5, L + 5, 12), np.linspace(-15, 15, 4))
    nadir_c = np.stack([cx.ravel(), cy.ravel(), np.full(cx.size, 30.0)], 1)
    nadir_v = np.tile([0.0, 0.0, -1.0], (len(nadir_c), 1))
    sx = np.linspace(0, L, 25)
    side_c = np.concatenate([np.stack([sx, np.full(25, y), np.full(25, -2.0)], 1) for y in (-18, 18)])
    side_v = np.concatenate([np.tile([0.0, -np.sign(y), 0.0], (25, 1)) for y in (-18, 18)])
    cams_c, cams_v = np.concatenate([nadir_c, side_c]), np.concatenate([nadir_v, side_v])
    # random similarity transform (what SfM gives)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    R = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                  [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                  [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])
    tvec = rng.normal(size=3) * 5
    tf = lambda p: scale * (p @ R.T) + tvec
    truth = {"spans_m": list(spans), "scale": scale, "n_girders": 5, "n_piers": 2,
             "pier_stations_m": [spans[0], spans[0] + spans[1]]}
    return tf(pts), tf(cams_c), cams_v @ R.T, truth, np.concatenate([cls, gc]), np.concatenate([inst, gi + 10000])


# ----------------------------------------------------------------------------- BrPCD-like tree
def write_brpcd_tree(root, n_real=4, n_variants=2, seed=0):
    """Small BrPCD-like tree: <type>/<bridge>/<class>/<class>_<k>.txt (x y z per line)."""
    rng = np.random.default_rng(seed)
    made = []
    specs = []
    for r in range(n_real):
        kind = ["girder", "girder", "cable-stayed", "suspension"][r % 4]
        specs.append((kind, r, None))
        for v in range(n_variants):
            specs.append((kind, r, v))
    for kind, r, v in specs:
        s = seed + 100 * r + (0 if v is None else v + 1)
        if kind == "girder":
            nsp = int(rng.integers(1, 4)) + (0 if v is None else v + 1)
            xyz, cls, inst = girder_bridge(spans=tuple([25.0 + 5 * (r % 2)] * nsp), n_girders=4 + r % 3 + (v or 0),
                                           continuous=bool(r % 2), seed=s, density=12.0)
            prefix = "c" if r % 2 else "s"
        elif kind == "cable-stayed":
            xyz, cls, inst = cable_stayed_bridge(stays_per_side=6 + 2 * (v or 0), seed=s, density=8.0)
            prefix = "cs"
        else:
            xyz, cls, inst = suspension_bridge(hanger_every=20.0 - 4 * (v or 0), seed=s, density=6.0)
            prefix = "ss"
        name = f"{prefix}-bridge{r + 1}" if v is None else f"{prefix}-f-bridge{r + 1}-{['com', 'pe', 'le', 's1'][v % 4]}"
        bdir = os.path.join(root, kind, name)
        for i in np.unique(inst):
            m = inst == i
            c = CLASS_NAMES[int(cls[m][0])]
            os.makedirs(os.path.join(bdir, c), exist_ok=True)
            np.savetxt(os.path.join(bdir, c, f"{c}_{int(i):03d}.txt"), xyz[m], fmt="%.4f")
        made.append(name)
    return made


# ----------------------------------------------------------------------------- fake VGGT output
def _look_rotation(view_dir):
    """Camera-from-world rotation (OpenCV: z forward, y down) whose optical axis is view_dir."""
    z = np.asarray(view_dir, float)
    z = z / np.linalg.norm(z)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])          # rows = camera axes in world coordinates


def rotmat_to_qvec(R):
    """Rotation matrix to COLMAP quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def write_colmap_txt_model(sparse_dir, names, centres, view_dirs, width=1600, height=1200, focal=1400.0):
    """Minimal COLMAP text model (one PINHOLE camera, no observations) for the synthetic cameras."""
    os.makedirs(sparse_dir, exist_ok=True)
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list\n1 PINHOLE %d %d %.1f %.1f %.1f %.1f\n" % (width, height, focal, focal, width / 2, height / 2))
    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        f.write("# Image list\n")
        for k, (n, c, v) in enumerate(zip(names, centres, view_dirs)):
            R = _look_rotation(v)
            t = -R @ np.asarray(c, float)
            q = rotmat_to_qvec(R)
            f.write("%d %.9f %.9f %.9f %.9f %.9f %.9f %.9f 1 %s\n\n" % (k + 1, *q, *t, n))
    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list (empty)\n")


def fake_vggt_predictions(xyz, names, centres, view_dirs, seed=0, hw=(392, 518), focal=420.0):
    """
    Predictions in the format vggt_points.py saves, from a point cloud in the scene frame: the
    cameras are expressed in a random similarity frame (VGGT knows nothing about the scene frame)
    and the depth maps are z-buffer renderings of the points. Confidence is 1 + noise where a
    point was seen and 0 elsewhere. Returns a dict to pass to np.savez_compressed.
    """
    rng = np.random.default_rng(seed)
    H, W = hw
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    Rv = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                   [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                   [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])
    sv, tv = float(rng.uniform(0.2, 3.0)), rng.normal(size=3)
    to_v = lambda p: sv * (p @ Rv.T) + tv                       # scene frame -> fake VGGT frame
    pts_v = to_v(np.asarray(xyz, float))
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], np.float32)
    ext = np.zeros((len(names), 3, 4), np.float32)
    depth = np.zeros((len(names), H, W), np.float32)
    conf = np.zeros((len(names), H, W), np.float32)
    for k, (cc, v) in enumerate(zip(centres, view_dirs)):
        R = _look_rotation(v) @ Rv.T                            # camera axes rotate with the frame
        t = -R @ to_v(np.asarray(cc, float))
        ext[k, :, :3], ext[k, :, 3] = R, t
        cam = pts_v @ R.T + t
        z = cam[:, 2]
        ok = z > 1e-3
        u = np.round(K[0, 0] * cam[ok, 0] / z[ok] + K[0, 2]).astype(int)
        w = np.round(K[1, 1] * cam[ok, 1] / z[ok] + K[1, 2]).astype(int)
        inside = (u >= 0) & (u < W) & (w >= 0) & (w < H)
        buf = np.full(H * W, np.inf, np.float32)
        np.minimum.at(buf, (w[inside] * W + u[inside]), z[ok][inside].astype(np.float32))
        seen = np.isfinite(buf)
        depth[k] = np.where(seen, buf, 0.0).reshape(H, W)
        conf[k] = (seen.reshape(H, W) * (1.0 + rng.uniform(0, 1, (H, W)))).astype(np.float32)
    return {"names": np.array(names), "image_hw": np.array(hw), "extrinsic": ext,
            "intrinsic": np.tile(K, (len(names), 1, 1)), "depth": depth, "conf": conf.astype(np.float16),
            "truth": {"scale": sv}}
