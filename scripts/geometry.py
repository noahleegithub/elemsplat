"""
geometry.py -- point-cloud geometry used by M1, M2 and the VTTI measurement.

    voxel_downsample_idx   one random point per occupied voxel
    median_spacing         median nearest-neighbour distance (for the automatic voxel size)
    estimate_normals       PCA normals from k nearest neighbours
    ransac_plane           Eq. (1): plane with the largest consensus set, optional normal constraint
    Frame                  bridge frame (u, w, n, d) and Eq. (2): station s, offset t, height h
    kde_peaks              Eq. (5): 1-D kernel density estimate and its prominent peaks
    sequential_lines       Eq. (6): sequential RANSAC lines (stay and suspender cables)
All functions are NumPy/SciPy only and accept float64 (N,3) arrays.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------- sampling
def voxel_downsample_idx(xyz, voxel, seed=0):
    """Indices (sorted) of one random point per occupied voxel of edge `voxel`."""
    n = len(xyz)
    if voxel is None or voxel <= 0 or n == 0:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = np.floor((xyz[perm] - xyz.min(0)) / voxel).astype(np.int64)
    dims = k.max(0) + 1
    if float(dims[0]) * float(dims[1]) * float(dims[2]) < 2 ** 62:
        key = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
        _, first = np.unique(key, return_index=True)
    else:
        _, first = np.unique(k, axis=0, return_index=True)
    return np.sort(perm[first])


def median_spacing(xyz, max_points=200000, seed=0):
    """Median distance to the nearest other point (on a random subset)."""
    if len(xyz) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    q = xyz if len(xyz) <= max_points else xyz[rng.choice(len(xyz), max_points, replace=False)]
    d, _ = cKDTree(xyz).query(q, k=2)
    return float(np.median(d[:, 1]))


def estimate_normals(xyz, k=16, chunk=100000):
    """Unit normals (N,3) float32 from the smallest PCA axis of k neighbours, oriented to +z."""
    n = len(xyz)
    out = np.zeros((n, 3), np.float32)
    if n < 3:
        out[:, 2] = 1.0
        return out
    k = min(k, n)
    tree = cKDTree(xyz)
    for s in range(0, n, chunk):
        _, idx = tree.query(xyz[s:s + chunk], k=k, workers=-1)
        nb = xyz[idx]
        nb = nb - nb.mean(1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb, nb)
        _, vec = np.linalg.eigh(cov)
        nrm = vec[:, :, 0]
        nrm[nrm[:, 2] < 0] *= -1
        out[s:s + chunk] = nrm
    return out


# ----------------------------------------------------------------------------- planes
def fit_plane_svd(pts):
    """Least-squares plane (unit normal n, offset d) with n.p + d = 0."""
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    return n, -float(n @ c)


def ransac_plane(xyz, thresh, iters=1000, seed=0, ref_normal=None, max_angle_deg=None,
                 sample_max=150000, batch=100):
    """
    Eq. (1). Candidate planes through random triples; the score is the number of points
    within `thresh` (counted on a random subset of at most `sample_max` points). If
    `ref_normal` is given, candidates whose normal is more than `max_angle_deg` away from it
    are discarded. Returns (n, d, inlier_mask) after a least-squares refit on all inliers;
    n is oriented so that n . ref_normal >= 0 (or n_z >= 0).
    """
    rng = np.random.default_rng(seed)
    if len(xyz) < 3:
        raise ValueError("ransac_plane needs at least 3 points")
    pts = xyz if len(xyz) <= sample_max else xyz[rng.choice(len(xyz), sample_max, replace=False)]
    pts32 = pts.astype(np.float32)
    i = rng.integers(0, len(pts), size=(iters, 3))
    p0, p1, p2 = pts[i[:, 0]], pts[i[:, 1]], pts[i[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-12
    nrm, p0 = nrm[ok] / ln[ok, None], p0[ok]
    if ref_normal is not None and max_angle_deg is not None:
        ref = np.asarray(ref_normal, float) / np.linalg.norm(ref_normal)
        keep = np.abs(nrm @ ref) >= np.cos(np.deg2rad(max_angle_deg))
        nrm, p0 = nrm[keep], p0[keep]
    if len(nrm) == 0:
        raise ValueError("no plane candidate satisfies the normal constraint")
    dd = -(nrm * p0).sum(1)
    best, best_count = 0, -1
    for b in range(0, len(nrm), batch):
        r = np.abs(pts32 @ nrm[b:b + batch].T.astype(np.float32) + dd[b:b + batch].astype(np.float32))
        cnt = (r < thresh).sum(0)
        j = int(np.argmax(cnt))
        if cnt[j] > best_count:
            best, best_count = b + j, int(cnt[j])
    n, d = nrm[best], dd[best]
    inl = np.abs(xyz @ n + d) < thresh
    for _ in range(2):                                 # refit on all inliers, twice
        if inl.sum() >= 3:
            n, d = fit_plane_svd(xyz[inl])
            inl = np.abs(xyz @ n + d) < thresh
    ref = np.asarray(ref_normal, float) if ref_normal is not None else np.array([0.0, 0.0, 1.0])
    if n @ ref < 0:
        n, d = -n, -d
    return n, d, inl


# ----------------------------------------------------------------------------- frame
class Frame:
    """Bridge frame: origin, axis u, transverse w = n x u, deck normal n, plane offset d."""

    def __init__(self, origin, u, w, n, d):
        self.origin = np.asarray(origin, float)
        self.u, self.w, self.n = (np.asarray(v, float) for v in (u, w, n))
        self.d = float(d)

    def local(self, xyz):
        """Eq. (2): station s, offset t, height h above the deck plane."""
        q = xyz - self.origin
        return q @ self.u, q @ self.w, xyz @ self.n + self.d

    def rotation(self):
        """3x3 matrix whose rows are u, w, n (maps world vectors into the frame)."""
        return np.stack([self.u, self.w, self.n])

    def to_dict(self):
        return {"origin": self.origin.tolist(), "u": self.u.tolist(), "w": self.w.tolist(),
                "n": self.n.tolist(), "d": self.d}

    @staticmethod
    def from_dict(d):
        return Frame(d["origin"], d["u"], d["w"], d["n"], d["d"])


def principal_axis_in_plane(pts, n):
    """Leading eigenvector of the covariance of pts projected onto the plane with normal n."""
    q = pts - pts.mean(0)
    q = q - np.outer(q @ n, n)
    _, _, vt = np.linalg.svd(q[:: max(1, len(q) // 200000)], full_matrices=False)
    u = vt[0] - (vt[0] @ n) * n
    return u / np.linalg.norm(u)


def frame_from_deck(deck_xyz, thresh, iters=1000, seed=0, ref_up=(0.0, 0.0, 1.0), max_angle_deg=30.0,
                    axis_hint=None):
    """Deck plane by RANSAC (Eq. 1), axis = principal direction of its inliers in the plane."""
    n, d, inl = ransac_plane(deck_xyz, thresh, iters, seed, ref_normal=ref_up, max_angle_deg=max_angle_deg)
    P = deck_xyz[inl] if inl.sum() >= 10 else deck_xyz
    u = principal_axis_in_plane(P, n)
    hint = np.asarray(axis_hint if axis_hint is not None else (1.0, 0.0, 0.0), float)
    if u @ hint < 0:
        u = -u
    w = np.cross(n, u)
    origin = P.mean(0)
    origin = origin - (origin @ n + d) * n             # put the origin on the plane
    return Frame(origin, u, w, n, d), inl


# ----------------------------------------------------------------------------- KDE
def kde_1d(x, bw, step=None, lo=None, hi=None, weights=None):
    """Gaussian KDE on a regular grid via a smoothed histogram (fast for millions of points)."""
    x = np.asarray(x, float)
    step = step or bw / 5.0
    lo = x.min() - 4 * bw if lo is None else lo
    hi = x.max() + 4 * bw if hi is None else hi
    nb = max(3, int(np.ceil((hi - lo) / step)))
    hist, edges = np.histogram(x, bins=nb, range=(lo, lo + nb * step), weights=weights)
    dens = gaussian_filter1d(hist.astype(float), sigma=bw / step, mode="constant")
    total = hist.sum()
    if total > 0:
        dens = dens / (total * step)
    return edges[:-1] + step / 2.0, dens


def kde_peaks(x, bw, eta, min_sep=None, step=None, valley_frac=None):
    """
    Eq. (5): peaks of the KDE whose prominence is at least eta * max(density), at least
    `min_sep` apart. Returns (peak positions, grid, density); positions are refined by a
    parabola through the three grid values around each maximum.
    valley_frac: merge two neighbouring peaks when the density between them never drops
    below valley_frac x the lower peak (the webs of one box girder, not two girders).
    """
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.asarray(x, float).copy(), x, np.ones_like(x))
    grid, dens = kde_1d(x, bw, step)
    h = grid[1] - grid[0]
    dist = max(1, int(round((min_sep or bw) / h)))
    pk, _ = find_peaks(dens, prominence=eta * dens.max(), distance=dist)
    if valley_frac is not None and len(pk) > 1:
        groups = [[int(pk[0])]]
        for a, b in zip(pk[:-1], pk[1:]):
            valley = dens[a:b + 1].min()
            if valley > valley_frac * min(dens[a], dens[b]):
                groups[-1].append(int(b))
            else:
                groups.append([int(b)])
        merged = []
        for g in groups:
            if len(g) == 1:
                merged.append(("peak", g[0]))
            else:
                merged.append(("mid", 0.5 * (grid[g[0]] + grid[g[-1]])))
        pos = [grid[v] if kind == "peak" else v for kind, v in merged]
        if any(kind == "mid" for kind, _ in merged):
            return np.array(sorted(pos)), grid, dens
    pos = []
    for p in pk:
        if 0 < p < len(dens) - 1:
            y0, y1, y2 = dens[p - 1], dens[p], dens[p + 1]
            den = y0 - 2 * y1 + y2
            off = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            pos.append(grid[p] + float(np.clip(off, -1, 1)) * h)
        else:
            pos.append(grid[p])
    return np.array(sorted(pos)), grid, dens


# ----------------------------------------------------------------------------- lines
def point_line_dist(pts, o, d):
    """Eq. (6): distance of points to the line o + r d (d unit)."""
    q = pts - o
    return np.sqrt(np.maximum((q * q).sum(1) - (q @ d) ** 2, 0.0))


def _fit_line(pts):
    o = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - o, full_matrices=False)
    return o, vt[0] / np.linalg.norm(vt[0])


def ransac_line(pts, thresh, n_hyp=200, seed=0, score_max=20000, local_radius=None, tree=None):
    """One RANSAC line: hypotheses through a random seed point and a neighbour within
    `local_radius` (if given), scored on a random subset; refit by PCA on the inliers."""
    rng = np.random.default_rng(seed)
    n = len(pts)
    sub = pts if n <= score_max else pts[rng.choice(n, score_max, replace=False)]
    a_idx = rng.integers(0, n, size=n_hyp)
    if local_radius is not None and tree is not None:
        b_idx = np.empty(n_hyp, np.int64)
        for j, ai in enumerate(a_idx):
            nb = tree.query_ball_point(pts[ai], local_radius)
            b_idx[j] = nb[rng.integers(0, len(nb))] if len(nb) > 1 else rng.integers(0, n)
    else:
        b_idx = rng.integers(0, n, size=n_hyp)
    a, b = pts[a_idx], pts[b_idx]
    dv = b - a
    ln = np.linalg.norm(dv, axis=1)
    ok = ln > 1e-9
    if not ok.any():
        return None
    a, dv = a[ok], dv[ok] / ln[ok, None]
    best, best_cnt = 0, -1
    for j in range(len(a)):
        cnt = int((point_line_dist(sub, a[j], dv[j]) < thresh).sum())
        if cnt > best_cnt:
            best, best_cnt = j, cnt
    o, d = a[best], dv[best]
    inl = point_line_dist(pts, o, d) < thresh
    if inl.sum() >= 2:
        o, d = _fit_line(pts[inl])
        inl = point_line_dist(pts, o, d) < thresh
    return o, d, inl


def sequential_lines(pts, thresh, min_inliers, max_lines=600, n_hyp=200, seed=0, local_radius=None):
    """Eq. (6) applied repeatedly: extract a line, remove its inliers, stop below m_min inliers."""
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(pts))
    lines = []
    while len(remaining) >= min_inliers and len(lines) < max_lines:
        P = pts[remaining]
        tree = cKDTree(P) if local_radius is not None else None
        res = ransac_line(P, thresh, n_hyp, int(rng.integers(1 << 30)), local_radius=local_radius, tree=tree)
        if res is None:
            break
        o, d, inl = res
        if inl.sum() < min_inliers:
            break
        lines.append((o, d))
        remaining = remaining[~inl]
    return lines, remaining


def edge_stations(x, bin_size, frac=0.2):
    """
    Both ends of a 1-D point distribution without the inward bias of percentiles: histogram
    with `bin_size`, then the outermost bins whose count reaches `frac` x the median count of
    the occupied bins. Returns (start, end).
    """
    x = np.asarray(x, float)
    if len(x) < 10:
        return float(np.min(x)), float(np.max(x))
    lo, hi = x.min(), x.max()
    nb = max(3, int(np.ceil((hi - lo) / bin_size)))
    cnt, edges = np.histogram(x, bins=nb, range=(lo, lo + nb * bin_size))
    occ = cnt[cnt > 0]
    thr = frac * np.median(occ)
    ok = np.flatnonzero(cnt >= thr)
    return float(edges[ok[0]]), float(edges[ok[-1] + 1])


# ----------------------------------------------------------------------------- similarity transform
def umeyama(src, dst):
    """Least-squares similarity dst ~ s R src + t (Umeyama, 1991). Returns (s, R, t)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((xs ** 2).sum() / len(src))
    return float(s), R, md - s * R @ ms


def robust_umeyama(src, dst, reject=3.0, min_keep=4):
    """umeyama() with one rejection pass: pairs whose residual exceeds `reject` times the median
    are dropped and the fit is repeated. Returns (s, R, t, residuals of every pair, kept mask)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    s, R, t = umeyama(src, dst)
    res = np.linalg.norm((s * src @ R.T + t) - dst, axis=1)
    keep = res <= reject * max(np.median(res), 1e-12)
    if min_keep <= keep.sum() < len(src):
        s, R, t = umeyama(src[keep], dst[keep])
        res = np.linalg.norm((s * src @ R.T + t) - dst, axis=1)
    else:
        keep = np.ones(len(src), bool)
    return s, R, t, res, keep
