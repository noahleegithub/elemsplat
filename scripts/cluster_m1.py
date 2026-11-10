"""
cluster_m1.py -- M1, the clustering baseline: DBSCAN per predicted class (Eq. 4).

For large eps the points of each class are first thinned to a voxel of eps/3 (never finer
than the data), DBSCAN runs on the thinned set, and every original point takes the label of
its nearest thinned point if that point lies within eps. This keeps memory bounded on dense
TLS clouds; m_pts is then counted at the thinned resolution.
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from geometry import voxel_downsample_idx


def dbscan_labels(xyz, eps, min_pts, base_voxel=0.0, n_jobs=-1):
    vox = max(eps / 3.0, base_voxel)
    idx = voxel_downsample_idx(xyz, vox) if vox > base_voxel * 1.01 else np.arange(len(xyz))
    lab_s = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=n_jobs).fit_predict(xyz[idx])
    if len(idx) == len(xyz):
        return lab_s
    d, j = cKDTree(xyz[idx]).query(xyz, k=1, workers=-1)
    lab = lab_s[j]
    lab[d > eps] = -1
    return lab


def prepare_thinning(xyz, eps, base_voxel=0.0):
    """Voxel thinning + nearest-thinned-point lookup for one eps.

    Both depend only on eps, so a hyperparameter sweep over min_pts can reuse them instead of
    re-querying the whole cloud for every cell -- on a 300 M-point cloud that query is the whole
    cost of tuning.
    """
    vox = max(eps / 3.0, base_voxel)
    if vox > base_voxel * 1.01:
        idx = voxel_downsample_idx(xyz, vox)
        d, j = cKDTree(xyz[idx]).query(xyz, k=1, workers=-1)
        return idx, d, j
    return np.arange(len(xyz)), None, None


def labels_from_prepared(xyz, prep, eps, min_pts, n_jobs=-1):
    """DBSCAN on the prepared thinning, expanded back to every original point."""
    idx, d, j = prep
    lab_s = DBSCAN(eps=eps, min_samples=min_pts, n_jobs=n_jobs).fit_predict(xyz[idx])
    if d is None:
        return lab_s
    lab = lab_s[j]
    lab[d > eps] = -1
    return lab


def run_m1(xyz, cls, params, base_voxel=0.0, n_jobs=-1):
    """params: {class id: (eps, min_pts)}. Returns instance ids unique across classes (-1 = noise)."""
    inst = np.full(len(xyz), -1, np.int64)
    nxt = 0
    for c, (eps, mp) in params.items():
        m = cls == c
        if m.sum() < max(3, mp):
            continue
        lab = dbscan_labels(xyz[m], eps, mp, base_voxel, n_jobs)
        good = lab >= 0
        if not good.any():
            continue
        u, inv = np.unique(lab[good], return_inverse=True)
        idx = np.flatnonzero(m)
        inst[idx[good]] = nxt + np.asarray(inv).ravel()
        nxt += len(u)
    return inst
