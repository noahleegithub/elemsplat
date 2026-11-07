"""
semseg_data.py -- block sampling of converted BrPCD clouds for training and inference.

A block is a block_size x block_size square in the bridge frame (x = axis), full height.
Point features (7 channels): block-local x, y, z (metres), normal (3), and the height in the
cloud normalised to [0, 1].
"""
import os

import numpy as np

try:
    import torch
    from torch.utils.data import IterableDataset, get_worker_info
except ImportError:  # the numpy parts still work without torch
    torch = None
    IterableDataset = object

N_FEAT = 7


class Cloud:
    def __init__(self, path, cell):
        z = np.load(path)
        self.id = os.path.splitext(os.path.basename(path))[0]
        self.xyz = z["xyz"].astype(np.float32)
        self.normal = z["normal"].astype(np.float32)
        self.cls = z["cls"].astype(np.int64)
        self.inst = z["inst"].astype(np.int64)
        self.zmin, self.zmax = float(self.xyz[:, 2].min()), float(self.xyz[:, 2].max())
        self.cell = cell
        self.origin = self.xyz[:, :2].min(0)
        ij = np.floor((self.xyz[:, :2] - self.origin) / cell).astype(np.int64)
        self.shape = ij.max(0) + 1
        key = ij[:, 0] * self.shape[1] + ij[:, 1]
        order = np.argsort(key, kind="stable")
        self.order = order
        ks = key[order]
        self.cell_start = np.searchsorted(ks, np.arange(self.shape[0] * self.shape[1]), side="left")
        self.cell_end = np.searchsorted(ks, np.arange(self.shape[0] * self.shape[1]), side="right")

    def __len__(self):
        return len(self.xyz)

    def block_indices(self, cx, cy, size):
        """Indices of the points within the square block centred at (cx, cy)."""
        lo = np.floor((np.array([cx, cy]) - size / 2 - self.origin) / self.cell).astype(int)
        hi = np.floor((np.array([cx, cy]) + size / 2 - self.origin) / self.cell).astype(int)
        lo, hi = np.maximum(lo, 0), np.minimum(hi, self.shape - 1)
        parts = []
        for i in range(lo[0], hi[0] + 1):
            a = i * self.shape[1] + lo[1]
            b = i * self.shape[1] + hi[1]
            if b < a:
                continue
            s, e = self.cell_start[a], self.cell_end[b]
            if e > s:
                parts.append(self.order[s:e])
        if not parts:
            return np.zeros(0, np.int64)
        idx = np.concatenate(parts)
        p = self.xyz[idx, :2]
        m = (np.abs(p[:, 0] - cx) <= size / 2) & (np.abs(p[:, 1] - cy) <= size / 2)
        return idx[m]

    def features(self, idx, cx, cy):
        x = self.xyz[idx]
        local = np.stack([x[:, 0] - cx, x[:, 1] - cy, x[:, 2] - self.zmin], 1)
        zr = (x[:, 2:3] - self.zmin) / max(self.zmax - self.zmin, 1e-6)
        return np.concatenate([local, self.normal[idx], zr], 1).astype(np.float32)


def load_clouds(data_dir, ids, cell):
    return [Cloud(os.path.join(data_dir, "clouds", f"{b}.npz"), cell) for b in ids]


def class_counts(clouds, n_cls):
    c = np.zeros(n_cls, np.int64)
    for cl in clouds:
        c += np.bincount(cl.cls, minlength=n_cls)[:n_cls]
    return c


def class_weights(counts, power=0.5):
    """Eq. (3): w_c proportional to n_c^-power over the classes present, scaled to mean 1."""
    w = np.zeros(len(counts), np.float64)
    m = counts > 0
    w[m] = counts[m].astype(float) ** (-power)
    w[m] *= m.sum() / w[m].sum()
    return w.astype(np.float32)


class BlockSampler(IterableDataset):
    """Endless random blocks: cloud ~ sqrt(size), centre = a random point of that cloud."""

    def __init__(self, clouds, block_size, n_points, augment, seed=0):
        self.clouds, self.size, self.n, self.aug, self.seed = clouds, block_size, n_points, augment or {}, seed
        w = np.sqrt(np.array([len(c) for c in clouds], float))
        self.p = w / w.sum()

    def sample(self, rng):
        while True:
            c = self.clouds[rng.choice(len(self.clouds), p=self.p)]
            j = rng.integers(len(c))
            cx, cy = c.xyz[j, 0], c.xyz[j, 1]
            idx = c.block_indices(cx, cy, self.size)
            if len(idx) >= 64:
                break
        sel = rng.choice(idx, self.n, replace=len(idx) < self.n)
        f = c.features(sel, cx, cy)
        y = c.cls[sel]
        a = self.aug
        if a.get("rotate_z_deg"):
            th = np.deg2rad(rng.uniform(-a["rotate_z_deg"], a["rotate_z_deg"]))
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
            f[:, :2] = f[:, :2] @ R.T
            f[:, 3:5] = f[:, 3:5] @ R.T
        if a.get("flip_y") and rng.random() < 0.5:
            f[:, 1] *= -1
            f[:, 4] *= -1
        if a.get("scale"):
            f[:, :3] *= rng.uniform(*a["scale"])
        if a.get("jitter"):
            f[:, :3] += rng.normal(0, a["jitter"], (len(f), 3)).astype(np.float32)
        return f.T.copy(), y

    def __iter__(self):
        wi = get_worker_info() if torch is not None else None
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0) * 7919)
        while True:
            f, y = self.sample(rng)
            yield torch.from_numpy(f), torch.from_numpy(y)


def inference_blocks(cloud, size, stride):
    """Block centres on a regular grid that covers the cloud (with overlap = size - stride)."""
    lo, hi = cloud.xyz[:, :2].min(0), cloud.xyz[:, :2].max(0)
    xs = np.arange(lo[0], hi[0] + stride, stride)
    ys = np.arange(lo[1], hi[1] + stride, stride)
    return [(x, y) for x in xs for y in ys]
