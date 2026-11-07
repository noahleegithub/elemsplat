"""
semseg_models.py -- PointNet and PointNet++ (SSG) segmentation networks in plain PyTorch.

No compiled CUDA ops: farthest point sampling, ball query and three-nearest-neighbour
interpolation are written with torch tensor ops, so the networks run wherever the PyTorch
wheel runs (HPRC A30/H100/A100, or CPU for the smoke test).
Input: features (B, C, N) whose first three channels are block-local xyz in metres.
Output: logits (B, n_classes, N).
"""
import torch
import torch.nn as nn


def _mlp1d(chs, last_act=True):
    layers = []
    for i in range(len(chs) - 1):
        layers += [nn.Conv1d(chs[i], chs[i + 1], 1), nn.BatchNorm1d(chs[i + 1])]
        if last_act or i < len(chs) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------- PointNet
class PointNetSeg(nn.Module):
    """PointNet segmentation (Qi et al., 2017a) without the input/feature transform nets."""

    def __init__(self, in_ch, n_cls):
        super().__init__()
        self.local = _mlp1d([in_ch, 64, 64])
        self.glob = _mlp1d([64, 64, 128, 1024])
        self.head = nn.Sequential(_mlp1d([64 + 1024, 512, 256, 128]), nn.Dropout(0.3), nn.Conv1d(128, n_cls, 1))

    def forward(self, x):
        f1 = self.local(x)
        g = self.glob(f1).max(2, keepdim=True)[0].expand(-1, -1, x.shape[2])
        return self.head(torch.cat([f1, g], 1))


# ----------------------------------------------------------------------------- PointNet++
def index_points(points, idx):
    """points (B, N, C), idx (B, ...) -> (B, ..., C)."""
    B = points.shape[0]
    flat = idx.reshape(B, -1)
    out = torch.gather(points, 1, flat.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
    return out.reshape(*idx.shape, points.shape[-1])


def farthest_point_sample(xyz, npoint, deterministic=False):
    B, N, _ = xyz.shape
    npoint = min(npoint, N)
    cent = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device, dtype=xyz.dtype)
    far = torch.zeros(B, dtype=torch.long, device=xyz.device) if deterministic else \
        torch.randint(0, N, (B,), device=xyz.device)
    ar = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        cent[:, i] = far
        c = xyz[ar, far].unsqueeze(1)
        dist = torch.minimum(dist, ((xyz - c) ** 2).sum(-1))
        far = dist.argmax(-1)
    return cent


def ball_query(radius, nsample, xyz, new_xyz):
    """Up to nsample nearest neighbours within radius (padded with the nearest one)."""
    d = torch.cdist(new_xyz, xyz) ** 2                     # (B, S, N)
    nsample = min(nsample, xyz.shape[1])
    vals, idx = d.topk(nsample, dim=-1, largest=False)
    first = idx[..., :1].expand_as(idx)
    return torch.where(vals > radius * radius, first, idx)


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_ch, mlp):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        layers, c = [], in_ch + 3
        for m in mlp:
            layers += [nn.Conv2d(c, m, 1), nn.BatchNorm2d(m), nn.ReLU(inplace=True)]
            c = m
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, feats):
        """xyz (B, N, 3), feats (B, N, C) -> new_xyz (B, S, 3), new_feats (B, S, C')."""
        fps = farthest_point_sample(xyz, self.npoint, deterministic=not self.training)
        new_xyz = index_points(xyz, fps)
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        g_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)
        g = torch.cat([g_xyz / self.radius, index_points(feats, idx)], -1) if feats is not None else g_xyz / self.radius
        g = self.mlp(g.permute(0, 3, 2, 1))                    # (B, C', nsample, S)
        return new_xyz, g.max(2)[0].transpose(1, 2)


class FeaturePropagation(nn.Module):
    def __init__(self, in_ch, mlp):
        super().__init__()
        self.mlp = _mlp1d([in_ch] + mlp)

    def forward(self, xyz1, xyz2, f1, f2):
        """Interpolate f2 (at xyz2) onto xyz1 by inverse distance of 3 neighbours, concat f1."""
        d = torch.cdist(xyz1, xyz2) ** 2
        k = min(3, xyz2.shape[1])
        dk, ik = d.topk(k, dim=-1, largest=False)
        w = 1.0 / (dk + 1e-8)
        w = w / w.sum(-1, keepdim=True)
        interp = (index_points(f2, ik) * w.unsqueeze(-1)).sum(2)
        x = torch.cat([f1, interp], -1) if f1 is not None else interp
        return self.mlp(x.transpose(1, 2)).transpose(1, 2)


class PointNet2Seg(nn.Module):
    """PointNet++ single-scale grouping (Qi et al., 2017b), radii in metres for 10 m blocks."""

    def __init__(self, in_ch, n_cls, radii=(0.5, 1.0, 2.0, 4.0), npoints=(1024, 256, 64, 16), nsample=32):
        super().__init__()
        f = in_ch - 3                                           # channels besides xyz
        self.sa1 = SetAbstraction(npoints[0], radii[0], nsample, f, [32, 32, 64])
        self.sa2 = SetAbstraction(npoints[1], radii[1], nsample, 64, [64, 64, 128])
        self.sa3 = SetAbstraction(npoints[2], radii[2], nsample, 128, [128, 128, 256])
        self.sa4 = SetAbstraction(npoints[3], radii[3], nsample, 256, [256, 256, 512])
        self.fp4 = FeaturePropagation(256 + 512, [256, 256])
        self.fp3 = FeaturePropagation(128 + 256, [256, 256])
        self.fp2 = FeaturePropagation(64 + 256, [256, 128])
        self.fp1 = FeaturePropagation(f + 128, [128, 128, 128])
        self.head = nn.Sequential(nn.Conv1d(128, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.5), nn.Conv1d(128, n_cls, 1))

    def forward(self, x):
        xyz = x[:, :3].transpose(1, 2).contiguous()
        f0 = x[:, 3:].transpose(1, 2).contiguous() if x.shape[1] > 3 else None
        l1x, l1f = self.sa1(xyz, f0)
        l2x, l2f = self.sa2(l1x, l1f)
        l3x, l3f = self.sa3(l2x, l2f)
        l4x, l4f = self.sa4(l3x, l3f)
        l3f = self.fp4(l3x, l4x, l3f, l4f)
        l2f = self.fp3(l2x, l3x, l2f, l3f)
        l1f = self.fp2(l1x, l2x, l1f, l2f)
        l0f = self.fp1(xyz, l1x, f0, l1f)
        return self.head(l0f.transpose(1, 2))


def build_model(name, in_ch, n_cls):
    if name == "pointnet":
        return PointNetSeg(in_ch, n_cls)
    if name == "pointnet2":
        return PointNet2Seg(in_ch, n_cls)
    raise ValueError(f"unknown model {name} (pointnet | pointnet2)")
