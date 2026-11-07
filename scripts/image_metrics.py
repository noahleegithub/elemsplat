#!/usr/bin/env python
"""
image_metrics.py -- PSNR (Eq. 23), SSIM and LPIPS(VGG) on the held-out views, computed the
same way for 3DGS and 2DGS from <model>/test/ours_<iter>/{renders,gt}/*.png.
SSIM uses the 11x11 Gaussian window (sigma 1.5) of the 3DGS code; LPIPS uses the `lpips`
package with the VGG network (weights cached in $TORCH_HOME by 00_setup_env.sh).

    python image_metrics.py --model <model_dir> --iteration 30000 --out metrics_images.json
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log, save_json  # noqa: E402


def ssim_torch(x, y, torch):
    import torch.nn.functional as F
    ws, sigma = 11, 1.5
    g = torch.exp(-((torch.arange(ws, dtype=torch.float32) - ws // 2) ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(x.device)
    w = (g[:, None] @ g[None, :]).expand(3, 1, ws, ws).contiguous()
    mu1 = F.conv2d(x, w, padding=ws // 2, groups=3)
    mu2 = F.conv2d(y, w, padding=ws // 2, groups=3)
    s11 = F.conv2d(x * x, w, padding=ws // 2, groups=3) - mu1 ** 2
    s22 = F.conv2d(y * y, w, padding=ws // 2, groups=3) - mu2 ** 2
    s12 = F.conv2d(x * y, w, padding=ws // 2, groups=3) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return float(m.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--iteration", type=int, default=30000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-lpips", action="store_true")
    a = ap.parse_args()
    import torch
    from PIL import Image
    d = os.path.join(a.model, "test", f"ours_{a.iteration}")
    rend = sorted(glob.glob(os.path.join(d, "renders", "*.png")))
    if not rend:
        sys.exit(f"no renders in {d}/renders -- run render.py first")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lp = None
    if not a.no_lpips:
        try:
            import lpips
            lp = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
        except Exception as e:  # noqa: BLE001
            log(f"LPIPS unavailable ({e}); reporting PSNR and SSIM only")
    per = {}
    for r in rend:
        g = os.path.join(d, "gt", os.path.basename(r))
        if not os.path.exists(g):
            continue
        to_t = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255.0) \
            .permute(2, 0, 1)[None].to(dev)
        x, y = to_t(r), to_t(g)
        if x.shape != y.shape:
            x = torch.nn.functional.interpolate(x, size=y.shape[-2:], mode="bilinear", align_corners=False)
        mse = float(((x - y) ** 2).mean())
        rec = {"psnr": 10 * np.log10(1.0 / max(mse, 1e-12)), "ssim": ssim_torch(x, y, torch)}
        if lp is not None:
            with torch.no_grad():
                rec["lpips"] = float(lp(x * 2 - 1, y * 2 - 1))
        per[os.path.basename(r)] = rec
    keys = sorted({k for v in per.values() for k in v})
    mean = {k: float(np.mean([v[k] for v in per.values() if k in v])) for k in keys}
    save_json({"model": a.model, "iteration": a.iteration, "n_views": len(per), "mean": mean, "per_view": per}, a.out)
    log(f"{a.model}: {len(per)} test views  " + "  ".join(f"{k} {v:.4f}" for k, v in mean.items()))


if __name__ == "__main__":
    main()
