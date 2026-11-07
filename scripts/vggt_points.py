#!/usr/bin/env python
"""
vggt_points.py -- feed-forward baseline for Track B: VGGT on the VTTI images (paper Section 3.3).

VGGT (Wang et al., CVPR 2025) predicts cameras, depth maps and point maps for all input views in
one forward pass and receives no camera information. This script
  1. takes a uniform subset of the registered images (at most vggt.max_frames, one pass in bf16),
  2. unprojects the predicted depth maps with the predicted cameras (the point source the authors
     recommend) and keeps the pixels whose depth confidence is above the vggt.conf_percentile,
  3. fits a similarity transform from the predicted camera centres to the COLMAP centres of the
     same images (Umeyama with one rejection pass) and maps the points into the scene frame, so
     03e_measure.slurm and 03f_merge.slurm treat them exactly like the 3DGS and 2DGS points,
  4. writes pts_vggt.npz (same format as pts_3dgs.npz), run_info.json (frames, time, GPU,
     alignment residuals), cameras_aligned.npz and predictions.npz, from which steps 2 to 4 can be
     repeated on a CPU (--predictions).

The COLMAP poses are used only after the network has run, to express the result in the shared
frame; the scale itself is fixed later by measure_vtti.py from the 150 ft main span, as for the
other methods.

    python vggt_points.py --scene $VTTI_SCENE --out $RESULTS_DIR/trackB/vggt          # GPU
    python vggt_points.py --scene $VTTI_SCENE --out ... --predictions .../predictions.npz   # CPU
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colmap_io as cio  # noqa: E402
from common import load_cfg, load_json, log, save_json  # noqa: E402
from geometry import robust_umeyama  # noqa: E402

TARGET = 518      # VGGT input width (vggt.utils.load_fn)
PATCH = 14        # ViT patch size: image sides are multiples of 14
WEIGHTS_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


# ----------------------------------------------------------------------------- frames and images
def select_frames(names, max_frames):
    """Uniform subset of the sorted image names (all of them when there are at most max_frames)."""
    names = sorted(names)
    if len(names) <= max_frames:
        return names
    idx = np.unique(np.round(np.linspace(0, len(names) - 1, max_frames)).astype(int))
    return [names[i] for i in idx]


def load_images(paths):
    """
    Same preprocessing as vggt.utils.load_fn.load_and_preprocess_images(mode="crop") without the
    torchvision dependency: width 518, height scaled to a multiple of 14 and centre-cropped to at
    most 518, bicubic resampling, values in [0, 1], tensor (S, 3, H, W). Images of different sizes
    are padded with white to the largest one, as in the original.
    """
    import torch
    from PIL import Image
    arrs = []
    for p in paths:
        im = Image.open(p)
        if im.mode == "RGBA":
            im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
        im = im.convert("RGB")
        w, h = im.size
        nw = TARGET
        nh = round(h * (nw / w) / PATCH) * PATCH
        im = im.resize((nw, nh), Image.Resampling.BICUBIC)
        a = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if nh > TARGET:
            y0 = (nh - TARGET) // 2
            a = a[:, y0:y0 + TARGET, :]
        arrs.append(a)
    H = max(a.shape[1] for a in arrs)
    W = max(a.shape[2] for a in arrs)
    out = np.ones((len(arrs), 3, H, W), np.float32)
    for k, a in enumerate(arrs):
        y0, x0 = (H - a.shape[1]) // 2, (W - a.shape[2]) // 2
        out[k, :, y0:y0 + a.shape[1], x0:x0 + a.shape[2]] = a
    return torch.from_numpy(out)


# ----------------------------------------------------------------------------- the network
def load_model(device, weights=None):
    """VGGT-1B from the Hugging Face cache (HF_HOME, offline on compute nodes), else from a local
    model.pt (--weights or $TORCH_HOME/hub/checkpoints/model.pt), else downloaded from WEIGHTS_URL."""
    import torch
    from vggt.models.vggt import VGGT
    if weights is None:
        try:
            return VGGT.from_pretrained("facebook/VGGT-1B").to(device), "facebook/VGGT-1B (hub cache)"
        except Exception as e:  # noqa: BLE001  -- no cache and no network, try the plain checkpoint
            log(f"from_pretrained failed ({type(e).__name__}); trying model.pt")
        weights = os.path.join(os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")),
                               "hub", "checkpoints", "model.pt")
    model = VGGT()
    if os.path.exists(weights):
        state = torch.load(weights, map_location="cpu", weights_only=True)
    else:
        state = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device), weights


def run_model(images, device, weights=None):
    """One forward pass. Returns extrinsics (S,3,4) camera-from-world, intrinsics (S,3,3) at the
    network resolution, depth (S,H,W) and depth confidence (S,H,W), all numpy float32."""
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    model, src = load_model(device, weights)
    model.eval()
    use_amp = device.type == "cuda"
    dtype = torch.bfloat16 if (use_amp and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    log(f"VGGT weights: {src}; {len(images)} frames of {tuple(images.shape[-2:])} on {device} ({dtype if use_amp else 'fp32'})")
    t0 = time.time()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            pred = model(images.to(device))
    ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    depth = pred["depth"].squeeze(0).squeeze(-1)
    conf = pred["depth_conf"].squeeze(0)
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if use_amp else 0.0
    log(f"forward pass {secs:.1f} s, peak GPU memory {peak:.1f} GB")
    out = {"extrinsic": ext.squeeze(0).float().cpu().numpy(), "intrinsic": intr.squeeze(0).float().cpu().numpy(),
           "depth": depth.float().cpu().numpy(), "conf": conf.float().cpu().numpy()}
    return out, secs, peak


# ----------------------------------------------------------------------------- geometry
def camera_centres(extrinsic):
    """Centres of camera-from-world [R|t] matrices: c = -R^T t."""
    R, t = extrinsic[:, :, :3], extrinsic[:, :, 3]
    return -np.einsum("sji,sj->si", R, t)


def unproject(depth, extrinsic, intrinsic):
    """Depth maps (S,H,W) to world points (S,H,W,3): x_cam = depth K^-1 [u, v, 1], x = R^T (x_cam - t).
    Pixel (u, v) is the array index, as in vggt.utils.geometry.depth_to_world_coords_points."""
    S, H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    out = np.empty((S, H, W, 3), np.float32)
    for k in range(S):
        fu, fv, cu, cv = intrinsic[k, 0, 0], intrinsic[k, 1, 1], intrinsic[k, 0, 2], intrinsic[k, 1, 2]
        cam = np.stack(((u - cu) * depth[k] / fu, (v - cv) * depth[k] / fv, depth[k]), -1)
        R, t = extrinsic[k, :, :3], extrinsic[k, :, 3]
        out[k] = (cam - t) @ R          # (x_cam - t) R  ==  R^T (x_cam - t) for every pixel
    return out


def points_in_scene_frame(pred, names, colmap_centres, conf_percentile, max_points, seed=0):
    """Confidence filter, similarity alignment on the camera centres, uniform subsample."""
    c_pred = camera_centres(pred["extrinsic"])
    c_ref = np.array([colmap_centres[n] for n in names])
    s, R, t, res, keep = robust_umeyama(c_pred, c_ref)
    c_al = s * c_pred @ R.T + t
    valid = pred["depth"] > 1e-6
    thr = float(np.percentile(pred["conf"][valid], conf_percentile))
    mask = (pred["conf"] >= thr) & valid
    pts = unproject(pred["depth"], pred["extrinsic"], pred["intrinsic"])[mask]
    pts = (s * pts @ R.T + t).astype(np.float32)
    rng = np.random.default_rng(seed)
    if len(pts) > max_points:
        pts = pts[rng.permutation(len(pts))[:max_points]]
    align = {"scale": s, "cameras_used": int(keep.sum()), "cameras_total": int(len(keep)),
             "residual_rms_sfm": float(np.sqrt(np.mean(res[keep] ** 2))), "residual_max_sfm": float(res.max()),
             "rejected": [names[i] for i in np.flatnonzero(~keep)]}
    return pts, align, thr, float(mask.mean()), c_al


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="VTTI scene: images/ and sparse/0 (02_prepare)")
    ap.add_argument("--out", required=True, help="folder for predictions.npz, cameras_aligned.npz, run_info.json")
    ap.add_argument("--points-out", default=None, help="output cloud (default <out>/pts_vggt.npz)")
    ap.add_argument("--predictions", default=None, help="reuse saved predictions (no GPU, no vggt package)")
    ap.add_argument("--weights", default=None, help="local model.pt (default: hub cache, then $TORCH_HOME)")
    ap.add_argument("--cfg", default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    vc = cfg["vggt"]
    os.makedirs(a.out, exist_ok=True)
    _, ims, _ = cio.read_model(os.path.join(a.scene, "sparse", "0"))
    poses = cio.camera_poses(ims)
    centres = {n: p[0] for n, p in poses.items()}
    info = {"method": "vggt", "n_registered": len(poses), "conf_percentile": vc["conf_percentile"],
            "max_frames": vc["max_frames"], "job": os.environ.get("SLURM_JOB_ID")}

    if a.predictions:
        z = np.load(a.predictions, allow_pickle=False)
        names = [str(n) for n in z["names"]]
        pred = {k: z[k].astype(np.float32) for k in ("extrinsic", "intrinsic", "depth", "conf")}
        info.update({"predictions": a.predictions, "n_frames": len(names)})
        log(f"reusing {a.predictions}: {len(names)} frames")
    else:
        import torch
        names = select_frames(list(poses), vc["max_frames"])
        paths = [os.path.join(a.scene, "images", n) for n in names]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            sys.exit(f"ERROR: {len(missing)} selected images are missing, e.g. {missing[0]}")
        images = load_images(paths)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred, secs, peak = run_model(images, device, a.weights)
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip() if device.type == "cuda" else "cpu"
        np.savez_compressed(os.path.join(a.out, "predictions.npz"), names=np.array(names), image_hw=np.array(images.shape[-2:]),
                            extrinsic=pred["extrinsic"], intrinsic=pred["intrinsic"], depth=pred["depth"],
                            conf=pred["conf"].astype(np.float16))
        info.update({"n_frames": len(names), "image_hw": [int(v) for v in images.shape[-2:]], "forward_seconds": secs,
                     "peak_gpu_gb": peak, "gpu": gpu, "torch": torch.__version__})
        try:
            import vggt
            info["vggt_commit"] = subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(vggt.__file__)), "rev-parse",
                                                  "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        except Exception:  # noqa: BLE001
            pass

    pts, align, thr, kept, c_al = points_in_scene_frame(pred, names, centres, vc["conf_percentile"], vc["max_points"])
    k0 = None
    s0 = os.path.join(os.path.dirname(a.out.rstrip("/")), "scale0.json")
    if os.path.exists(s0):
        k0 = load_json(s0)["kappa0"]
        align["residual_rms_m_provisional"] = align["residual_rms_sfm"] * k0
    info.update({"alignment": align, "conf_threshold": thr, "pixels_kept": kept, "n_points": int(len(pts))})
    pts_path = a.points_out or os.path.join(a.out, "pts_vggt.npz")
    os.makedirs(os.path.dirname(os.path.abspath(pts_path)), exist_ok=True)
    np.savez_compressed(pts_path, xyz=pts, n_frames=len(names), conf_percentile=vc["conf_percentile"])
    np.savez_compressed(os.path.join(a.out, "cameras_aligned.npz"), names=np.array(names), centres=c_al.astype(np.float32),
                        centres_colmap=np.array([centres[n] for n in names], np.float32))
    save_json(info, os.path.join(a.out, "run_info.json"))
    log(f"{pts_path}: {len(pts):,} points from {len(names)} frames ({100 * kept:.0f}% of pixels above conf {thr:.2f}); "
        f"camera alignment RMS {align['residual_rms_sfm']:.4f} SfM units"
        + (f" = {align['residual_rms_m_provisional']:.2f} m (provisional scale)" if k0 else "")
        + f", {align['cameras_used']}/{align['cameras_total']} cameras used")


if __name__ == "__main__":
    main()
