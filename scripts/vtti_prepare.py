#!/usr/bin/env python
"""
vtti_prepare.py -- VTTI images + shipped COLMAP model -> a 3DGS/2DGS scene (paper Section 3.4).

1. find the COLMAP sparse model inside the extracted <model>.zip (largest one) and its images
2. report registered images, camera model, image size, mean reprojection error, track length
3. undistort to PINHOLE (pycolmap) unless the cameras are already pinhole
4. downsample to vtti.long_side_px and rescale cameras.bin to match
5. list the test views (every 8th sorted name, the 3DGS --eval rule) and read EXIF GPS
6. provisional frame and scale kappa0 from the sparse points (for metric thresholds/TSDF voxels)

    python vtti_prepare.py --vtti $VTTI_DIR --scene $VTTI_SCENE --model FRO --out $RESULTS_DIR/trackB
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colmap_io as cio  # noqa: E402
from bridge_frame import provisional  # noqa: E402
from common import FT_TO_M, load_cfg, log, save_json  # noqa: E402

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG")


def find_models(root):
    out = []
    for d, _, files in os.walk(root):
        names = set(files)
        for ext in (".bin", ".txt"):
            if {"cameras" + ext, "images" + ext, "points3D" + ext} <= names:
                out.append(d)
                break
    return out


def find_image_dir(root, names):
    want = set(os.path.basename(n) for n in names)
    best, best_hits = None, 0
    for d, _, files in os.walk(root):
        hits = len(want & set(files))
        if hits > best_hits:
            best, best_hits = d, hits
    return best, best_hits


def gps_from_exif(path):
    try:
        from PIL import Image
        ex = Image.open(path).getexif()
        g = ex.get_ifd(0x8825)
        if not g or 2 not in g or 4 not in g:
            return None
        deg = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
        lat = deg(g[2]) * (-1 if g.get(1, "N") in ("S", b"S") else 1)
        lon = deg(g[4]) * (-1 if g.get(3, "E") in ("W", b"W") else 1)
        alt = float(g.get(6, 0.0)) * (-1 if g.get(5, 0) in (1, b"\x01") else 1)
        return lat, lon, alt
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtti", required=True, help="folder with the extracted VTTI zips")
    ap.add_argument("--scene", required=True, help="output scene folder for 3DGS/2DGS")
    ap.add_argument("--model", default="FRO")
    ap.add_argument("--out", required=True, help="results folder (trackB)")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--threads", type=int, default=-1)
    a = ap.parse_args()
    cfg = load_cfg(a.cfg)
    vc = cfg["vtti"]
    os.makedirs(a.out, exist_ok=True)
    root = os.path.join(a.vtti, a.model) if os.path.isdir(os.path.join(a.vtti, a.model)) else a.vtti
    models = find_models(root)
    if not models:
        sys.exit(f"no COLMAP model (cameras/images/points3D) under {root}")
    sizes = []
    for m in models:
        _, ims, _ = cio.read_model(m)
        sizes.append(len(ims))
        log(f"model {m}: {len(ims)} images")
    model_dir = models[int(np.argmax(sizes))]
    cams, ims, pts = cio.read_model(model_dir)
    names = sorted(im.name for im in ims.values())
    img_dir, hits = find_image_dir(a.vtti, names)
    log(f"using model {model_dir} ({len(ims)} registered images); images in {img_dir} ({hits}/{len(names)} found)")
    if hits < len(names):
        log("WARNING: some registered images were not found; they will be missing from the scene")
    errs = np.array([p.error for p in pts.values()])
    tracks = np.array([p.track_length for p in pts.values()])
    cam_models = sorted({c.model for c in cams.values()})
    report = {"model_dir": model_dir, "image_dir": img_dir, "n_registered": len(ims), "n_found": hits,
              "n_points3D": len(pts), "mean_reproj_px": float(errs.mean()) if len(errs) else None,
              "mean_track": float(tracks.mean()) if len(tracks) else None, "camera_models": cam_models,
              "image_sizes": sorted({(c.width, c.height) for c in cams.values()})}
    log(f"COLMAP: {report}")

    # ---- undistort (or copy) into a temporary scene ------------------------------------------
    tmp = a.scene + "_full"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    t0 = time.time()
    if all(m in ("PINHOLE", "SIMPLE_PINHOLE") for m in cam_models):
        os.makedirs(os.path.join(tmp, "images"))
        for n in names:
            src = os.path.join(img_dir, os.path.basename(n))
            if os.path.exists(src):
                dst = os.path.join(tmp, "images", n)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        shutil.copytree(model_dir, os.path.join(tmp, "sparse", "0"))
    else:
        import pycolmap
        pycolmap.undistort_images(tmp, model_dir, img_dir, output_type="COLMAP", num_threads=a.threads)
        sp = os.path.join(tmp, "sparse")
        os.makedirs(os.path.join(sp, "0"), exist_ok=True)
        for f in os.listdir(sp):
            if f != "0":
                shutil.move(os.path.join(sp, f), os.path.join(sp, "0", f))
        for junk in ("stereo", "run-colmap-geometric.sh", "run-colmap-photometric.sh"):
            p = os.path.join(tmp, junk)
            shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else (os.path.exists(p) and os.remove(p))
    log(f"undistortion/copy done in {time.time() - t0:.0f} s")

    # ---- downsample to long side and rescale cameras ---------------------------------------------
    from PIL import Image
    shutil.rmtree(a.scene, ignore_errors=True)
    os.makedirs(os.path.join(a.scene, "images"), exist_ok=True)
    shutil.copytree(os.path.join(tmp, "sparse", "0"), os.path.join(a.scene, "sparse", "0"))
    ucams, uims, _ = cio.read_model(os.path.join(a.scene, "sparse", "0"))
    scale_of_cam = {}
    for im in uims.values():
        src = os.path.join(tmp, "images", im.name)
        if not os.path.exists(src):
            continue
        with Image.open(src) as I:
            w, h = I.size
            s = min(1.0, vc["long_side_px"] / max(w, h))
            dst = os.path.join(a.scene, "images", im.name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if s < 1.0:
                I.convert("RGB").resize((round(w * s), round(h * s)), Image.LANCZOS).save(dst, quality=95)
            else:
                shutil.copy2(src, dst)
        scale_of_cam.setdefault(im.camera_id, (s, w, h))
    new = {}
    for cid, c in ucams.items():
        s, w, h = scale_of_cam.get(cid, (1.0, c.width, c.height))
        p = list(c.params)
        p = [v * s for v in p]                               # PINHOLE fx fy cx cy / SIMPLE_PINHOLE f cx cy
        new[cid] = cio.Camera(cid, c.model, round(c.width * s), round(c.height * s), p)
    bin_path = os.path.join(a.scene, "sparse", "0", "cameras.bin")
    txt_path = os.path.join(a.scene, "sparse", "0", "cameras.txt")
    if os.path.exists(txt_path):
        os.remove(txt_path)
    cio.write_cameras_bin(new, bin_path)
    shutil.rmtree(tmp, ignore_errors=True)

    # ---- drop registered images that have no file ---------------------------------------------
    # pycolmap.undistort_images keeps every camera of the source model, but the FRO project
    # registers 12 DJI_ frames that are not in the Full Resolution archive. Left in, they enter
    # the train/test split and 3DGS dies on the first missing file, so prune before splitting.
    pr = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "prune_scene.py"),
                         "--scene", a.scene, "--max-frac", "0.10", "--keep-original"],
                        capture_output=True, text=True)
    log((pr.stdout.strip() or pr.stderr.strip())[-500:])
    if pr.returncode != 0:
        log("WARNING: scene pruning failed -- the scene may reference images that do not exist")
    ucams, uims, _ = cio.read_model(os.path.join(a.scene, "sparse", "0"))
    names = sorted(im.name for im in uims.values())
    report["n_images_scene"] = len(uims)

    # ---- split, GPS, provisional frame -------------------------------------------------------
    train, test = cio.llff_split([im.name for im in uims.values()], vc["test_every"])
    with open(os.path.join(a.out, "test_views.txt"), "w") as f:
        f.write("\n".join(test) + "\n")
    gps_rows = []
    for n in names:
        g = gps_from_exif(os.path.join(img_dir, os.path.basename(n)))
        if g:
            gps_rows.append((n, *g))
    if gps_rows:
        with open(os.path.join(a.out, "gps.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "lat", "lon", "alt"])
            w.writerows(gps_rows)
    xyz = np.array([p.xyz for p in pts.values()])
    poses = cio.camera_poses(ims)
    vdirs = np.array([v for _, v in poses.values()])
    total_m = sum(vc["spans_ft"]) * FT_TO_M
    prov = provisional(xyz, vdirs, total_m, cfg["measure"])
    g2 = cfg["gs2d"]
    k0 = prov["kappa0"]
    prov["tsdf_voxel_sfm"] = g2["tsdf_voxel_m"] / k0
    prov["sdf_trunc_sfm"] = g2["sdf_trunc_voxels"] * g2["tsdf_voxel_m"] / k0
    prov["depth_trunc_sfm"] = g2["depth_trunc_m"] / k0
    report.update({"n_train": len(train), "n_test": len(test), "n_gps": len(gps_rows), "provisional": prov,
                   "scene": a.scene, "long_side_px": vc["long_side_px"]})
    save_json(report, os.path.join(a.out, "vtti_prepare.json"))
    save_json(prov, os.path.join(a.out, "scale0.json"))
    log(f"scene ready: {a.scene}  train {len(train)} / test {len(test)}  kappa0 {k0:.4f} m per SfM unit  "
        f"TSDF voxel {prov['tsdf_voxel_sfm']:.4g} SfM units  GPS tags: {len(gps_rows)}")


if __name__ == "__main__":
    main()
