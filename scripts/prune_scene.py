#!/usr/bin/env python
"""
prune_scene.py -- remove registered images that have no file in <scene>/images.

pycolmap.undistort_images keeps every camera of the source model, so a project that
registers images absent from the image folder produces a scene that 3DGS/2DGS cannot
open: the loader builds its camera list from images.bin and then dies on the first
missing file. VTTI's FRO project registers 292 images while the Full Resolution
archive holds 280 (12 DJI_ frames live only in the reduced-resolution set), so the
scene has to be pruned to the frames that actually exist.

The SfM frame is untouched: dropping a camera removes its observations, and the
remaining cameras, intrinsics and 3D points stay in the same coordinate system.

    python prune_scene.py --scene $VTTI_SCENE              # prune in place, keep sparse/0_dropped
    python prune_scene.py --scene $VTTI_SCENE --dry-run    # report only
    python prune_scene.py --scene $VTTI_SCENE --max-frac 0.05   # refuse if more than 5% is missing

Exit codes: 0 = pruned or nothing to do, 1 = refused / verification failed.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--sparse", default=None, help="model subfolder (default: <scene>/sparse/0)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-frac", type=float, default=0.10,
                    help="refuse to prune when more than this fraction of images is missing")
    ap.add_argument("--keep-original", action="store_true",
                    help="keep the unpruned model next to the pruned one as sparse/0_dropped")
    a = ap.parse_args()

    import pycolmap
    root = os.path.join(a.scene, "sparse")
    model = a.sparse or os.path.join(root, "0")
    img_dir = os.path.join(a.scene, "images")
    if not os.path.isdir(img_dir):
        sys.exit(f"ERROR: no image folder at {img_dir}")

    rec = pycolmap.Reconstruction(model)
    on_disk = set(os.listdir(img_dir))
    missing = sorted(im.name for im in rec.images.values() if im.name not in on_disk)
    n_img = len(rec.images)
    log(f"model {model}: {n_img} images registered, {len(on_disk)} files on disk, "
        f"{len(missing)} registered without a file, {len(rec.points3D)} points3D")

    report = {"model": model, "n_images_before": n_img, "n_files_on_disk": len(on_disk),
              "n_missing": len(missing), "missing": missing[:50], "pruned": False}
    if not missing:
        log("nothing to prune")
        save_json(report, os.path.join(a.scene, "prune_report.json"))
        return 0
    frac = len(missing) / max(n_img, 1)
    if frac > a.max_frac:
        log(f"REFUSING: {frac:.1%} of the images are missing (> --max-frac {a.max_frac:.0%}); "
            f"check --image-dir before pruning")
        save_json(report, os.path.join(a.scene, "prune_report.json"))
        return 1
    if a.dry_run:
        log("dry run: would drop " + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
        save_json(report, os.path.join(a.scene, "prune_report.json"))
        return 0

    # one frame may hold several images; drop the frame only when all of its images are missing
    kept_ids = {im.image_id for im in rec.images.values() if im.name in on_disk}
    frame_ids = sorted({im.frame_id for im in rec.images.values() if im.name not in on_disk})
    droppable = [f for f in frame_ids if not (set(rec.frames[f].image_ids) & kept_ids)]
    for f in droppable:
        rec.deregister_frame(f)
    if len(rec.images) != len(kept_ids):
        log(f"WARNING: {len(rec.images)} images left, expected {len(kept_ids)}")

    tmp = os.path.join(root, "0_pruned")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)          # pycolmap.write() will not create the folder
    rec.write(str(tmp))

    # verify before touching the live model
    chk = pycolmap.Reconstruction(tmp)
    still = [im.name for im in chk.images.values() if im.name not in on_disk]
    if still:
        sys.exit(f"ERROR: pruned model still references {len(still)} missing files, e.g. {still[0]}")
    if len(chk.images) != len(kept_ids):
        sys.exit(f"ERROR: pruned model has {len(chk.images)} images, expected {len(kept_ids)}")

    backup = os.path.join(root, "0_dropped")
    shutil.rmtree(backup, ignore_errors=True)
    if a.keep_original:
        shutil.copytree(model, backup)
    shutil.rmtree(model)
    shutil.move(tmp, model)

    report.update({"pruned": True, "n_images_after": len(chk.images), "n_points3D_after": len(chk.points3D),
                   "frames_dropped": len(droppable), "backup": backup if a.keep_original else None})
    save_json(report, os.path.join(a.scene, "prune_report.json"))
    log(f"pruned {len(missing)} images: {n_img} -> {len(chk.images)} images, "
        f"{len(chk.points3D)} points3D; scene at {a.scene}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
