#!/usr/bin/env python
"""
convert_pycolmap.py -- drop-in replacement for the official  convert.py  that uses
pycolmap (COLMAP's official Python bindings, `pip install pycolmap`) instead of the
`colmap` command-line binary.  Same stages, same options, same output layout:

    <scene>/input/*.jpg                       (your images)
    <scene>/distorted/database.db             feature_extractor  (OPENCV model, single camera)
                                              exhaustive_matcher
    <scene>/distorted/sparse/0/               mapper (ba_global_function_tolerance = 1e-6)
    <scene>/images/, <scene>/sparse/0/        image_undistorter  -> PINHOLE camera, what 3DGS reads

Usage:
    python convert_pycolmap.py -s <scene_dir> [--camera OPENCV] [--gpu] [--threads N]

Why this exists: on some clusters the conda-forge `colmap` binary is broken
(missing shared libraries) while the pycolmap wheel is self-contained.
"""
import argparse
import os
import shutil
import sys
import time

import pycolmap


def banner(t):
    print(f"\n==== {t}  [{time.strftime('%H:%M:%S')}] ====", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--source_path", required=True)
    ap.add_argument("--camera", default="OPENCV", help="camera model for SfM (convert.py default: OPENCV)")
    ap.add_argument("--gpu", action="store_true", help="use CUDA SIFT if the wheel/GPU support it (default: CPU)")
    ap.add_argument("--threads", type=int, default=-1)
    ap.add_argument("--skip_matching", action="store_true")
    a = ap.parse_args()

    scene = os.path.abspath(a.source_path)
    image_dir = os.path.join(scene, "input")
    dist_dir = os.path.join(scene, "distorted")
    db_path = os.path.join(dist_dir, "database.db")
    sparse_dist = os.path.join(dist_dir, "sparse")
    device = pycolmap.Device.cuda if a.gpu else pycolmap.Device.cpu
    print(f"pycolmap {pycolmap.__version__} | scene {scene} | device {device.name} | camera {a.camera}")
    if not os.path.isdir(image_dir):
        sys.exit(f"ERROR: {image_dir} not found")
    n_img = len([f for f in os.listdir(image_dir) if not f.startswith('._')])
    print(f"input images: {n_img}")
    t_all = time.time()

    if not a.skip_matching:
        os.makedirs(sparse_dist, exist_ok=True)
        if os.path.exists(db_path):
            os.remove(db_path)

        banner("1/4 feature_extractor")
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = a.camera                       # --ImageReader.camera_model OPENCV
        ext = pycolmap.FeatureExtractionOptions()
        ext.use_gpu = a.gpu                                  # --SiftExtraction.use_gpu
        ext.num_threads = a.threads
        pycolmap.extract_features(db_path, image_dir,
                                  camera_mode=pycolmap.CameraMode.SINGLE,   # --ImageReader.single_camera 1
                                  reader_options=reader, extraction_options=ext, device=device)

        banner("2/4 exhaustive_matcher")
        mo = pycolmap.FeatureMatchingOptions()
        mo.use_gpu = a.gpu                                   # --SiftMatching.use_gpu
        mo.num_threads = a.threads
        pycolmap.match_exhaustive(db_path, matching_options=mo, device=device)

        banner("3/4 mapper")
        opt = pycolmap.IncrementalPipelineOptions()
        opt.ba_global_function_tolerance = 1e-6              # --Mapper.ba_global_function_tolerance=0.000001
        opt.num_threads = a.threads
        recs = pycolmap.incremental_mapping(db_path, image_dir, sparse_dist, options=opt)
        if not recs:
            sys.exit("ERROR: mapper produced no reconstruction (too few matches?)")
        for k, r in sorted(recs.items()):
            print(f"  model {k}: {r.num_reg_images()} registered images, {r.num_points3D()} points, "
                  f"mean reproj. error {r.compute_mean_reprojection_error():.3f} px")
        # convert.py always continues with distorted/sparse/0.  If COLMAP split the scene into
        # several models, put the LARGEST one at index 0 so downstream steps use it.
        best = max(recs, key=lambda k: recs[k].num_reg_images())
        if best != 0:
            print(f"  WARNING: {len(recs)} models; moving largest model {best} -> 0")
            tmp = os.path.join(sparse_dist, "tmp_swap")
            os.rename(os.path.join(sparse_dist, "0"), tmp)
            os.rename(os.path.join(sparse_dist, str(best)), os.path.join(sparse_dist, "0"))
            os.rename(tmp, os.path.join(sparse_dist, str(best)))
        if len(recs) > 1:
            print("  WARNING: multiple models means some images could not be connected -- check colmap_summary.txt")

    banner("4/4 image_undistorter  (-> images/, sparse/0/, PINHOLE)")
    for d in ("images", "sparse", "stereo"):
        shutil.rmtree(os.path.join(scene, d), ignore_errors=True)
    pycolmap.undistort_images(scene, os.path.join(sparse_dist, "0"), image_dir,
                              output_type="COLMAP", num_threads=a.threads)
    # same post-processing as convert.py: sparse/* -> sparse/0/*
    sparse = os.path.join(scene, "sparse")
    os.makedirs(os.path.join(sparse, "0"), exist_ok=True)
    for f in os.listdir(sparse):
        if f == "0":
            continue
        shutil.move(os.path.join(sparse, f), os.path.join(sparse, "0", f))
    for junk in ("stereo", "run-colmap-geometric.sh", "run-colmap-photometric.sh"):
        p = os.path.join(scene, junk)
        (shutil.rmtree if os.path.isdir(p) else lambda q: os.path.exists(q) and os.remove(q))(p)

    n_out = len(os.listdir(os.path.join(scene, "images")))
    print(f"\nDone in {time.time() - t_all:.0f} s: {n_out} undistorted images in {scene}/images, model in {scene}/sparse/0")


if __name__ == "__main__":
    main()
