#!/bin/bash
# 90b_gate_check.sh -- pre-flight gate: verify the prepared data before spending any GPU time
#
#   bash 90b_gate_check.sh   (login node; exits 0 only when every check passes)
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
source ./config.sh || exit 1
hw_load_modules cuda || exit 1
hw_activate_gs || exit 1
python - <<'PYEOF'
import json, os
import pycolmap

scene = os.environ["VTTI_SCENE"]
res = os.environ["RESULTS_DIR"]
ok = True

rec = pycolmap.Reconstruction(os.path.join(scene, "sparse", "0"))
disk = set(os.listdir(os.path.join(scene, "images")))
names = {im.name for im in rec.images.values()}
print(f"scene: {len(rec.images)} images, {len(disk)} files, missing {len(names - disk)}, "
      f"{len(rec.points3D)} points3D")
ok &= len(names - disk) == 0 and len(names) == len(disk)

tv = [l.strip() for l in open(os.path.join(res, "trackB", "test_views.txt")) if l.strip()]
missing_tv = [n for n in tv if n not in disk]
print(f"test views: {len(tv)} listed, {len(missing_tv)} missing from disk")
ok &= not missing_tv

p = json.load(open(os.path.join(res, "trackB", "vtti_prepare.json")))
print(f"prepare: n_registered={p['n_registered']} n_found={p['n_found']} "
      f"scene_images={p.get('n_images_scene')} train={p['n_train']} test={p['n_test']} "
      f"reproj={p['mean_reproj_px']:.3f}px kappa0={p['provisional']['kappa0']:.4f}")
ok &= p.get("n_images_scene") == len(rec.images)

s0 = json.load(open(os.path.join(res, "trackB", "scale0.json")))
print(f"scale0: kappa0={s0['kappa0']:.4f} tsdf_voxel={s0['tsdf_voxel_sfm']:.5f} SfM units")

b = os.environ["BRPCD_DIR"]
cs = json.load(open(os.path.join(b, "convert_summary.json")))
nz = len([f for f in os.listdir(os.path.join(b, "clouds")) if f.endswith(".npz")])
sp = json.load(open(os.path.join(b, "split.json")))
print(f"brpcd: voxel={cs['voxel']} m, {nz} clouds, {len(sp['folds'])} folds, "
      f"all={len(sp['all'])}, unit_scale={__import__('yaml').safe_load(open(os.path.join(os.environ['HW_SCRIPTS'],'configs','experiment.yaml')))['brpcd']['unit_scale']}")
ok &= nz == 98 and len(sp["folds"]) == 10

print("\nGATE:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
PYEOF
echo "EXIT=$?"
