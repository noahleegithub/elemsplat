#!/bin/bash
# =============================================================================
#  03d_gs2d.slurm  --  2DGS on VTTI: train, render test views, TSDF meshes, metrics, point clouds
#
#  Submit from this folder:   sbatch 03d_gs2d.slurm          (ACES H100; Grace: --gres=gpu:a100:1)
#  TSDF voxel / truncation / depth cut are given in metres in experiment.yaml and converted to SfM
#  units with the provisional scale in trackB/scale0.json. One mesh per depth ratio
#  (1 = median depth, 0 = expected depth). Expected on an H100: ~2-3 h in total.
# =============================================================================
#SBATCH --job-name=fp_gs2d
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/%x_%j.out
##SBATCH --account=XXXXXXXX

for _cfg in "$(dirname "${BASH_SOURCE[0]}")/config.sh" "${PWD}/config.sh" "${SLURM_SUBMIT_DIR:-/nonexistent}/config.sh"; do
  # shellcheck disable=SC1090  # config.sh is located at run time
  if [ -f "${_cfg}" ]; then source "${_cfg}"; break; fi
done
[ -n "${HW_SCRIPTS}" ] || { echo "ERROR: config.sh not found -- cd into final_hprc first"; exit 1; }
hw_load_modules cuda || exit 1
hw_activate_gs || exit 1
OUT="${RESULTS_DIR}/trackB/gs2d"
cfgv () { python -c "import yaml;print(yaml.safe_load(open('${EXPERIMENT_YAML}'))$1)"; }
IT=$(cfgv "['gs2d']['iterations']"); LAM=$(cfgv "['gs2d']['lambda_dssim']")
LD=$(cfgv "['gs2d']['lambda_dist']"); LN=$(cfgv "['gs2d']['lambda_normal']"); TDR=$(cfgv "['gs2d']['train_depth_ratio']")
DRS=$(cfgv "['gs2d']['mesh_depth_ratios']" | tr -d '[],'); NCL=$(cfgv "['gs2d']['num_cluster']")
S0="${RESULTS_DIR}/trackB/scale0.json"
[ -f "${S0}" ] || { echo "ERROR: ${S0} missing -- run 02_prepare.slurm (STEP=vtti)"; exit 1; }
VOX=$(python -c "import json;print(json.load(open('${S0}'))['tsdf_voxel_sfm'])")
TRC=$(python -c "import json;print(json.load(open('${S0}'))['sdf_trunc_sfm'])")
DTR=$(python -c "import json;print(json.load(open('${S0}'))['depth_trunc_sfm'])")
python -c "import diff_surfel_rasterization, simple_knn" 2>/dev/null || {
  echo "2DGS extension not importable here -> rebuilding for this GPU"; unset TORCH_CUDA_ARCH_LIST
  ( cd "${GS2D_REPO}" && python -m pip install --no-build-isolation ./submodules/diff-surfel-rasterization ) \
      > "${HW_SCRIPTS}/logs/rebuild_gs2d_${SLURM_JOB_ID}.log" 2>&1 || exit 1; }
cd "${GS2D_REPO}" || exit 1
# flag names differ between 2DGS versions: check before training
FLAGS=$(python train.py --help 2>&1)
DFLAG="--lambda_dist"; echo "${FLAGS}" | grep -q -- "--lambda_dist" || DFLAG="--lambda_distortion"
hw_banner "2DGS train  iterations=${IT} ${DFLAG}=${LD} lambda_normal=${LN} depth_ratio=${TDR}  job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mkdir -p "${OUT}"
T0=$(date +%s)
python train.py -s "${VTTI_SCENE}" -m "${OUT}" --eval --iterations "${IT}" --lambda_dssim "${LAM}" \
       "${DFLAG}" "${LD}" --lambda_normal "${LN}" --depth_ratio "${TDR}" \
       --test_iterations 7000 "${IT}" --save_iterations "${IT}" || { echo "TRAINING FAILED"; exit 1; }
T1=$(date +%s)
hw_banner "2DGS render test views"
python render.py -m "${OUT}" -s "${VTTI_SCENE}" --iteration "${IT}" --skip_train --skip_mesh --depth_ratio "${TDR}" \
  || { echo "RENDER FAILED"; exit 1; }
for DR in ${DRS}; do
  hw_banner "2DGS TSDF mesh  depth_ratio=${DR}  voxel=${VOX} trunc=${TRC} depth_trunc=${DTR} (SfM units)"
  python render.py -m "${OUT}" -s "${VTTI_SCENE}" --iteration "${IT}" --skip_train --skip_test --depth_ratio "${DR}" \
         --voxel_size "${VOX}" --sdf_trunc "${TRC}" --depth_trunc "${DTR}" --num_cluster "${NCL}" || { echo "MESH FAILED"; exit 1; }
  MD="${OUT}/train/ours_${IT}"
  for f in fuse.ply fuse_post.ply; do [ -f "${MD}/${f}" ] && mv -f "${MD}/${f}" "${MD}/${f%.ply}_dr${DR}.ply"; done
done
T2=$(date +%s)
NG=$(python -c "import sys;sys.path.insert(0,'${HW_SCRIPTS}/scripts');from pcio import ply_vertex_count as c;print(c('${OUT}/point_cloud/iteration_${IT}/point_cloud.ply'))")
python - <<EOF
import json, subprocess
json.dump({"method": "2dgs", "train_seconds": $((T1 - T0)), "mesh_seconds": $((T2 - T1)), "iterations": ${IT},
           "n_gaussians": ${NG:-0}, "distortion_flag": "${DFLAG}",
           "gpu": subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),
           "repo_commit": subprocess.run(["git","-C","${GS2D_REPO}","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip(),
           "job": "${SLURM_JOB_ID}"}, open("${OUT}/run_info.json","w"), indent=2)
EOF
hw_activate_pc || exit 1
python "${HW_SCRIPTS}/scripts/image_metrics.py" --model "${OUT}" --iteration "${IT}" --out "${OUT}/metrics_images.json" || exit 1
mkdir -p "${RESULTS_DIR}/trackB/points"
MD="${OUT}/train/ours_${IT}"
python "${HW_SCRIPTS}/scripts/gs_points.py" --mesh "${MD}/fuse_post_dr1.0.ply" --out "${RESULTS_DIR}/trackB/points/pts_2dgs.npz" \
  || python "${HW_SCRIPTS}/scripts/gs_points.py" --mesh "${MD}/fuse_dr1.0.ply" --out "${RESULTS_DIR}/trackB/points/pts_2dgs.npz" || exit 1
[ -f "${MD}/fuse_post_dr0.0.ply" ] && python "${HW_SCRIPTS}/scripts/gs_points.py" --mesh "${MD}/fuse_post_dr0.0.ply" \
       --out "${RESULTS_DIR}/trackB/points/pts_2dgs_mean.npz"
hw_banner "2DGS DONE  ($(( (T1 - T0) / 60 )) min training, $(( (T2 - T1) / 60 )) min render + meshes, ${NG} Gaussians)"
