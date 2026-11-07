#!/bin/bash
# =============================================================================
#  03c_gs3d.slurm  --  3DGS on VTTI: train (held-out every 8th view), render, metrics, point cloud
#
#  Submit from this folder:   sbatch 03c_gs3d.slurm          (ACES H100; Grace: --gres=gpu:a100:1)
#  Expected on an H100: ~1-2 h for 30k iterations at 1600 px (depends on the image count).
#  Out of memory: set gs3d.data_device: cpu in experiment.yaml, or vtti.long_side_px: 1200 and rerun 02.
# =============================================================================
#SBATCH --job-name=fp_gs3d
#SBATCH --time=08:00:00
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
OUT="${RESULTS_DIR}/trackB/gs3d"
cfgv () { python -c "import yaml;print(yaml.safe_load(open('${EXPERIMENT_YAML}'))$1)"; }
IT=$(cfgv "['gs3d']['iterations']"); LAM=$(cfgv "['gs3d']['lambda_dssim']")
DGT=$(cfgv "['gs3d']['densify_grad_threshold']"); DUI=$(cfgv "['gs3d']['densify_until_iter']")
DDEV=$(cfgv "['gs3d']['data_device']")
[ -d "${VTTI_SCENE}/sparse/0" ] || { echo "ERROR: ${VTTI_SCENE} missing -- run 02_prepare.slurm (STEP=vtti)"; exit 1; }
python -c "import diff_gaussian_rasterization, simple_knn, fused_ssim" 2>/dev/null || {
  echo "3DGS extensions not importable here -> rebuilding for this GPU"; unset TORCH_CUDA_ARCH_LIST
  ( cd "${GS3D_REPO}" && python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization \
      ./submodules/simple-knn ./submodules/fused-ssim ) > "${HW_SCRIPTS}/logs/rebuild_gs3d_${SLURM_JOB_ID}.log" 2>&1 || exit 1; }
hw_banner "3DGS train  iterations=${IT} lambda=${LAM} densify_grad=${DGT}  job=${SLURM_JOB_ID} node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mkdir -p "${OUT}"
cd "${GS3D_REPO}" || exit 1
T0=$(date +%s)
python train.py -s "${VTTI_SCENE}" -m "${OUT}" --eval --iterations "${IT}" --lambda_dssim "${LAM}" \
       --densify_grad_threshold "${DGT}" --densify_until_iter "${DUI}" --data_device "${DDEV}" \
       --test_iterations 7000 "${IT}" --save_iterations 7000 "${IT}" --disable_viewer || { echo "TRAINING FAILED"; exit 1; }
T1=$(date +%s)
python render.py -m "${OUT}" --iteration "${IT}" --skip_train || { echo "RENDER FAILED"; exit 1; }
python metrics.py -m "${OUT}" || echo "WARNING: official metrics.py failed (image_metrics.py below is the one used in the paper)"
NG=$(python -c "import sys;sys.path.insert(0,'${HW_SCRIPTS}/scripts');from pcio import ply_vertex_count as c;print(c('${OUT}/point_cloud/iteration_${IT}/point_cloud.ply'))")
python - <<EOF
import json, subprocess
json.dump({"method": "3dgs", "train_seconds": $((T1 - T0)), "iterations": ${IT}, "n_gaussians": ${NG:-0},
           "gpu": subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),
           "repo_commit": subprocess.run(["git","-C","${GS3D_REPO}","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip(),
           "job": "${SLURM_JOB_ID}"}, open("${OUT}/run_info.json","w"), indent=2)
EOF
hw_activate_pc || exit 1
python "${HW_SCRIPTS}/scripts/image_metrics.py" --model "${OUT}" --iteration "${IT}" --out "${OUT}/metrics_images.json" || exit 1
mkdir -p "${RESULTS_DIR}/trackB/points"
python "${HW_SCRIPTS}/scripts/gs_points.py" --gs3d "${OUT}" --iteration "${IT}" --out "${RESULTS_DIR}/trackB/points/pts_3dgs.npz" || exit 1
hw_banner "3DGS DONE  ($(( (T1 - T0) / 60 )) min training, ${NG} Gaussians)"
