#!/bin/bash
# =============================================================================
#  03h_pgsr_grace.slurm  --  PGSR on Grace (A100)
#
#  Same stages as the ACES script (train -> render -> metrics -> points) with two
#  difference Grace forces:
#    * partition gpu with gres gpu:a100:1 (Grace has no H100s)
#    * everything runs in the single final_gs env: Grace has no final_pc env, and
#      image_metrics.py / gs_points.py only need torch + numpy + PIL + plyfile.
#
#      CLUSTER=grace sbatch 03h_pgsr_grace.slurm
#      CLUSTER=grace ITERS=2000 sbatch 03h_pgsr_grace.slurm     # smoke
# =============================================================================
#SBATCH --job-name=pgsr
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=132699935111
#SBATCH --output=/scratch/user/noahvlee/ECEN689/logs/%x_%A.out

export CLUSTER=grace
for _cfg in "$(dirname "${BASH_SOURCE[0]}")/config.sh" "${PWD}/config.sh" "${SLURM_SUBMIT_DIR:-/nonexistent}/config.sh"; do
  # shellcheck disable=SC1090  # config.sh is located at run time
  if [ -f "${_cfg}" ]; then source "${_cfg}"; break; fi
done
[ -n "${HW_SCRIPTS}" ] || { echo "ERROR: config.sh not found -- cd into elemsplat first"; exit 1; }
hw_load_modules cuda || exit 1
hw_activate_gs || exit 1

PGSR="${HW_ROOT}/repos/PGSR"
[ -d "${PGSR}" ] || { echo "ERROR: ${PGSR} missing"; exit 1; }
IT="${ITERS:-30000}"
OUT="${RESULTS_DIR}/trackB/gs_pgsr${OUT_SUFFIX:-}"
EXTRA="${PGSR_EXTRA:-}"
mkdir -p "${OUT}" "${RESULTS_DIR}/trackB/points" "${HW_ROOT}/logs"
cd "${PGSR}" || exit 1

hw_banner "PGSR train (Grace)  iterations=${IT} job=${SLURM_JOB_ID} node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch, diff_plane_rasterization, simple_knn._C;print('extensions ok, torch', torch.__version__)" || exit 1
echo "extra training args: ${EXTRA:-<none: upstream defaults>}"

T0=$(date +%s)
# shellcheck disable=SC2086  # EXTRA is a deliberate argument list
python train.py -s "${VTTI_SCENE}" -m "${OUT}" --eval --iterations "${IT}" \
       --test_iterations "${IT}" --save_iterations "${IT}" --quiet ${EXTRA} \
    || { echo "TRAINING FAILED"; exit 1; }
T1=$(date +%s)
echo "training took $(( (T1 - T0) / 60 )) min"

python render.py -m "${OUT}" --skip_train || { echo "RENDER FAILED"; exit 1; }
python metrics.py -m "${OUT}" 2>/dev/null || echo "WARNING: PGSR metrics.py failed (our image_metrics.py is the one reported)"

NG=$(python -c "import sys;sys.path.insert(0,'${HW_SCRIPTS}/scripts');from pcio import ply_vertex_count as c;print(c('${OUT}/point_cloud/iteration_${IT}/point_cloud.ply'))" 2>/dev/null || echo 0)
python - <<EOF
import json, subprocess
json.dump({"method": "pgsr", "config": "${OUT_SUFFIX:-defaults}", "extra_args": "${EXTRA}", "train_seconds": $((T1 - T0)), "iterations": ${IT}, "n_gaussians": ${NG:-0},
           "gpu": subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),
           "repo_commit": subprocess.run(["git","-C","${PGSR}","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip(),
           "cluster": "grace", "job": "${SLURM_JOB_ID}"}, open("${OUT}/run_info.json","w"), indent=2)
EOF

# same env: these two scripts need only torch/numpy/PIL/plyfile
python "${HW_SCRIPTS}/scripts/image_metrics.py" --model "${OUT}" --iteration "${IT}" \
       --out "${OUT}/metrics_images.json" || exit 1
python "${HW_SCRIPTS}/scripts/gs_points.py" --gs3d "${OUT}" --iteration "${IT}" \
       --out "${RESULTS_DIR}/trackB/points/pts_pgsr${OUT_SUFFIX:-}.npz" || exit 1
hw_banner "PGSR DONE on Grace  ($(( (T1 - T0) / 60 )) min training, ${NG} Gaussians)"
