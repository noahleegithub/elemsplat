#!/bin/bash
# =============================================================================
#  03h_pgsr.slurm  --  PGSR (planar-based GS) reconstruction of the VTTI scene
#
#  Third method beside 3DGS (03c) and 2DGS (03d). Same scene, same 245/35 split,
#  same downstream steps, so the comparison is apples to apples:
#      train -> render test views -> metrics_images.json -> pts_pgsr.npz
#
#  Build notes (see build_pgsr.sh): the repo is at ${HW_ROOT}/repos/PGSR, cloned and
#  patched in place -- CUDA arch 80;90 (A30=sm_80, H100=sm_90) and a local
#  quaternion_to_matrix replacing pytorch3d on the training path. The two extensions
#  are installed into the existing final_gs env because /scratch is at its 250k file
#  cap and a third torch env does not fit.
#
#      sbatch 03h_pgsr.slurm                      # full 30k run
#      ITERS=1500 sbatch 03h_pgsr.slurm           # smoke
# =============================================================================
#SBATCH --job-name=fp_pgsr
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/%x_%A.out
##SBATCH --account=XXXXXXXX

for _cfg in "$(dirname "${BASH_SOURCE[0]}")/config.sh" "${PWD}/config.sh" "${SLURM_SUBMIT_DIR:-/nonexistent}/config.sh"; do
  # shellcheck disable=SC1090  # config.sh is located at run time
  if [ -f "${_cfg}" ]; then source "${_cfg}"; break; fi
done
[ -n "${HW_SCRIPTS}" ] || { echo "ERROR: config.sh not found -- cd into elemsplat first"; exit 1; }
hw_load_modules cuda || exit 1
hw_activate_gs || exit 1                     # PGSR extensions live in final_gs

PGSR="${HW_ROOT}/repos/PGSR"
[ -d "${PGSR}" ] || { echo "ERROR: ${PGSR} missing -- run build_pgsr.sh"; exit 1; }
IT="${ITERS:-30000}"
OUT="${RESULTS_DIR}/trackB/gs_pgsr"
mkdir -p "${OUT}" "${RESULTS_DIR}/trackB/points"
cd "${PGSR}" || exit 1

hw_banner "PGSR train  iterations=${IT} job=${SLURM_JOB_ID} node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch,diff_plane_rasterization,simple_knn._C;print('extensions ok, torch', torch.__version__)" || exit 1

T0=$(date +%s)
python train.py -s "${VTTI_SCENE}" -m "${OUT}" --eval --iterations "${IT}" \
       --test_iterations "${IT}" --save_iterations "${IT}" --quiet \
    || { echo "TRAINING FAILED"; exit 1; }
T1=$(date +%s)

python render.py -m "${OUT}" --skip_train || { echo "RENDER FAILED"; exit 1; }
python metrics.py -m "${OUT}" 2>/dev/null || echo "WARNING: PGSR metrics.py failed (our image_metrics.py is the one reported)"

NG=$(python -c "import sys;sys.path.insert(0,'${HW_SCRIPTS}/scripts');from pcio import ply_vertex_count as c;print(c('${OUT}/point_cloud/iteration_${IT}/point_cloud.ply'))" 2>/dev/null || echo 0)
python - <<EOF
import json, subprocess
json.dump({"method": "pgsr", "train_seconds": $((T1 - T0)), "iterations": ${IT}, "n_gaussians": ${NG:-0},
           "gpu": subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],capture_output=True,text=True).stdout.strip(),
           "repo_commit": subprocess.run(["git","-C","${PGSR}","rev-parse","--short","HEAD"],capture_output=True,text=True).stdout.strip(),
           "job": "${SLURM_JOB_ID}"}, open("${OUT}/run_info.json","w"), indent=2)
EOF

hw_activate_pc || exit 1
python "${HW_SCRIPTS}/scripts/image_metrics.py" --model "${OUT}" --iteration "${IT}" \
       --out "${OUT}/metrics_images.json" || exit 1
python "${HW_SCRIPTS}/scripts/gs_points.py" --gs3d "${OUT}" --iteration "${IT}" \
       --out "${RESULTS_DIR}/trackB/points/pts_pgsr.npz" || exit 1
hw_banner "PGSR DONE  ($(( (T1 - T0) / 60 )) min training, ${NG} Gaussians)"
echo "next: measure_vtti.py --points pts_pgsr.npz --method pgsr --label PGSR"
