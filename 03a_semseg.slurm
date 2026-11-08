#!/bin/bash
# =============================================================================
#  03a_semseg.slurm  --  semantic network, one fold per array task (GPU small: ACES A30)
#
#  Array index k < n_folds -> fold k (leave-one-bridge-out); k == n_folds -> "all" (network for RQ3);
#  larger indices exit at once, so --array=0-10 fits 10 real scans.
#      MODEL=pointnet2 sbatch --array=0-10 03a_semseg.slurm
#      MODEL=pointnet  sbatch --array=0-10 03a_semseg.slurm
#  Expected per task on an A30: PointNet ~20-30 min, PointNet++ ~1-1.5 h (20k iterations).
#  Measured on ACES: 43 min to 3 h for most folds; the heaviest fold needed >6 h and hit the
#  walltime, so the limit is 12 h (SUs are charged on walltime used, not requested).
# =============================================================================
#SBATCH --job-name=fp_semseg
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --array=0-10
#SBATCH --output=logs/%x_%A_%a.out
##SBATCH --account=XXXXXXXX

for _cfg in "$(dirname "${BASH_SOURCE[0]}")/config.sh" "${PWD}/config.sh" "${SLURM_SUBMIT_DIR:-/nonexistent}/config.sh"; do
  # shellcheck disable=SC1090  # config.sh is located at run time
  if [ -f "${_cfg}" ]; then source "${_cfg}"; break; fi
done
[ -n "${HW_SCRIPTS}" ] || { echo "ERROR: config.sh not found -- cd into final_hprc first"; exit 1; }
hw_load_modules cuda || exit 1
hw_activate_pc || exit 1
MODEL="${MODEL:-pointnet2}"
K="${SLURM_ARRAY_TASK_ID:-${FOLD:-0}}"
NF=$(python -c "import json;print(len(json.load(open('${BRPCD_DIR}/split.json'))['folds']))") || exit 1
if [ "${K}" -lt "${NF}" ]; then FOLD="${K}"; elif [ "${K}" -eq "${NF}" ]; then FOLD="all"; else
  echo "array index ${K} > ${NF} folds: nothing to do"; exit 0; fi
hw_banner "SEMSEG model=${MODEL} fold=${FOLD} job=${SLURM_JOB_ID} node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "${HW_SCRIPTS}/scripts/semseg_train.py" --data "${BRPCD_DIR}" --model "${MODEL}" --fold "${FOLD}" \
       --out "${RESULTS_DIR}/trackA/semseg" --workers $(( ${SLURM_CPUS_PER_TASK:-8} - 2 )) || exit 1
