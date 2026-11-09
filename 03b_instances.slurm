#!/bin/bash
# =============================================================================
#  03b_instances.slurm  --  M1 and M2 on one fold per array task (CPU, ~0.5-2 h per fold)
#
#  Tunes M1 (per class) and M2 (grid in experiment.yaml) on the fold's validation bridges, then
#  scores the test bridges. Runs on the predicted classes of MODEL and, with TRUE=1 (default),
#  also on the true classes (the control that isolates the instance step).
#      MODEL=pointnet2 sbatch --array=0-9 03b_instances.slurm
#      MODEL=pointnet TRUE=0 sbatch --array=0-9 03b_instances.slurm
# =============================================================================
#SBATCH --job-name=fp_instances
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# mem: the cable-stayed and suspension clouds are 100-300 M points and DBSCAN's neighbour
# structures scale with them -- fold 1 peaked at 67 GB and was OOM-killed at 64 GB, while the
# six mid-size folds stay under 6 GB. 256 G fits the largest fold on a 500 GB cpu node.
#SBATCH --mem=256G
#SBATCH --partition=cpu
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
##SBATCH --account=XXXXXXXX

for _cfg in "$(dirname "${BASH_SOURCE[0]}")/config.sh" "${PWD}/config.sh" "${SLURM_SUBMIT_DIR:-/nonexistent}/config.sh"; do
  # shellcheck disable=SC1090  # config.sh is located at run time
  if [ -f "${_cfg}" ]; then source "${_cfg}"; break; fi
done
[ -n "${HW_SCRIPTS}" ] || { echo "ERROR: config.sh not found -- cd into final_hprc first"; exit 1; }
hw_load_modules || exit 1
hw_activate_pc || exit 1
MODEL="${MODEL:-pointnet2}"
K="${SLURM_ARRAY_TASK_ID:-${FOLD:-0}}"
NF=$(python -c "import json;print(len(json.load(open('${BRPCD_DIR}/split.json'))['folds']))") || exit 1
[ "${K}" -lt "${NF}" ] || { echo "array index ${K} >= ${NF} folds: nothing to do"; exit 0; }
hw_banner "INSTANCES fold=${K} classes from ${MODEL} (+ true classes: ${TRUE:-1})"
python "${HW_SCRIPTS}/scripts/run_instances.py" --data "${BRPCD_DIR}" --fold "${K}" --classes pred \
       --semseg-model "${MODEL}" --semseg-root "${RESULTS_DIR}/trackA/semseg" --out "${RESULTS_DIR}/trackA/instances" || exit 1
if [ "${TRUE:-1}" = "1" ]; then
  python "${HW_SCRIPTS}/scripts/run_instances.py" --data "${BRPCD_DIR}" --fold "${K}" --classes true \
         --out "${RESULTS_DIR}/trackA/instances" || exit 1
fi
