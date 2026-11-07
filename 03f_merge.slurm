#!/bin/bash
# =============================================================================
#  03f_merge.slurm  --  RQ3: Track A network + M1/M2 on the scaled 2DGS points (GPU small, ~15 min)
#                       METHOD=vggt sbatch 03f_merge.slurm  repeats it on the VGGT points (-> rq3_vggt)
#
#  Needs: 03a fold "all" of merge.semseg_model, 03b (tuned parameters), 03e (aligned_2dgs.npz).
#  Fill configs/vtti_overrides.yaml: girder_count_ref after counting the girders on the photos.
# =============================================================================
#SBATCH --job-name=fp_merge
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --output=logs/%x_%j.out
##SBATCH --account=XXXXXXXX

for _cfg in "$(dirname "${BASH_SOURCE[0]}")/config.sh" "${PWD}/config.sh" "${SLURM_SUBMIT_DIR:-/nonexistent}/config.sh"; do
  # shellcheck disable=SC1090  # config.sh is located at run time
  if [ -f "${_cfg}" ]; then source "${_cfg}"; break; fi
done
[ -n "${HW_SCRIPTS}" ] || { echo "ERROR: config.sh not found -- cd into final_hprc first"; exit 1; }
hw_load_modules cuda || exit 1
hw_activate_pc || exit 1
METHOD="${METHOD:-2dgs}"                                   # 2dgs (paper) | 3dgs | 2dgs_mean | vggt
OUT="${RESULTS_DIR}/rq3"; [ "${METHOD}" = "2dgs" ] || OUT="${RESULTS_DIR}/rq3_${METHOD}"
[ -f "${RESULTS_DIR}/trackB/measure/aligned_${METHOD}.npz" ] || { echo "aligned_${METHOD}.npz missing -- run 03e first (or the ${METHOD} job failed)"; exit 1; }
hw_banner "MERGE (RQ3) on ${METHOD} points, $(hostname)"
python "${HW_SCRIPTS}/scripts/merge_vtti.py" --trackb "${RESULTS_DIR}/trackB" --method "${METHOD}" --brpcd "${BRPCD_DIR}" \
       --semseg-root "${RESULTS_DIR}/trackA/semseg" --instances-root "${RESULTS_DIR}/trackA/instances" \
       --out "${OUT}" || exit 1
hw_banner "MERGE DONE -> ${OUT}  (vtti_m2_instances.ply opens in CloudCompare)"
