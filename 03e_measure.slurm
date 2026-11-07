#!/bin/bash
# =============================================================================
#  03e_measure.slurm  --  scale, spans and deck residual for each VTTI point cloud (CPU, ~20 min)
#
#  2DGS (median-depth mesh) runs first and fixes the shared frame + pier boxes
#  (trackB/measure/frame.json); 3DGS, the expected-depth mesh and the VGGT points reuse them.
#  After the first run: open trackB/measure/diag_2dgs.png, compare the boxes with the photos,
#  and if needed copy corrected boxes into configs/vtti_overrides.yaml, then rerun:
#      NEW_FRAME=1 sbatch 03e_measure.slurm
# =============================================================================
#SBATCH --job-name=fp_measure
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --output=logs/%x_%j.out
##SBATCH --account=XXXXXXXX

for _cfg in "$(dirname "${BASH_SOURCE[0]}")/config.sh" "${PWD}/config.sh" "${SLURM_SUBMIT_DIR:-/nonexistent}/config.sh"; do
  # shellcheck disable=SC1090  # config.sh is located at run time
  if [ -f "${_cfg}" ]; then source "${_cfg}"; break; fi
done
[ -n "${HW_SCRIPTS}" ] || { echo "ERROR: config.sh not found -- cd into final_hprc first"; exit 1; }
hw_load_modules || exit 1
hw_activate_pc || exit 1
TB="${RESULTS_DIR}/trackB"
P="${TB}/points"
[ "${NEW_FRAME:-0}" = "1" ] && rm -f "${TB}/measure/frame.json"
run () {   # run <points npz> <method> <label>
  [ -f "$1" ] || { echo "skip $2: $1 missing"; return 0; }
  python "${HW_SCRIPTS}/scripts/measure_vtti.py" --points "$1" --method "$2" --label "$3" \
         --sparse "${VTTI_SCENE}/sparse/0" --trackb "${TB}" || return 1
}
hw_banner "MEASURE on $(hostname)"
run "${P}/pts_2dgs.npz"      2dgs      "2DGS"                  || exit 1
run "${P}/pts_3dgs.npz"      3dgs      "3DGS"                  || exit 1
run "${P}/pts_2dgs_mean.npz" 2dgs_mean "2DGS (expected depth)" || exit 1
run "${P}/pts_vggt.npz"      vggt      "VGGT (feed-forward)"   || exit 1
hw_banner "MEASURE DONE -> ${TB}/measure  (check diag_*.png before trusting the numbers)"
