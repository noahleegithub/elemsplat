#!/bin/bash
# =============================================================================
#  00b_smoke_test.slurm  --  synthetic end-to-end test of the analysis code (CPU, ~15-30 min)
#
#  Submit from this folder:   sbatch 00b_smoke_test.slurm      (ACES: CPU partition)
#  Grace:                     sbatch --partition= 00b_smoke_test.slurm   (or delete the partition line)
#  Output: ${SMOKE_DIR} (tables and figures stamped "SYNTHETIC SMOKE TEST - NOT RESULTS")
# =============================================================================
#SBATCH --job-name=fp_smoke
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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
hw_banner "SMOKE TEST on $(hostname) -> ${SMOKE_DIR}"
rm -rf "${SMOKE_DIR}"
python "${HW_SCRIPTS}/scripts/smoke_test.py" --out "${SMOKE_DIR}" || exit 1
hw_banner "SMOKE TEST PASSED.  Figures: ${SMOKE_DIR}/paper_assets/figures   Next: bash 01_download.sh"
