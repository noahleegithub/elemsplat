#!/bin/bash
# =============================================================================
#  02_prepare.slurm  --  checks after download + conversion (CPU, ~0.5-2 h)
#
#    BrPCD : inspect_report.txt -> convert to one npz per bridge -> split.json
#    VTTI  : find the shipped COLMAP model, undistort, 1600 px, cameras.bin rescaled,
#            test views (every 8th), EXIF GPS, provisional scale kappa0 (scale0.json)
#  Submit from this folder:  sbatch 02_prepare.slurm        (STEP=brpcd or STEP=vtti for one half)
#  READ ${BRPCD_DIR}/inspect_report.txt afterwards (class mapping, units, variant -> scan groups).
# =============================================================================
#SBATCH --job-name=fp_prepare
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=cpu
#SBATCH --output=logs/%x_%j.out
##SBATCH --account=XXXXXXXX

for _cfg in "$(dirname "${BASH_SOURCE[0]}")/config.sh" "${PWD}/config.sh" "${SLURM_SUBMIT_DIR:-/nonexistent}/config.sh"; do
  # shellcheck disable=SC1090  # config.sh is located at run time
  if [ -f "${_cfg}" ]; then source "${_cfg}"; break; fi
done
[ -n "${HW_SCRIPTS}" ] || { echo "ERROR: config.sh not found -- cd into final_hprc first"; exit 1; }
STEP="${STEP:-all}"
hw_load_modules || exit 1
mkdir -p "${BRPCD_DIR}" "${RESULTS_DIR}/trackB"
NCPU="${SLURM_CPUS_PER_TASK:-8}"

if [ "${STEP}" = "all" ] || [ "${STEP}" = "brpcd" ]; then
  hw_activate_pc || exit 1
  hw_banner "BrPCD: inspect"
  RAW="${BRPCD_RAW}/tree"; [ -d "${RAW}" ] || RAW="${BRPCD_RAW}"
  python "${HW_SCRIPTS}/scripts/brpcd_inspect.py" --raw "${RAW}" --out "${BRPCD_DIR}" || exit 1
  hw_banner "BrPCD: convert (${NCPU} workers)"
  python "${HW_SCRIPTS}/scripts/brpcd_convert.py" --raw "${RAW}" --out "${BRPCD_DIR}" --workers $((NCPU / 2)) || exit 1
  python "${HW_SCRIPTS}/scripts/make_splits.py" --data "${BRPCD_DIR}" || exit 1
fi

if [ "${STEP}" = "all" ] || [ "${STEP}" = "vtti" ]; then
  hw_activate_gs || exit 1                         # pycolmap lives in final_gs
  hw_banner "VTTI: scene from the shipped ${VTTI_MODEL} project"
  python "${HW_SCRIPTS}/scripts/vtti_prepare.py" --vtti "${VTTI_DIR}" --scene "${VTTI_SCENE}" --model "${VTTI_MODEL}" \
         --out "${RESULTS_DIR}/trackB" --threads "${NCPU}" || exit 1
fi
hw_banner "PREPARE DONE.  Check ${BRPCD_DIR}/inspect_report.txt and ${RESULTS_DIR}/trackB/vtti_prepare.json"
