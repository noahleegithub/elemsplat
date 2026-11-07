#!/bin/bash
# =============================================================================
#  04_eval_pack.sh  --  summary, LaTeX tables, figures and one small tarball to download
#
#  Usage:   bash 04_eval_pack.sh          (login node is fine: a few minutes, no GPU)
#     or:   sbatch --partition=cpu --time=00:30:00 --mem=16G 04_eval_pack.sh
#  Output:  ${RESULTS_DIR}/paper_assets/{tables,figures}  ->  copy into paper/results and
#           paper/figures/results (or unpack final_results_pack.tar.gz on your PC)
# =============================================================================
#SBATCH --job-name=fp_pack
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
set -o pipefail
for _cfg in "$(dirname "${BASH_SOURCE[0]}")/config.sh" "${PWD}/config.sh" "${SLURM_SUBMIT_DIR:-/nonexistent}/config.sh"; do
  # shellcheck disable=SC1090  # config.sh is located at run time
  if [ -f "${_cfg}" ]; then source "${_cfg}"; break; fi
done
[ -n "${HW_SCRIPTS}" ] || { echo "ERROR: config.sh not found -- cd into final_hprc first"; exit 1; }
hw_load_modules || exit 1
hw_activate_pc || exit 1
PA="${RESULTS_DIR}/paper_assets"
hw_banner "AGGREGATE -> ${RESULTS_DIR}/summary"
python "${HW_SCRIPTS}/scripts/aggregate.py" --results "${RESULTS_DIR}" --data "${BRPCD_DIR}" || exit 1
python "${HW_SCRIPTS}/scripts/make_tables.py" --summary "${RESULTS_DIR}/summary/summary.json" --out "${PA}/tables" || exit 1
python "${HW_SCRIPTS}/scripts/make_figures.py" --results "${RESULTS_DIR}" --data "${BRPCD_DIR}" --out "${PA}/figures" || exit 1
mkdir -p "${PA}/diagnostics"
cp -f "${RESULTS_DIR}"/trackB/measure/diag_*.png "${PA}/diagnostics/" 2>/dev/null || true
cp -f "${BRPCD_DIR}/inspect_report.txt" "${BRPCD_DIR}/index.csv" "${PA}/diagnostics/" 2>/dev/null || true
cp -f "${RESULTS_DIR}/env_info.txt" "${PA}/" 2>/dev/null || true
hw_banner "PACK -> ${HW_ROOT}/final_results_pack.tar.gz"
cd "${RESULTS_DIR}/.." || exit 1
R=$(basename "${RESULTS_DIR}")
# shellcheck disable=SC2046   # the file list is split on purpose (paths on $SCRATCH have no spaces)
tar -czf "${HW_ROOT}/final_results_pack.tar.gz" \
    "${R}/paper_assets" "${R}/summary" \
    $(cd "${RESULTS_DIR}/.." && ls -d "${R}"/trackA/semseg/*/fold_*/metrics_semseg.json "${R}"/trackA/semseg/*/fold_*/train_log.csv \
          "${R}"/trackA/instances/*/fold_*/*.json "${R}"/trackB/*.json "${R}"/trackB/*.txt "${R}"/trackB/*/metrics_images.json \
          "${R}"/trackB/*/run_info.json "${R}"/trackB/measure/*.json "${R}"/trackB/measure/*.yaml "${R}"/rq3/*.json \
          "${R}"/rq3/*.ply 2>/dev/null) || exit 1
ls -lh "${HW_ROOT}/final_results_pack.tar.gz"
hw_banner "DONE.  Download final_results_pack.tar.gz (portal Files app) and copy paper_assets/ into the paper folder."
