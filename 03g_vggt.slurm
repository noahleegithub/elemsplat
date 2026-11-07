#!/bin/bash
# =============================================================================
#  03g_vggt.slurm  --  feed-forward baseline: VGGT-1B on the VTTI images (GPU big, ~10 min)
#
#  One forward pass on a uniform subset of the registered images (vggt.max_frames in
#  experiment.yaml), no camera input. The predicted cameras are then aligned to the COLMAP
#  cameras of the same images (similarity transform) so that the points land in the scene frame
#  and 03e_measure.slurm / 03f_merge.slurm treat them like the 3DGS and 2DGS points.
#
#  Submit from this folder:   sbatch 03g_vggt.slurm        (ACES H100; Grace: --gres=gpu:a100:1)
#  Out of memory: lower vggt.max_frames (200 frames need about 40 GB in bf16).
#  Weights: cached by 00_setup_env.sh under ${HF_HOME}; compute nodes run offline.
#  Rerun the alignment only (CPU, no GPU):
#      python scripts/vggt_points.py --scene $VTTI_SCENE --out $RESULTS_DIR/trackB/vggt \
#             --predictions $RESULTS_DIR/trackB/vggt/predictions.npz --points-out $RESULTS_DIR/trackB/points/pts_vggt.npz
# =============================================================================
#SBATCH --job-name=fp_vggt
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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
cfgv () { python -c "import yaml;print(yaml.safe_load(open('${EXPERIMENT_YAML}'))$1)"; }
[ "$(cfgv "['vggt']['enabled']")" = "True" ] || { echo "vggt.enabled is false in experiment.yaml -- nothing to do"; exit 0; }
[ -d "${VTTI_SCENE}/sparse/0" ] || { echo "ERROR: ${VTTI_SCENE} missing -- run 02_prepare.slurm (STEP=vtti)"; exit 1; }
python -c "import vggt" 2>/dev/null || { echo "ERROR: the vggt package is not installed in ${GS_ENV} -- rerun 00_setup_env.sh"; exit 1; }
export HF_HUB_OFFLINE=1          # weights come from the cache filled by 00_setup_env.sh
OUT="${RESULTS_DIR}/trackB/vggt"
mkdir -p "${OUT}" "${RESULTS_DIR}/trackB/points"
hw_banner "VGGT  max_frames=$(cfgv "['vggt']['max_frames']")  conf_percentile=$(cfgv "['vggt']['conf_percentile']")  job=${SLURM_JOB_ID} node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python "${HW_SCRIPTS}/scripts/vggt_points.py" --scene "${VTTI_SCENE}" --out "${OUT}" \
       --points-out "${RESULTS_DIR}/trackB/points/pts_vggt.npz" || { echo "VGGT FAILED"; exit 1; }
hw_banner "VGGT DONE -> ${RESULTS_DIR}/trackB/points/pts_vggt.npz  (alignment residuals in ${OUT}/run_info.json)"
