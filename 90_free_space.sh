#!/bin/bash
# =============================================================================
#  90_free_space.sh  --  free $SCRATCH space / file count after the results are packed
#
#  bash 90_free_space.sh zips      # downloaded archives (re-download is possible)
#  bash 90_free_space.sh models    # 3DGS/2DGS checkpoints and renders, VGGT raw predictions (keeps metrics + run_info)
#  bash 90_free_space.sh envs      # both conda envs (rerun 00_setup_env.sh to restore)
# =============================================================================
for _cfg in "$(dirname "${BASH_SOURCE[0]}")/config.sh" "${PWD}/config.sh"; do
  # shellcheck disable=SC1090  # config.sh is located at run time
  if [ -f "${_cfg}" ]; then source "${_cfg}"; break; fi
done
[ -n "${HW_SCRIPTS}" ] || { echo "ERROR: config.sh not found"; exit 1; }
case "${1:-}" in
  zips)   du -sh "${VTTI_RAW}" "${BRPCD_RAW}/archives" 2>/dev/null; rm -rf "${VTTI_RAW}" "${BRPCD_RAW}/archives" ;;
  models) for m in gs3d gs2d; do d="${RESULTS_DIR}/trackB/${m}"; [ -d "$d" ] || continue
            find "$d" -mindepth 1 -maxdepth 1 ! -name '*.json' -exec rm -rf {} +; done
          rm -f "${RESULTS_DIR}/trackB/vggt/predictions.npz" ;;
  envs)   rm -rf "${PC_ENV}" "${GS_ENV}" ;;
  *)      echo "usage: bash 90_free_space.sh zips|models|envs"; exit 1 ;;
esac
command -v showquota >/dev/null 2>&1 && showquota
