#!/bin/bash
# =============================================================================
#  config.sh  --  ONE place for every cluster / path / software setting
#                 (ECEN 689 final project: BrPCD member instances + VTTI 3DGS/2DGS)
#
#  Every other script does:   source "<this folder>/config.sh"
#  Analysis settings (voxel size, splits, M1/M2, 3DGS/2DGS, measurement) live in
#  configs/experiment.yaml, not here.
#
#  Default target: TAMU HPRC ACES  (H100 80 GB for 3DGS/2DGS, A30 24 GB for point clouds).
#  Grace works too:  CLUSTER=grace bash run_all.sh   (A100 40 GB for every GPU job).
# =============================================================================

# -----------------------------------------------------------------------------
# 0. Cluster switch (only run_all.sh and 00_setup_env.sh read it)
# -----------------------------------------------------------------------------
export CLUSTER="${CLUSTER:-aces}"                 # aces | grace
case "${CLUSTER}" in
  aces)
    export GPU_BIG="h100"; export GPU_SMALL="a30"          # 3DGS/2DGS | point-cloud nets
    export CPU_PARTITION="cpu"; export GPU_PARTITION="gpu"
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"   # A30 = 8.0, H100 = 9.0
    ;;
  grace)
    export GPU_BIG="a100"; export GPU_SMALL="a100"
    export CPU_PARTITION="medium"; export GPU_PARTITION="gpu"      # Grace: short/medium/long (check: sinfo -s)
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
    ;;
  *) echo "[config] unknown CLUSTER=${CLUSTER} (use aces or grace)"; return 1 2>/dev/null || exit 1 ;;
esac
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"         # set if `myproject` lists several allocations

# -----------------------------------------------------------------------------
# 1. Locations.  HW_ROOT = the folder that CONTAINS final_hprc/ (e.g. $SCRATCH/ECEN689).
#    Everything lives on $SCRATCH: $HOME has a 10 GB / 10k-file quota.
# -----------------------------------------------------------------------------
HW_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HW_SCRIPTS
case "$(basename "${HW_SCRIPTS}")" in
  final_hprc|elemsplat) HW_ROOT="$(dirname "${HW_SCRIPTS}")" ;;
  *)          HW_ROOT="${HW_SCRIPTS}" ;;
esac
export HW_ROOT
# envs, repos and results are large: keep them off $HOME (HPRC gives $HOME 10 GB / 10k files)
case "${HW_ROOT}" in
  "${HOME}"/*) echo "[config] WARNING: HW_ROOT=${HW_ROOT} is under \$HOME -- envs and data will hit the" \
                    "home quota. Clone to \$SCRATCH/ECEN689/elemsplat instead." >&2 ;;
esac
# export HW_ROOT="${SCRATCH}/ECEN689"                # <- uncomment to force a location
export ENV_ROOT="${HW_ROOT}/envs"
export PC_ENV="${ENV_ROOT}/final_pc"               # point clouds: PyTorch nets, M1/M2, metrics, figures
export GS_ENV="${ENV_ROOT}/final_gs"               # 3DGS + 2DGS CUDA extensions, pycolmap, open3d
export REPO_ROOT="${HW_ROOT}/repos"
export GS3D_REPO="${REPO_ROOT}/gaussian-splatting"
export GS2D_REPO="${REPO_ROOT}/2d-gaussian-splatting"
export GS3D_COMMIT="54c035f"                       # same commit as HW1 (graphdeco-inria/gaussian-splatting)
export GS2D_COMMIT=""                              # empty = HEAD at setup time (recorded in env_info.txt)
export VGGT_REPO="${REPO_ROOT}/vggt"               # feed-forward baseline (facebookresearch/vggt)
export VGGT_COMMIT="a288dd0"                       # 2026-05-18 (memory fix: 2-3x more frames per pass)
export VGGT_HF_ID="facebook/VGGT-1B"               # weights (non-commercial licence; course use)

export DATA_ROOT="${HW_ROOT}/final_data"
export BRPCD_RAW="${DATA_ROOT}/brpcd_raw"          # Google Drive download, extracted
export BRPCD_DIR="${DATA_ROOT}/brpcd"              # converted clouds (*.npz) + index.csv + split.json
export VTTI_RAW="${DATA_ROOT}/vtti_raw"            # figshare zips
export VTTI_DIR="${DATA_ROOT}/vtti"                # extracted dataset
export VTTI_SCENE="${DATA_ROOT}/vtti_scene"        # 3DGS/2DGS scene: images/ (undistorted, 1600 px) + sparse/0
export RESULTS_DIR="${HW_ROOT}/final_results"
export SMOKE_DIR="${HW_ROOT}/final_smoke"          # synthetic end-to-end test (CPU, minutes)
export EXPERIMENT_YAML="${HW_SCRIPTS}/configs/experiment.yaml"

# -----------------------------------------------------------------------------
# 2. Data sources
#    BrPCD: Google Drive folder linked from github.com/wxiong12/SEU-Digital-Bridge
#    VTTI : Virginia Tech Data Repository, doi 10.7294/29901590.v2 (figshare file ids, md5)
# -----------------------------------------------------------------------------
export BRPCD_GDRIVE_FOLDER="1hoV1DeMOxDDwcQd7nwWBLmElzHuWlXk8"
# "name|file_id|md5|needed(1/0)"  -- needed=0 files are skipped unless VTTI_ALL=1
export VTTI_FILES="
Full Resolution Reconstruction Images.zip|59291387|e2269e7392988e6ec9ee8d5c1cee9998|1
FRO.zip|59291393|d2e6fe8812b9364057dc615bd2e201fe|1
Image_Master_List.xlsx|59183807|f60d32810bdbb121b36ba21f685240b0|1
COLMAP_Models_Options.xlsx|59183789|e2c39db39ff5c072730d565213b3f4f6|1
COLMAP_instructions.docx|57479815|ddd7bdddf0b9ab816c5fb7016ac1bbe8|1
README.rtf|59357165|7c88c8670b9809a76d367456ae234c64|1
UAV Query Images.zip|59291378|695098028ec01362f5ca842b182dd130|0
Cell Phone Query Images.zip|59291372|ef3f3f1720e20fb9d86a87782a172c15|0
Reduced Resolution Reconstruction Images.zip|59291375|9fb6ea266854e3df18937df6fbe34047|0
FRR.zip|59291396|399476edadd7cc9f8de3080e233e77fb|0
RRO.zip|59291399|cbe5b765c0916bd73b9ec874306c7e90|0
RRR.zip|59325554|76b5de52e53cb2ed2b183d4bb6c3a63f|0
"
export VTTI_MODEL="${VTTI_MODEL:-FRO}"             # COLMAP project used for poses (FRO = full res, OpenCV)

# -----------------------------------------------------------------------------
# 3. Software stack (same pairing as HW1: CUDA 12.1 <-> torch 2.3.1+cu121, numpy < 2)
#    The first module that loads wins; check names with  module spider CUDA / GCC.
# -----------------------------------------------------------------------------
export ANACONDA_MODULES="Anaconda3/2024.02-1 Anaconda3"
export GCC_MODULES="GCC/11.3.0 GCC/12.2.0 GCC/12.3.0"
export CUDA_MODULES="CUDA/12.1.1 CUDA/12.1.0 CUDA/12.2.0 CUDA/12.4.0"
export PYTHON_VERSION="3.10"
export TORCH_VERSION="2.3.1"
export TORCHVISION_VERSION="0.18.1"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

# -----------------------------------------------------------------------------
# 4. Caches: OUT of $HOME and identical in every script (compute nodes reuse them)
# -----------------------------------------------------------------------------
export CONDA_PKGS_DIRS="${HW_ROOT}/.cache/conda_pkgs"
export PIP_CACHE_DIR="${HW_ROOT}/.cache/pip"
export TORCH_HOME="${HW_ROOT}/.cache/torch"
export TORCH_EXTENSIONS_DIR="${HW_ROOT}/.cache/torch_extensions"
export HF_HOME="${HW_ROOT}/.cache/huggingface"      # VGGT weights (about 5 GB), filled by 00_setup_env.sh
export MPLCONFIGDIR="${HW_ROOT}/.cache/matplotlib"
export QT_QPA_PLATFORM=offscreen
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# -----------------------------------------------------------------------------
# 5. Helpers
# -----------------------------------------------------------------------------
_hw_load_first () {        # _hw_load_first "A/1 A/2 ..."  -> loads the first module that exists
  local m
  for m in $1; do
    if module load "${m}" >/dev/null 2>&1; then
      # Lmod module load affects the current shell environment; print name on fd 3 or stdout
      echo "${m}"
      return 0
    fi
  done
  return 1
}

hw_load_modules () {       # hw_load_modules [cuda]   (cuda = also GCC + CUDA, for building/running CUDA code)
  module purge >/dev/null 2>&1
  local a g c
  for m in ${ANACONDA_MODULES}; do
    if module load "${m}" >/dev/null 2>&1; then a="${m}"; break; fi
  done
  [ -n "${a}" ] || { echo "[config] no Anaconda3 module -- run: module spider Anaconda3"; return 1; }
  HW_ANACONDA_LOADED="${a}"

  if [ "${1:-}" = "cuda" ]; then
    for m in ${GCC_MODULES}; do
      if module load "${m}" >/dev/null 2>&1; then g="${m}"; break; fi
    done
    [ -n "${g}" ] || { echo "[config] no GCC module from: ${GCC_MODULES}"; return 1; }
    HW_GCC_LOADED="${g}"

    for m in ${CUDA_MODULES}; do
      if module load "${m}" >/dev/null 2>&1; then c="${m}"; break; fi
    done
    [ -n "${c}" ] || { echo "[config] no CUDA module from: ${CUDA_MODULES}"; return 1; }
    HW_CUDA_LOADED="${c}"
  fi
  module load WebProxy >/dev/null 2>&1 || true      # outbound internet on compute nodes (harmless elsewhere)
  export HW_ANACONDA_LOADED HW_GCC_LOADED HW_CUDA_LOADED
  return 0
}

hw_activate () {           # hw_activate <env prefix>
  local env="$1"
  eval "$(conda shell.bash hook)" || { echo "[config] conda hook failed"; return 1; }
  # `module purge` drops an active env from PATH while CONDA_* survive (HW1 lesson) -> deactivate fully first
  local _i=0
  while [ "${CONDA_SHLVL:-0}" -gt 0 ] && [ ${_i} -lt 6 ]; do conda deactivate >/dev/null 2>&1 || break; _i=$((_i+1)); done
  conda activate "${env}" || { echo "[config] cannot activate ${env} -- run 00_setup_env.sh first"; return 1; }
  export PATH="${env}/bin:${PATH}"
  [ "$(command -v python)" = "${env}/bin/python" ] || { echo "[config] wrong python: $(command -v python)"; return 1; }
  return 0
}
hw_activate_pc () { hw_activate "${PC_ENV}"; }
hw_activate_gs () { hw_activate "${GS_ENV}"; }

hw_banner () {
  echo ""
  echo "=================================================================="
  echo "  $1"
  echo "  $(date)"
  echo "=================================================================="
}
