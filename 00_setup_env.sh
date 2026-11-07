#!/bin/bash
# =============================================================================
#  00_setup_env.sh  --  ONE-TIME setup (LOGIN node, ~30-45 min, mostly downloads + CUDA builds)
#
#  Usage:   cd $SCRATCH/ECEN689/final_hprc && bash 00_setup_env.sh
#  Disk:    ~25 GB / ~90k files on $SCRATCH (two conda envs, three repos, VGGT weights) -- check `showquota`
#  Idempotent: safe to re-run after a failure (finished steps are skipped).
#
#    1. clone graphdeco-inria/gaussian-splatting (commit ${GS3D_COMMIT}), hbb1/2d-gaussian-splatting
#       and facebookresearch/vggt (commit ${VGGT_COMMIT})
#    2. env final_gs : Python 3.10 + torch 2.3.1+cu121 + 3DGS/2DGS CUDA extensions + pycolmap + open3d + vggt
#    3. env final_pc : Python 3.10 + torch 2.3.1+cu121 + numpy/scipy/sklearn/plyfile/laspy/gdown/lpips
#    4. pre-download the LPIPS/VGG weights and the VGGT-1B weights (compute nodes may have no internet)
#    5. write ${RESULTS_DIR}/env_info.txt
# =============================================================================
set -o pipefail
for _cfg in "$(dirname "${BASH_SOURCE[0]}")/config.sh" "${PWD}/config.sh" "${SLURM_SUBMIT_DIR:-/nonexistent}/config.sh"; do
  # shellcheck disable=SC1090  # config.sh is located at run time
  if [ -f "${_cfg}" ]; then source "${_cfg}"; break; fi
done
[ -n "${HW_SCRIPTS}" ] || { echo "ERROR: config.sh not found -- cd into final_hprc first"; exit 1; }

hw_banner "STEP 0/5  workspace = ${HW_ROOT}  (cluster ${CLUSTER}, CUDA arch ${TORCH_CUDA_ARCH_LIST})"
mkdir -p "${ENV_ROOT}" "${REPO_ROOT}" "${DATA_ROOT}" "${RESULTS_DIR}" "${HW_SCRIPTS}/logs" \
         "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" "${TORCH_EXTENSIONS_DIR}" "${MPLCONFIGDIR}" "${HF_HOME}"
command -v showquota >/dev/null 2>&1 && { echo "--- disk quota before setup ---"; showquota; }
hw_load_modules cuda || exit 1
echo "modules: ${HW_ANACONDA_LOADED} ${HW_GCC_LOADED} ${HW_CUDA_LOADED}"
nvcc --version | tail -n 2 || { echo "nvcc not found -- edit CUDA_MODULES in config.sh"; exit 1; }
HTTP=$(curl -sS -m 20 -o /dev/null -w "%{http_code}" https://pypi.org/simple/pip/ 2>/dev/null)
[ "${HTTP}" = "200" ] || { echo "ERROR: pypi.org not reachable from $(hostname) -- run on a LOGIN node"; exit 1; }

# -----------------------------------------------------------------------------
hw_banner "STEP 1/5  clone the official repositories"
if [ ! -d "${GS3D_REPO}/.git" ]; then
  git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git "${GS3D_REPO}" || exit 1
fi
( cd "${GS3D_REPO}" && { [ -z "${GS3D_COMMIT}" ] || git checkout -q "${GS3D_COMMIT}"; } && git submodule update --init --recursive ) || exit 1
if [ ! -d "${GS2D_REPO}/.git" ]; then
  git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git "${GS2D_REPO}" || exit 1
fi
( cd "${GS2D_REPO}" && { [ -z "${GS2D_COMMIT}" ] || git checkout -q "${GS2D_COMMIT}"; } && git submodule update --init --recursive ) || exit 1
if [ ! -d "${VGGT_REPO}/.git" ]; then
  git clone https://github.com/facebookresearch/vggt.git "${VGGT_REPO}" || exit 1
fi
( cd "${VGGT_REPO}" && { [ -z "${VGGT_COMMIT}" ] || git checkout -q "${VGGT_COMMIT}"; } ) || exit 1
echo "3DGS commit $(cd "${GS3D_REPO}" && git rev-parse --short HEAD) | 2DGS commit $(cd "${GS2D_REPO}" && git rev-parse --short HEAD) | VGGT commit $(cd "${VGGT_REPO}" && git rev-parse --short HEAD)"

eval "$(conda shell.bash hook)"
conda config --set auto_activate_base false >/dev/null 2>&1 || true
export PIP_NO_CACHE_DIR=1

make_env () {   # make_env <prefix>
  if [ ! -x "$1/bin/python" ]; then
    conda create -y -p "$1" --override-channels -c conda-forge "python=${PYTHON_VERSION}" pip || return 1
  fi
  hw_activate "$1" || return 1
  python -m pip install --upgrade pip wheel setuptools >/dev/null
  python -m pip install "numpy<2" || return 1
  python -m pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_INDEX_URL}" || return 1
}

# -----------------------------------------------------------------------------
hw_banner "STEP 2/5  env ${GS_ENV}: 3DGS + 2DGS"
make_env "${GS_ENV}" || exit 1
# opencv-python-headless pinned: newer builds force numpy 2, which breaks torch 2.3.1 (HW1 lesson)
python -m pip install plyfile tqdm joblib "opencv-python-headless==4.10.0.84" ninja pillow matplotlib pyyaml scipy \
       pycolmap "open3d==0.18.0" trimesh scikit-image mediapy || exit 1
python -m pip install "numpy<2"
export MAX_JOBS=4
BUILD_LOG="${HW_SCRIPTS}/logs/build_extensions.log"
python -c "import diff_gaussian_rasterization, simple_knn, fused_ssim" 2>/dev/null || {
  ( cd "${GS3D_REPO}" && python -m pip install --no-build-isolation -v ./submodules/diff-gaussian-rasterization \
        ./submodules/simple-knn ./submodules/fused-ssim ) > "${BUILD_LOG}" 2>&1 || {
    tail -n 30 "${BUILD_LOG}"; echo "3DGS EXTENSION BUILD FAILED -- see ${BUILD_LOG}"; exit 1; }
}
python -c "import diff_surfel_rasterization" 2>/dev/null || {
  ( cd "${GS2D_REPO}" && python -m pip install --no-build-isolation -v ./submodules/diff-surfel-rasterization ) \
        >> "${BUILD_LOG}" 2>&1 || { tail -n 30 "${BUILD_LOG}"; echo "2DGS EXTENSION BUILD FAILED -- see ${BUILD_LOG}"; exit 1; }
}
# VGGT: the package itself without its dependency list (it would pull opencv-python and numpy 2);
# the model only needs einops, safetensors and huggingface_hub on top of torch/torchvision/Pillow
python -m pip install --no-deps -e "${VGGT_REPO}" einops safetensors huggingface_hub || exit 1
python -m pip install "numpy<2"
python - <<'EOF' || exit 1
import torch, numpy, diff_gaussian_rasterization, diff_surfel_rasterization, simple_knn, fused_ssim, pycolmap, open3d
from vggt.models.vggt import VGGT  # noqa: F401
print(f"final_gs OK | torch {torch.__version__} (CUDA {torch.version.cuda}) | numpy {numpy.__version__} | "
      f"pycolmap {pycolmap.__version__} | open3d {open3d.__version__} | vggt importable")
torch.from_numpy(numpy.zeros(3))
EOF

# -----------------------------------------------------------------------------
hw_banner "STEP 3/5  env ${PC_ENV}: point clouds, analysis, figures"
make_env "${PC_ENV}" || exit 1
python -m pip install scipy scikit-learn pandas matplotlib pyyaml plyfile "laspy[lazrs]" tqdm openpyxl gdown \
       pillow lpips py7zr || exit 1
python -m pip install "numpy<2"
python - <<'EOF' || exit 1
import torch, numpy, scipy, sklearn, pandas, matplotlib, yaml, plyfile, laspy, gdown, lpips
print(f"final_pc OK | torch {torch.__version__} | numpy {numpy.__version__} | sklearn {sklearn.__version__}")
torch.from_numpy(numpy.zeros(3))
EOF

# -----------------------------------------------------------------------------
hw_banner "STEP 4/5  pre-download LPIPS (VGG) weights into TORCH_HOME=${TORCH_HOME} and VGGT-1B into HF_HOME=${HF_HOME}"
python -c "import lpips; lpips.LPIPS(net='vgg', verbose=False); print('LPIPS VGG weights cached')" || \
  echo "WARNING: LPIPS weights not cached; image_metrics.py will try again (WebProxy) or skip LPIPS"
hw_activate_gs || exit 1
# only the files are fetched (about 5 GB); the model is not built here (login-node memory limit)
python - <<EOF || echo "WARNING: VGGT weights not cached; 03g_vggt.slurm will fail offline -- download model.pt by hand into ${TORCH_HOME}/hub/checkpoints/"
from huggingface_hub import snapshot_download
p = snapshot_download("${VGGT_HF_ID}", allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin"])
print("VGGT-1B weights cached:", p)
EOF

# -----------------------------------------------------------------------------
hw_banner "STEP 5/5  ${RESULTS_DIR}/env_info.txt"
{
  echo "date        : $(date)"
  echo "login node  : $(hostname)   cluster: ${CLUSTER}"
  echo "modules     : ${HW_ANACONDA_LOADED} ${HW_GCC_LOADED} ${HW_CUDA_LOADED} WebProxy"
  echo "nvcc        : $(nvcc --version | grep release | sed 's/^ *//')"
  echo "3DGS        : $(cd "${GS3D_REPO}" && git rev-parse --short HEAD)  https://github.com/graphdeco-inria/gaussian-splatting"
  echo "2DGS        : $(cd "${GS2D_REPO}" && git rev-parse --short HEAD)  https://github.com/hbb1/2d-gaussian-splatting"
  echo "VGGT        : $(cd "${VGGT_REPO}" && git rev-parse --short HEAD)  https://github.com/facebookresearch/vggt  weights ${VGGT_HF_ID}"
  echo "torch       : $(python -c 'import torch;print(torch.__version__)')   arch list: ${TORCH_CUDA_ARCH_LIST}"
} | tee "${RESULTS_DIR}/env_info.txt"
conda clean --all -y >/dev/null 2>&1 || true
command -v showquota >/dev/null 2>&1 && { echo "--- disk quota after setup ---"; showquota; }
hw_banner "SETUP COMPLETE.  Next:  sbatch 00b_smoke_test.slurm   then   bash 01_download.sh"
