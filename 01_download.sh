#!/bin/bash
# =============================================================================
#  01_download.sh  --  BrPCD (Google Drive) + the needed VTTI files (figshare, md5-checked)
#
#  Usage (LOGIN node, resumable):   bash 01_download.sh            # both datasets
#                                   bash 01_download.sh vtti       # one of: brpcd | vtti
#  VTTI_ALL=1 also fetches the reduced-resolution images and the FRR/RRO/RRR projects (+23 GB).
#  Disk: VTTI ~8.6 GB zips + the same again extracted; BrPCD a few GB.
#
#  If Google Drive refuses the automated download ("Too many users" / quota), download the three
#  BrPCD archives in a browser from github.com/wxiong12/SEU-Digital-Bridge and upload them to
#  ${BRPCD_RAW}/archives/ (portal Files app or scp), then rerun:  bash 01_download.sh brpcd
# =============================================================================
set -o pipefail
for _cfg in "$(dirname "${BASH_SOURCE[0]}")/config.sh" "${PWD}/config.sh" "${SLURM_SUBMIT_DIR:-/nonexistent}/config.sh"; do
  # shellcheck disable=SC1090  # config.sh is located at run time
  if [ -f "${_cfg}" ]; then source "${_cfg}"; break; fi
done
[ -n "${HW_SCRIPTS}" ] || { echo "ERROR: config.sh not found -- cd into final_hprc first"; exit 1; }
WHAT="${1:-all}"
hw_load_modules || exit 1
hw_activate_pc || exit 1
mkdir -p "${BRPCD_RAW}/archives" "${VTTI_RAW}" "${VTTI_DIR}"

extract () {   # extract <archive> <dest dir>
  local f="$1" d="$2"
  mkdir -p "${d}"
  case "${f,,}" in
    *.zip)            python - "$f" "$d" <<'EOF'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); z.extractall(sys.argv[2]); print(f"  extracted {len(z.namelist())} entries")
EOF
                      ;;
    *.tar|*.tar.gz|*.tgz) tar -xf "$f" -C "$d" ;;
    *.7z)             python -c "import py7zr,sys; py7zr.SevenZipFile(sys.argv[1]).extractall(sys.argv[2])" "$f" "$d" ;;
    *.rar)            if command -v unrar >/dev/null; then unrar x -o+ "$f" "$d/"; else
                        echo "  .rar needs unrar: try  module spider unrar  /  module spider p7zip  (7z x file.rar)"; return 1; fi ;;
    *)                echo "  not an archive: $f"; return 0 ;;
  esac
}

# -----------------------------------------------------------------------------
if [ "${WHAT}" = "all" ] || [ "${WHAT}" = "brpcd" ]; then
  hw_banner "BrPCD -> ${BRPCD_RAW}"
  find "${BRPCD_RAW}/archives" -name "*part*" -delete
  if [ -z "$(ls -A "${BRPCD_RAW}/archives" 2>/dev/null)" ]; then
    ( cd "${BRPCD_RAW}/archives" && gdown --folder "${BRPCD_GDRIVE_FOLDER}" -O . ) \
      || echo "WARNING: gdown failed -- see the header of this script for the manual route"
  else
    echo "archives already present: $(ls "${BRPCD_RAW}/archives" | tr '\n' ' ')"
  fi
  find "${BRPCD_RAW}/archives" -type f | while read -r f; do
    stamp="${f}.extracted"
    [ -f "${stamp}" ] && continue
    echo "extracting $(basename "$f")"
    extract "$f" "${BRPCD_RAW}/tree" && touch "${stamp}"
  done
  echo "BrPCD tree: $(find "${BRPCD_RAW}/tree" -type f 2>/dev/null | wc -l) files"
fi

# -----------------------------------------------------------------------------
if [ "${WHAT}" = "all" ] || [ "${WHAT}" = "vtti" ]; then
  hw_banner "VTTI (doi 10.7294/29901590.v2) -> ${VTTI_RAW}"
  echo "${VTTI_FILES}" | while IFS='|' read -r name fid md5 needed; do
    [ -z "${name}" ] && continue
    if [ "${needed}" != "1" ] && [ "${VTTI_ALL:-0}" != "1" ]; then echo "skip   ${name} (not needed)"; continue; fi
    out="${VTTI_RAW}/${name}"
    if [ -f "${out}.md5ok" ]; then echo "ok     ${name}"; continue; fi
    for attempt in 1 2 3; do
      echo "fetch  ${name} (attempt ${attempt})"
      curl -L --fail -C - --retry 5 --retry-delay 10 -o "${out}" "https://ndownloader.figshare.com/files/${fid}" || true
      got=$(md5sum "${out}" 2>/dev/null | cut -d' ' -f1)
      if [ "${got}" = "${md5}" ]; then touch "${out}.md5ok"; echo "md5 OK ${name}"; break; fi
      echo "md5 mismatch (${got} != ${md5}) -> restarting the file"; rm -f "${out}"
    done
  done
  for z in "${VTTI_RAW}"/*.zip; do
    [ -f "${z}.md5ok" ] || continue
    dest="${VTTI_DIR}/$(basename "${z}" .zip)"
    [ -f "${dest}/.extracted" ] && { echo "already extracted: $(basename "${z}")"; continue; }
    echo "extracting $(basename "${z}") -> ${dest}"
    extract "${z}" "${dest}" && touch "${dest}/.extracted"
  done
  cp -f "${VTTI_RAW}"/*.xlsx "${VTTI_RAW}"/*.docx "${VTTI_RAW}"/*.rtf "${VTTI_DIR}/" 2>/dev/null || true
  du -sh "${VTTI_RAW}" "${VTTI_DIR}" 2>/dev/null
fi
command -v showquota >/dev/null 2>&1 && showquota
hw_banner "DOWNLOAD DONE.  Next:  sbatch 02_prepare.slurm   (or bash run_all.sh for the whole chain)"
