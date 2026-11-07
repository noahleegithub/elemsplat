#!/bin/bash
# =============================================================================
#  run_all.sh  --  submit the whole chain with Slurm dependencies (after 00 setup and 01 download)
#
#      bash run_all.sh                 # ACES (default)
#      CLUSTER=grace bash run_all.sh   # Grace: A100 for every GPU job, CPU partition "medium"
#      SKIP_PREPARE=1 bash run_all.sh  # 02 already done
#      VGGT=0 bash run_all.sh          # without the feed-forward baseline (03g)
#
#  02 prepare ─┬─ 03a semseg (pointnet2, pointnet; 10 folds + all) ── 03b instances ─┐
#              ├─ 03c 3DGS ─┐                                                       ├─ 03f merge ── 04 pack
#              ├─ 03d 2DGS ─┼─ 03e measure ─────────────────────────────────────────┘
#              └─ 03g VGGT ─┘   (optional: VGGT=0 skips it; a failed 03g does not block 03e)
# =============================================================================
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
source ./config.sh || exit 1
mkdir -p logs
CPU=(); [ -n "${CPU_PARTITION}" ] && CPU=(--partition="${CPU_PARTITION}")
GS=(--partition="${GPU_PARTITION}" --gres="gpu:${GPU_SMALL}:1")
GB=(--partition="${GPU_PARTITION}" --gres="gpu:${GPU_BIG}:1")
ACC=(); [ -n "${SLURM_ACCOUNT}" ] && ACC=(--account="${SLURM_ACCOUNT}")
sb () { sbatch --parsable "${ACC[@]}" "$@"; }
# dep prints one "--dependency=afterok:a:b" token or nothing; it is left unquoted on purpose
dep () { local d=""; for j in "$@"; do [ -n "$j" ] && d="${d:+${d}:}$j"; done; [ -n "$d" ] && echo "--dependency=afterok:${d}"; }
need () { [ -n "$1" ] || { echo "submitting $2 failed -- cancel the jobs submitted above (scancel <id>) and fix it"; exit 1; }; }
# shellcheck disable=SC2046
{
if [ "${SKIP_PREPARE:-0}" = "1" ]; then P=""; else P=$(sb "${CPU[@]}" 02_prepare.slurm); need "$P" 02_prepare; fi
NARR="${NARR:-0-10}"                                 # 10 folds + "all"; extra indices exit at once
S2=$(sb "${GS[@]}" $(dep "$P") --array="${NARR}" --export=ALL,MODEL=pointnet2 03a_semseg.slurm); need "$S2" 03a_pointnet2
S1=$(sb "${GS[@]}" $(dep "$P") --array="${NARR}" --export=ALL,MODEL=pointnet 03a_semseg.slurm);   need "$S1" 03a_pointnet
I2=$(sb "${CPU[@]}" $(dep "$S2") --array=0-9 --export=ALL,MODEL=pointnet2,TRUE=1 03b_instances.slurm); need "$I2" 03b_pointnet2
I1=$(sb "${CPU[@]}" $(dep "$S1") --array=0-9 --export=ALL,MODEL=pointnet,TRUE=0 03b_instances.slurm);  need "$I1" 03b_pointnet
G3=$(sb "${GB[@]}" $(dep "$P") 03c_gs3d.slurm);                       need "$G3" 03c_gs3d
G2=$(sb "${GB[@]}" $(dep "$P") 03d_gs2d.slurm);                       need "$G2" 03d_gs2d
if [ "${VGGT:-1}" = "1" ]; then G4=$(sb "${GB[@]}" $(dep "$P") 03g_vggt.slurm); need "$G4" 03g_vggt; else G4=""; fi
# 03e waits for 3DGS and 2DGS to succeed; VGGT only has to finish (afterany), so a VGGT failure costs one table row
ME=$(sb "${CPU[@]}" $(dep "$G3" "$G2")${G4:+,afterany:${G4}} 03e_measure.slurm); need "$ME" 03e_measure
MG=$(sb "${GS[@]}" $(dep "$ME" "$S2" "$I2") 03f_merge.slurm);         need "$MG" 03f_merge
PK=$(sb "${CPU[@]}" --dependency=afterany:"${MG}":"${I1}" 04_eval_pack.sh); need "$PK" 04_eval_pack
}
cat <<EOF
submitted on ${CLUSTER}:
  02 prepare ${P:-skipped} | 03a semseg pointnet2 ${S2}, pointnet ${S1} | 03b instances ${I2}, ${I1}
  03c 3DGS ${G3} | 03d 2DGS ${G2} | 03g VGGT ${G4:-skipped} | 03e measure ${ME} | 03f merge ${MG} | 04 pack ${PK}
watch:   squeue -u \$USER        logs:  logs/<job>_<id>.out
EOF
