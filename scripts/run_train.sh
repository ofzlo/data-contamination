#!/bin/bash
# Training runner: the run specs in RUN_SPECS (space-separated paths, each named <run_id>.json, e.g.
# from scripts/build_g0.py or scripts/build_g1.py), one after another, on one GPU. Resumable: rerun
# the same command after a preemption or a SLURM time limit; finished runs are no-ops (DONE), an
# interrupted run resumes from its last checkpoint. A run another job is training (run lock, exit 76)
# is skipped, so several jobs can share a list. Run from anywhere:
#
#   RUN_SPECS="outputs/runs/G1/G1_Paired_d4_s0.json outputs/runs/G1/G1_Caption_d4_s0.json" bash scripts/run_train.sh
#   RUN_SPECS="..." sbatch --export=ALL -J g1_train scripts/run_train.sh   # requeues itself on preemption / time limit
#
# A preempted run requeues the job only for a batch job named "train" or "*_train" (not inside an
# interactive allocation); NO_REQUEUE=1 turns that off. The end summary lists DONE / not DONE run
# ids of SUMMARY_IDS (default: the run ids of RUN_SPECS); scripts/run_g0_train.sh sets the
# SUMMARY_* variables for its per-seed summary.
#
#SBATCH -J train
#SBATCH -p public
#SBATCH -G a100:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 0-08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o outputs/logs/train_%j.log
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/run_train.sh" ] && REPO=$SLURM_SUBMIT_DIR
cd "$REPO"
PY=${CONTAM_PY:-/scratch/$USER/envs/contam/bin/python}
CKPT=${CONTAM_CKPT_DIR:-data/ckpt}  # paths.ckpt_dir
SPECS=${RUN_SPECS:?set RUN_SPECS to the run spec paths}
EXIT_PREEMPTED=75
EXIT_BUSY=76
stamp() { echo "[$(date '+%F %T')] $*"; }

ids=()
for spec in $SPECS; do
    [ -f "$spec" ] || { stamp "missing $spec"; exit 1; }
    ids+=("$(basename "$spec" .json)")
done
SUMMARY_IDS=${SUMMARY_IDS:-${ids[*]}}

child=0
forward() {  # SLURM signals the batch shell; pass it on so the trainer checkpoints and exits 75
    stamp "got $1; forwarding to trainer pid $child"
    [ "$child" -gt 0 ] && kill -"$1" "$child" 2>/dev/null
}
trap 'forward USR1' USR1
trap 'forward TERM' TERM
trap 'forward HUP' HUP

$PY -m contam.data.freeze --check || { stamp "manifest lock check failed"; exit 1; }

for spec in $SPECS; do
    id=$(basename "$spec" .json)
    stamp "$id: start"
    $PY scripts/train_lora.py --run-spec "$spec" &
    child=$!
    while :; do  # `wait` returns early when a trapped signal arrives; keep waiting for the trainer
        wait "$child"; rc=$?
        kill -0 "$child" 2>/dev/null || break
    done
    child=0
    if [ "$rc" -eq "$EXIT_PREEMPTED" ]; then
        stamp "$id: preempted (checkpoint saved)"
        if [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_PARTITION:-}" != "" ] && [ -z "${NO_REQUEUE:-}" ]; then
            case "${SLURM_JOB_NAME:-}" in
                train|*_train) stamp "requeueing job $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID" ;;
            esac
        fi
        exit "$EXIT_PREEMPTED"
    elif [ "$rc" -eq "$EXIT_BUSY" ]; then
        stamp "$id: busy (another job is training it), skipping"
        continue
    elif [ "$rc" -ne 0 ]; then
        stamp "$id: FAILED (exit $rc)"
        exit "$rc"
    fi
    stamp "$id: done"
done
done_ids=(); open_ids=()
for id in $SUMMARY_IDS; do
    if [ -f "$CKPT/$id/DONE" ]; then done_ids+=("${id#"${SUMMARY_STRIP:-}"}"); else open_ids+=("${id#"${SUMMARY_STRIP:-}"}"); fi
done
stamp "DONE ${SUMMARY_NOUN:-runs}: ${done_ids[*]:-none}; not DONE: ${open_ids[*]:-none}"
if [ "${#open_ids[@]}" -eq 0 ]; then stamp "${SUMMARY_NAME:+$SUMMARY_NAME }ALL DONE"; else stamp "nothing left for this job (${SUMMARY_JOB:-${ids[*]}})"; fi
