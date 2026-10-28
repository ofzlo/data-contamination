#!/bin/bash
# G0 training: d=0 x 5 seeds, one after another, on one GPU (D2: M = 130,000; specs from
# `python scripts/build_g0.py --M 130000`). Resumable: rerun the same command after a preemption or
# a SLURM time limit; finished seeds are no-ops (DONE), an interrupted seed resumes from its last
# checkpoint. Several jobs can share the seeds: G0_SEEDS picks this job's seeds (default all five, in
# order); a seed another job is training (run lock, exit 76) is skipped. Run from anywhere:
#
#   bash scripts/run_g0_train.sh                       # in an interactive GPU session (setsid nohup ...)
#   sbatch scripts/run_g0_train.sh                     # batch: requeues itself on preemption / time limit
#   G0_SEEDS="3 4" sbatch --export=ALL scripts/run_g0_train.sh   # a second job for seeds 3 and 4
#
#SBATCH -J g0_train
#SBATCH -p public
#SBATCH -G a100:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 0-08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o outputs/logs/g0_train_%j.log
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/run_g0_train.sh" ] && REPO=$SLURM_SUBMIT_DIR
cd "$REPO"
PY=${CONTAM_PY:-/scratch/$USER/envs/contam/bin/python}
SPECS=${G0_SPECS:-outputs/runs/G0}
CKPT=${CONTAM_CKPT_DIR:-data/ckpt}  # paths.ckpt_dir
SEEDS=${G0_SEEDS:-0 1 2 3 4}
EXIT_PREEMPTED=75
EXIT_BUSY=76
stamp() { echo "[$(date '+%F %T')] $*"; }

child=0
forward() {  # SLURM signals the batch shell; pass it on so the trainer checkpoints and exits 75
    stamp "got $1; forwarding to trainer pid $child"
    [ "$child" -gt 0 ] && kill -"$1" "$child" 2>/dev/null
}
trap 'forward USR1' USR1
trap 'forward TERM' TERM
trap 'forward HUP' HUP

$PY -m contam.data.freeze --check || { stamp "manifest lock check failed"; exit 1; }

for k in $SEEDS; do
    spec="$SPECS/G0_d0_s$k.json"
    [ -f "$spec" ] || { stamp "missing $spec (run scripts/build_g0.py --M 130000)"; exit 1; }
    stamp "G0_d0_s$k: start"
    $PY scripts/train_lora.py --run-spec "$spec" &
    child=$!
    while :; do  # `wait` returns early when a trapped signal arrives; keep waiting for the trainer
        wait "$child"; rc=$?
        kill -0 "$child" 2>/dev/null || break
    done
    child=0
    if [ "$rc" -eq "$EXIT_PREEMPTED" ]; then
        stamp "G0_d0_s$k: preempted (checkpoint saved)"
        if [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_PARTITION:-}" != "" ] && [ -z "${G0_NO_REQUEUE:-}" ] && [ "${SLURM_JOB_NAME:-}" = "g0_train" ]; then
            stamp "requeueing job $SLURM_JOB_ID"
            scontrol requeue "$SLURM_JOB_ID"
        fi
        exit "$EXIT_PREEMPTED"
    elif [ "$rc" -eq "$EXIT_BUSY" ]; then
        stamp "G0_d0_s$k: busy (another job is training it), skipping"
        continue
    elif [ "$rc" -ne 0 ]; then
        stamp "G0_d0_s$k: FAILED (exit $rc)"
        exit "$rc"
    fi
    stamp "G0_d0_s$k: done"
done
done_seeds=(); open_seeds=()
for k in 0 1 2 3 4; do
    if [ -f "$CKPT/G0_d0_s$k/DONE" ]; then done_seeds+=("$k"); else open_seeds+=("$k"); fi
done
stamp "DONE seeds: ${done_seeds[*]:-none}; not DONE: ${open_seeds[*]:-none}"
if [ "${#open_seeds[@]}" -eq 0 ]; then stamp "G0 ALL DONE"; else stamp "nothing left for this job (seeds: $SEEDS)"; fi
