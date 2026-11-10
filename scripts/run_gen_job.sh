#!/bin/bash
# GPU stages of evaluation for trained checkpoints: T=0 caption generation under p_cap and
# teacher-forced scoring of all 10 references (inputs to S/A/V/Q). Both stages are resumable; on a
# SLURM time limit (USR1) the running stage is stopped and the job requeues itself; TERM / HUP (scancel)
# stop it and exit 75 without requeueing. CPU evaluation
# (evaluate_checkpoint) runs separately.
#
#   RUN_IDS="G1_Paired_d4_s0" bash scripts/run_gen_job.sh              # interactive GPU session
#   RUN_IDS="G1_Caption_d4_s0" sbatch --export=ALL scripts/run_gen_job.sh
#
# Outputs: outputs/<gate lower-case>/<run_id>_caption.jsonl and outputs/logprobs/<run_id>.jsonl.
#
#SBATCH -J gen_eval
#SBATCH -p public
#SBATCH -G a100:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 0-08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o outputs/logs/gen_eval_%j.log
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/run_gen_job.sh" ] && REPO=$SLURM_SUBMIT_DIR
cd "$REPO"
PY=${CONTAM_PY:-python}  # CONTAM_PY: interpreter of the contam env
SPLIT=${SPLIT:-data/splits/split_A.json}
CKPT=${CONTAM_CKPT_DIR:-data/ckpt}
: "${RUN_IDS:?set RUN_IDS, e.g. RUN_IDS=\"G1_Paired_d4_s0\"}"
stamp() { echo "[$(date '+%F %T')] $*"; }

child=0
stopping=0
requeue=0
on_signal() {
    stamp "got $1; stopping the running stage (pid $child)"
    stopping=1
    [ "$1" = USR1 ] && requeue=1  # only the time-limit warning requeues; scancel (TERM) / HUP just stop
    [ "$child" -gt 0 ] && kill -TERM "$child" 2>/dev/null
}
stop_now() {
    if [ "$requeue" -eq 1 ] && [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_NAME:-}" = "gen_eval" ] && [ -z "${NO_REQUEUE:-}" ]; then
        stamp "requeueing job $SLURM_JOB_ID"
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit 75
}
trap 'on_signal USR1' USR1
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP

run() {
    [ "$stopping" -eq 1 ] && stop_now  # a signal arrived between stages
    stamp "stage: $*"
    "$@" &
    child=$!
    while :; do
        wait "$child"; rc=$?
        kill -0 "$child" 2>/dev/null || break
    done
    child=0
    [ "$stopping" -eq 1 ] && stop_now
    [ "$rc" -eq 0 ] || { stamp "FAILED (exit $rc): $*"; exit "$rc"; }
}

$PY -m contam.data.freeze --check || { stamp "manifest lock check failed"; exit 1; }
for id in $RUN_IDS; do
    [ -f "$CKPT/$id/DONE" ] || { stamp "$id is not DONE"; exit 1; }
    gate=$(echo "${id%%_*}" | tr '[:upper:]' '[:lower:]')
    mkdir -p "outputs/$gate" outputs/logprobs
    run $PY -m contam.generate --checkpoint-id "$id" --adapter "$CKPT/$id/adapter" --split "$SPLIT" \
        --prompt-id caption --temperature 0 --out "outputs/$gate/${id}_caption.jsonl"
    run $PY scripts/score_refs.py --checkpoint-id "$id" --adapter "$CKPT/$id/adapter" --split "$SPLIT" \
        --prompt-id caption --out "outputs/logprobs/$id.jsonl"
    stamp "$id: GPU stages done"
done
stamp "ALL GPU STAGES DONE ($RUN_IDS)"
