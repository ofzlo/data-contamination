#!/bin/bash
# EXPLORATORY short-prompt evaluation (docs/deviations.md; not a gate input): T=0 captions under
# nocaps_like ("Describe the image in one short sentence.") for trained checkpoints and the untrained
# base (RUN_IDS entry "base": no adapter), then evaluate_checkpoint without log-probs (Q, S, exact
# match). Resumable: generation resumes per fixed chunk, and a run whose metrics file exists is
# skipped. On a SLURM time limit (USR1) the running stage is stopped and the job requeues itself;
# TERM / HUP (scancel) stop it and exit 75 without requeueing.
#
#   RUN_IDS="base G0_d0_s0 G0_d0_s1 G0_d0_s2 G0_d0_s3 G0_d0_s4" bash scripts/run_short_eval_job.sh
#   RUN_IDS="G2_Paired_d1_s0 G1_Paired_d4_s0" sbatch --export=ALL scripts/run_short_eval_job.sh
#
# Outputs (new files only): outputs/short/<run_id>_nocaps_like.jsonl and
# outputs/short/<run_id>_metrics_short_<split>.json, where <split> is A for split_A.json.
#
#SBATCH -J short_eval
#SBATCH -p public
#SBATCH -G a100:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 0-08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o outputs/logs/short_eval_%j.log
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/run_short_eval_job.sh" ] && REPO=$SLURM_SUBMIT_DIR
cd "$REPO"
PY=${CONTAM_PY:-/scratch/$USER/envs/contam/bin/python}
SPLIT=${SPLIT:-data/splits/split_A.json}
CKPT=${CONTAM_CKPT_DIR:-data/ckpt}
PROMPT_ID=${PROMPT_ID:-nocaps_like}
OUT=${SHORT_OUT:-outputs/short}
: "${RUN_IDS:?set RUN_IDS, e.g. RUN_IDS=\"base G0_d0_s0\"}"
SNAME=$(basename "$SPLIT" .json); SNAME=${SNAME#split_}
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
    if [ "$requeue" -eq 1 ] && [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_NAME:-}" = "short_eval" ] && [ -z "${NO_REQUEUE:-}" ]; then
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
mkdir -p "$OUT"
for id in $RUN_IDS; do
    gens="$OUT/${id}_${PROMPT_ID}.jsonl"
    metrics="$OUT/${id}_metrics_short_${SNAME}.json"
    if [ -f "$metrics" ]; then
        stamp "$id: $metrics exists, skipping"
        continue
    fi
    if [ "$id" = "base" ]; then
        adapter=()
    else
        [ -f "$CKPT/$id/DONE" ] || { stamp "$id is not DONE"; exit 1; }
        adapter=(--adapter "$CKPT/$id/adapter")
    fi
    run $PY -m contam.generate --checkpoint-id "$id" "${adapter[@]}" --split "$SPLIT" \
        --prompt-id "$PROMPT_ID" --temperature 0 --out "$gens"
    run $PY scripts/evaluate_checkpoint.py --checkpoint-id "$id" --generations "$gens" --split "$SPLIT" \
        --prompt-id "$PROMPT_ID" --out "$metrics"
    stamp "$id: short-prompt eval done"
done
stamp "ALL SHORT-PROMPT EVALS DONE ($RUN_IDS)"
