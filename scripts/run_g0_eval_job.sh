#!/bin/bash
# G0 evaluation job (after the 5 d=0 runs are DONE): re-evaluate the untrained base under the locked
# p_cap, then scripts/run_g0_eval.py on the 5 d=0 adapters (generation, teacher-forced scoring,
# CIDEr/SPICE, then compute_margins -> delta_S, delta_A). Every stage is resumable; on a SLURM time
# limit the running stage is stopped and the job requeues itself.
#
#   sbatch scripts/run_g0_eval_job.sh
#
#SBATCH -J g0_eval
#SBATCH -p public
#SBATCH -G a100:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 0-08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o outputs/logs/g0_eval_%j.log
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/scripts/run_g0_eval_job.sh" ] && REPO=$SLURM_SUBMIT_DIR
cd "$REPO"
PY=${CONTAM_PY:-python}  # CONTAM_PY: interpreter of the contam env
SPLIT=data/splits/split_A.json
CKPT=${CONTAM_CKPT_DIR:-data/ckpt}
stamp() { echo "[$(date '+%F %T')] $*"; }

child=0
stopping=0
on_signal() {
    stamp "got $1; stopping the running stage (pid $child)"
    stopping=1
    [ "$child" -gt 0 ] && kill -TERM "$child" 2>/dev/null
}
trap 'on_signal USR1' USR1
trap 'on_signal TERM' TERM
trap 'on_signal HUP' HUP

run() {  # run one stage in the background so signals reach this shell; requeue if stopped by one
    stamp "stage: $*"
    "$@" &
    child=$!
    while :; do
        wait "$child"; rc=$?
        kill -0 "$child" 2>/dev/null || break
    done
    child=0
    if [ "$stopping" -eq 1 ]; then
        if [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_NAME:-}" = "g0_eval" ] && [ -z "${G0_NO_REQUEUE:-}" ]; then
            stamp "requeueing job $SLURM_JOB_ID"
            scontrol requeue "$SLURM_JOB_ID"
        fi
        exit 75
    fi
    [ "$rc" -eq 0 ] || { stamp "FAILED (exit $rc): $*"; exit "$rc"; }
}

$PY -m contam.data.freeze --check || { stamp "manifest lock check failed"; exit 1; }
for k in 0 1 2 3 4; do
    [ -f "$CKPT/G0_d0_s$k/DONE" ] || { stamp "G0_d0_s$k is not DONE yet"; exit 1; }
done

# 1. untrained base under the locked p_cap. The 2026-10-21 base outputs used the old placeholder
#    p_cap ("Describe the image."); keep them aside, never mix them with the new ones.
OLD=outputs/g0/old_pcap_20261021
if [ ! -d "$OLD" ]; then
    mkdir -p "$OLD"
    for f in outputs/g0/base_caption.jsonl outputs/g0/base_metrics.json outputs/g0/base_metrics_v.json outputs/logprobs/base.jsonl; do
        [ -f "$f" ] && mv "$f" "$OLD/" && stamp "moved old-p_cap $f -> $OLD/"
    done
fi
mkdir -p outputs/g0 outputs/logprobs
run $PY -m contam.generate --checkpoint-id base --split $SPLIT --prompt-id caption --temperature 0 --out outputs/g0/base_caption.jsonl
run $PY scripts/score_refs.py --checkpoint-id base --split $SPLIT --prompt-id caption --out outputs/logprobs/base.jsonl
if [ ! -f outputs/g0/base_metrics_v.json ]; then
    run $PY scripts/evaluate_checkpoint.py --checkpoint-id base --generations outputs/g0/base_caption.jsonl \
        --logprobs outputs/logprobs/base.jsonl --split $SPLIT --prompt-id caption --out outputs/g0/base_metrics_v.json
fi

# 2. the five d=0 checkpoints + margins
RUNS=()
for k in 0 1 2 3 4; do RUNS+=(--run "G0_d0_s$k=$CKPT/G0_d0_s$k/adapter"); done
run $PY scripts/run_g0_eval.py --split $SPLIT "${RUNS[@]}" --base outputs/g0/base_metrics_v.json
# G0 is finished only when the margins are gate-valid and carry the base check (prereg G0 pass rule).
if $PY - <<'EOF'
import json, sys
m = json.load(open("outputs/g0/margins_A.json"))
ok = m.get("gate_valid") is True and m.get("base_cider_check") is not None
print("margins_A.json gate_valid", m.get("gate_valid"), "| base check", m.get("base_check"), "| in band", (m.get("base_cider_check") or {}).get("in_band"))
sys.exit(0 if ok else 1)
EOF
then
    stamp "G0 EVAL DONE"
else
    stamp "G0 INCOMPLETE: margins are not gate-valid (see run_g0_eval output above)"
    exit 2
fi
