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
# A thin wrapper around scripts/run_train.sh (signal forwarding, requeue, run-lock skip, summary).
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
SPECS=${G0_SPECS:-outputs/runs/G0}
SEEDS=${G0_SEEDS:-0 1 2 3 4}

specs=()
for k in $SEEDS; do
    spec="$SPECS/G0_d0_s$k.json"
    [ -f "$spec" ] || { echo "[$(date '+%F %T')] missing $spec (run scripts/build_g0.py --M 130000)"; exit 1; }
    specs+=("$spec")
done
export RUN_SPECS="${specs[*]}" SUMMARY_IDS="G0_d0_s0 G0_d0_s1 G0_d0_s2 G0_d0_s3 G0_d0_s4" SUMMARY_STRIP=G0_d0_s SUMMARY_NOUN=seeds \
    SUMMARY_NAME=G0 SUMMARY_JOB="seeds: $SEEDS"
[ -n "${G0_NO_REQUEUE:-}" ] && export NO_REQUEUE=1
exec bash "$REPO/scripts/run_train.sh"
