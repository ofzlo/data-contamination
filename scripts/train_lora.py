#!/usr/bin/env python
"""LoRA SFT of one run on the GPU of the current allocation (one process). Recipe, example format and
resume rules: contam.train.

Input is either a run spec (scripts/build_g0.py, later G1/G2 builders: data path + sha256, seed
slot, model, eval splits) or a plain LLaVA-format JSON with --seed-slot and --run-id. With a run
spec the manifest lock must hold and contain general_order.jsonl and the spec's eval splits
(--skip-manifest-check: timing / smoke only; the manifest then records null).

Auto-resumes from <out-dir>/checkpoints; a DONE run is a no-op. On SIGTERM / SIGUSR1 / SIGHUP it
saves a checkpoint and exits 75 (requeue and rerun the same command). If another process is training the
same out dir (run lock, contam.train) it exits 76 without touching it. For SLURM, e.g.
``sbatch --signal=B:USR1@180 ...`` with ``exec python scripts/train_lora.py ...``.

Examples:
    python scripts/train_lora.py --run-spec outputs/runs/G0/G0_d0_s0.json
    python scripts/train_lora.py --data sample.json --seed-slot 0 --run-id smoke --out-dir data/ckpt/smoke \
        --skip-manifest-check --allow-missing-images --max-steps 4
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from contam.config import load_config, read_json
from contam.data.freeze import DEFAULT_LOCK
from contam.train import EXIT_BUSY, EXIT_PREEMPTED, Preempted, RunBusy, check_run_spec_frozen, train


# variable-length micro-batches: avoid allocator fragmentation (no effect on the numerics)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-spec", default=None, help="run spec JSON (build_g0.py)")
    ap.add_argument("--data", default=None, help="LLaVA-format training JSON (without --run-spec)")
    ap.add_argument("--seed-slot", type=int, default=None, help="training.seeds slot (without --run-spec)")
    ap.add_argument("--run-id", default=None, help="without --run-spec")
    ap.add_argument("--out-dir", default=None, help="default: <paths.ckpt_dir>/<run_id>")
    ap.add_argument("--image-root", default=None, help="default: paths.llava_images")
    ap.add_argument("--micro-batch-size", type=int, default=None, help="override training.micro_batch_size (part of the config hash)")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps (timing; no DONE)")
    ap.add_argument("--max-examples", type=int, default=None, help="first N examples only (smoke tests; part of the config hash)")
    ap.add_argument("--allow-missing-images", action="store_true", help="timing / smoke only: drop examples whose image is missing (count recorded)")
    ap.add_argument("--lock", default=str(DEFAULT_LOCK))
    ap.add_argument("--skip-manifest-check", action="store_true", help="timing / smoke only: no manifest lock check (manifest records null)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.micro_batch_size:
        cfg["training"]["micro_batch_size"] = args.micro_batch_size
    if args.num_workers is not None:
        cfg["training"]["num_workers"] = args.num_workers
    spec = read_json(args.run_spec) if args.run_spec else None
    if spec:
        if args.data or args.seed_slot is not None or args.run_id:
            ap.error("--run-spec sets the data, seed slot and run id")
        data, slot, run_id = spec["data"]["path"], int(spec["seed_slot"]), spec["run_id"]
    else:
        if not (args.data and args.seed_slot is not None and args.run_id):
            ap.error("without --run-spec, give --data, --seed-slot and --run-id")
        data, slot, run_id = args.data, args.seed_slot, args.run_id
    if args.skip_manifest_check:
        print("WARNING: --skip-manifest-check: this run is for timing / smoke tests only", file=sys.stderr)
        frozen = None
    elif spec:
        frozen = check_run_spec_frozen(cfg, spec, args.lock)
    else:
        ap.error("a plain --data run has no run spec to check against the manifest lock; pass --skip-manifest-check (timing / smoke only)")
    out = Path(args.out_dir or Path(cfg["paths"]["ckpt_dir"]) / run_id)
    out.mkdir(parents=True, exist_ok=True)
    log_file = None  # opened at the first message: a busy exit must not touch train.log (its mtime is part of the run lock)

    def log(msg: str) -> None:
        nonlocal log_file
        line = f"[{datetime.now():%F %T}] {msg}"
        print(line, flush=True)
        log_file = log_file or open(out / "train.log", "a")
        log_file.write(line + "\n")
        log_file.flush()

    try:
        res = train(cfg, data, out, slot, run_id, run_spec=spec, run_spec_path=args.run_spec, frozen=frozen, image_root=args.image_root,
                    max_steps=args.max_steps, max_examples=args.max_examples, allow_missing_images=args.allow_missing_images, log=log)
    except RunBusy as e:
        print(f"[{datetime.now():%F %T}] {run_id}: busy, not started: {e}; exiting {EXIT_BUSY}", file=sys.stderr, flush=True)
        sys.exit(EXIT_BUSY)
    except Preempted as e:
        log(f"{run_id}: stopped by {e}; checkpoint saved, exiting {EXIT_PREEMPTED} (rerun to resume)")
        sys.exit(EXIT_PREEMPTED)
    log(f"{run_id}: {res['status']}")


if __name__ == "__main__":
    main()
