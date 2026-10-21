#!/usr/bin/env python
"""Prepare G0: d=0 x 5 training seeds, plus the untrained base (eval only). Does NOT start training.

d=0 has no exposed examples, so every seed trains on the same data: the first M examples of the
frozen general order (``general_order.jsonl``). Only the training seed changes. M is D2 and must be
given explicitly. The manifests must be frozen (``python -m contam.data.freeze``) with
``general_order.jsonl`` in the lock, and the lock must still match.

The d=0 model sees no NoCaps item, so one set of d=0 checkpoints serves both splits (prereg
Section 2): each run spec lists split A and split B for evaluation.

Writes, under --out (default <outputs_dir>/runs):
    data/general_M{M}.json     LLaVA-format training file, identical for the 5 seeds
    G0/G0_d0_s{k}.json         one run spec per seed slot k = 0..4
    G0/G0_base.json            untrained base model, eval only

Example:
    python scripts/build_g0.py --M 300000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contam.config import file_sha256, load_config, read_json, read_jsonl, write_json
from contam.data.freeze import DEFAULT_LOCK, verify

N_G0_SEEDS = 5


def build(cfg: dict, M: int, out: Path, lock_path=DEFAULT_LOCK) -> dict:
    verify(cfg, lock_path, required=["split_A.json", "split_B.json", "general_order.jsonl"])
    lock_sha = file_sha256(lock_path)
    splits_dir = Path(cfg["paths"]["splits_dir"])
    seeds = list(cfg["training"]["seeds"])
    if len(seeds) < N_G0_SEEDS or len(set(seeds)) != len(seeds):
        raise SystemExit(f"need {N_G0_SEEDS} distinct training seeds, got {seeds}")

    order = list(read_jsonl(splits_dir / "general_order.jsonl"))
    if not 0 < M <= len(order):
        raise SystemExit(f"M={M} must be in 1..{len(order)} (general pool size)")
    mix = read_json(cfg["paths"]["llava_mix"])
    examples = [mix[o["src_index"]] for o in order[:M]]

    data_path = out / "data" / f"general_M{M}.json"
    write_json(examples, data_path, indent=None)
    data_sha = file_sha256(data_path)

    splits = {n: read_json(splits_dir / f"split_{n}.json")["content_hash"] for n in ("A", "B")}
    common = {
        "gate": "G0",
        "condition": "d0",
        "d": 0,
        "K": 0,
        "M": M,
        "manifest_lock_sha256": lock_sha,
        "eval_splits": {n: {"path": f"split_{n}.json", "content_hash": h} for n, h in splits.items()},
        "eval_temperatures": [0.0, 1.0],
    }
    g0 = out / "G0"
    runs = []
    for k in range(N_G0_SEEDS):
        spec = {
            **common,
            "run_id": f"G0_d0_s{k}",
            "seed_slot": k,
            "train_seed": int(seeds[k]),
            "data": {"path": str(data_path.resolve()), "sha256": data_sha, "n_examples": M, "general_prefix": M, "exposed": 0},
            "model": cfg["models"]["main"],
        }
        write_json(spec, g0 / f"{spec['run_id']}.json")
        runs.append(spec["run_id"])
    write_json({**common, "run_id": "G0_base", "M": 0, "train_seed": None, "data": None, "model": cfg["models"]["main"]}, g0 / "G0_base.json")
    return {"runs": runs, "data": str(data_path), "data_sha256": data_sha, "n_examples": M}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--M", type=int, required=True, help="total examples per run (D2)")
    ap.add_argument("--out", default=None, help="default: <outputs_dir>/runs")
    ap.add_argument("--lock", default=str(DEFAULT_LOCK))
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out) if args.out else Path(cfg["paths"]["outputs_dir"]) / "runs"
    print(json.dumps(build(cfg, args.M, out, args.lock), indent=1))


if __name__ == "__main__":
    main()
