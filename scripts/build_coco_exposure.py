#!/usr/bin/env python
"""Prepare the COCO image-study runs: training data + one run spec per (level, half, seed slot).
Does NOT start training.

As in ``build_g1.py``, a run with K = N_exp x d exposed examples trains on the first M - K examples of
the frozen general order followed by the K exposed ones.
* N_exp is the size of the exposed half: every pool image of that half is exposed.
* The levels are L3, L2, L1 and Cap (``contam.data.coco_study``).

Checks before anything is written:
* the manifest lock holds, and ``coco_pool.json`` and ``general_order.jsonl`` are locked and
  unchanged;
* the COCO captions and the VQAv2 questions / annotations are the files the pool was built from;
* every exposed image decodes under ``paths.llava_images`` as ``coco_val2017/<file>``. ``--link-coco``
  creates the symlink ``<llava_images>/coco_val2017 -> coco_study.images_dir`` if it is missing.

Writes, under --out (default <outputs_dir>/runs):
    data/COCO_{level}_{half}_d{d}_M{M}.json            LLaVA-format training file (M examples)
    data/COCO_{level}_{half}_d{d}_M{M}.exposure.json   the exposed rows + meta
    C/C_{level}_{half}_d{d}_s{slot}.json               run spec for scripts/run_train.sh

The training and exposure files depend only on the frozen inputs, so Minho (H200) and Yoorhim (Sol)
get byte-identical files. Compare them with Minho's ``outputs/coco/coco_runs_sha256.txt`` on HF. Run
specs hold absolute paths, so they differ between servers; each records its data file's sha256.

Example (all 8 runs, seed slot 0):
    python scripts/build_coco_exposure.py --link-coco
    for f in outputs/runs/C/*.json; do id=$(basename $f .json); RUN_SPECS=$f sbatch --export=ALL -J ${id}_train scripts/run_train.sh; done
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from contam.config import content_hash, file_sha256, load_config, load_prompts, read_json, read_jsonl, write_json
from contam.data import coco_study as cs
from contam.data.freeze import DEFAULT_LOCK, verify
from contam.train import find_missing_images

POOL_FILE = "coco_pool.json"


def ensure_coco_link(cfg: dict, create: bool) -> None:
    link, target = Path(cfg["paths"]["llava_images"]) / cs.IMAGE_PREFIX, Path(cfg["coco_study"]["images_dir"])
    if not link.exists() and not link.is_symlink() and create:
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target.resolve(), link)
        print(f"created symlink {link} -> {target.resolve()}")
    if not link.exists():
        raise SystemExit(f"{link} is missing: exposed COCO images are read as {cs.IMAGE_PREFIX}/<file>. Rerun with --link-coco, or: ln -sfn {target.resolve()} {link}")
    if link.resolve() != target.resolve():
        raise SystemExit(f"{link} resolves to {link.resolve()}, not coco_study.images_dir {target.resolve()}")


def _write_once(obj, path: Path, indent: int | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".new")
    write_json(obj, tmp, indent=indent)
    sha = file_sha256(tmp)
    if path.exists() and file_sha256(path) != sha:
        tmp.unlink()
        raise SystemExit(f"{path} exists with different content (sha256 {file_sha256(path)}, new {sha}); move it away to rebuild")
    if path.exists():
        tmp.unlink()
    else:
        tmp.replace(path)
    return sha


def build(cfg: dict, levels, halves, slots, d: int, M: int, out: Path, lock_path=DEFAULT_LOCK, check_images: bool = True) -> dict:
    seeds = list(cfg["training"]["seeds"])
    if not slots or not all(0 <= s < len(seeds) for s in slots):
        raise SystemExit(f"seed slots must be in 0..{len(seeds) - 1}, got {slots}")
    required = [POOL_FILE, "general_order.jsonl"]
    try:
        lock = verify(cfg, lock_path, allow_unlocked=True, required=required)
    except RuntimeError as e:
        raise SystemExit(str(e))
    lock_sha = file_sha256(lock_path)
    splits_dir = Path(cfg["paths"]["splits_dir"])
    pool = read_json(splits_dir / POOL_FILE)
    cs.validate_pool(pool)
    if lock["files"][POOL_FILE].get("content_hash") != pool["content_hash"]:
        raise SystemExit(f"{POOL_FILE} content hash differs from the lock")
    caps, vqa_by_image, _ = cs.load_inputs(cfg, pool)
    prompts = load_prompts()
    order = list(read_jsonl(splits_dir / "general_order.jsonl"))
    mix = read_json(cfg["paths"]["llava_mix"])
    eval_splits = {"coco": {"path": POOL_FILE, "content_hash": pool["content_hash"]}}
    runs, cells = [], []
    for level in levels:
        for half in halves:
            ex = cs.build_exposure(level, half, pool, caps, vqa_by_image, d, prompts)
            K = ex["meta"]["K"]
            if not 0 < K < M:
                raise SystemExit(f"K = {K} must be in 1..M-1 (M={M})")
            exposed = cs.expand(ex)
            if check_images and level != "Cap":
                missing = find_missing_images([{"image": r["image"]} for r in ex["rows"]], cfg["paths"]["llava_images"])
                if missing:
                    raise SystemExit(f"{level} {half}: {len(missing)} exposed images are missing under {cfg['paths']['llava_images']} (--link-coco)")
            stem = f"COCO_{level}_{half}_d{d}_M{M}"
            exp_path, data_path = out / "data" / f"{stem}.exposure.json", out / "data" / f"{stem}.json"
            _write_once(ex, exp_path)
            data_sha = _write_once([mix[o["src_index"]] for o in order[: M - K]] + exposed, data_path)
            ids = [r["coco_image_id"] for r in ex["rows"]]
            exposure = {"level": level, "half": half, "n_exp": len(ids), "d": d, "K": K, "exposure_ids_sha256": content_hash(sorted(ids)),
                        "exposure_content_hash": ex["content_hash"], "exposure_path": str(exp_path.resolve()), "exposure_sha256": file_sha256(exp_path),
                        "prompts": ex["meta"]["prompts"], "uninformative": ex["meta"]["uninformative"], "vqa_suffix": ex["meta"]["vqa_suffix"]}
            cells.append({"level": level, "half": half, "K": K, "general": M - K, "data": str(data_path), "data_sha256": data_sha})
            for k in slots:
                spec = {"gate": "C", "run_id": cs.run_id(level, half, d, k), "condition": f"COCO_{level}", "level": level, "half": half, "d": d, "K": K,
                        "M": M, "N_exp": len(ids), "split": f"coco_{half}", "seed_slot": k, "train_seed": int(seeds[k]),
                        "data": {"path": str(data_path.resolve()), "sha256": data_sha, "n_examples": M, "general_prefix": M - K, "exposed": K},
                        "exposure": exposure, "model": cfg["models"]["main"], "manifest_lock_sha256": lock_sha,
                        "manifest_entries": {n: lock["files"][n]["sha256"] for n in required}, "eval_splits": eval_splits}
                _write_once(spec, out / "C" / f"{spec['run_id']}.json", indent=1)
                runs.append(spec["run_id"])
    return {"runs": runs, "cells": cells, "M": M, "d": d, "pool_content_hash": pool["content_hash"], "manifest_lock_sha256": lock_sha}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--levels", nargs="+", default=list(cs.LEVELS), choices=list(cs.LEVELS))
    ap.add_argument("--halves", nargs="+", default=list(cs.HALVES), choices=list(cs.HALVES))
    ap.add_argument("--slots", nargs="+", type=int, default=[0])
    ap.add_argument("--d", type=int, default=None, help="default: coco_study.d")
    ap.add_argument("--M", type=int, default=None, help="default: coco_study.M")
    ap.add_argument("--link-coco", action="store_true", help=f"create <llava_images>/{cs.IMAGE_PREFIX} -> coco_study.images_dir if missing")
    ap.add_argument("--out", default=None, help="default: <outputs_dir>/runs")
    ap.add_argument("--lock", default=str(DEFAULT_LOCK))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    d, M = args.d or int(cfg["coco_study"]["d"]), args.M or int(cfg["coco_study"]["M"])
    if set(args.levels) - {"Cap"}:
        ensure_coco_link(cfg, args.link_coco)
    out = Path(args.out) if args.out else Path(cfg["paths"]["outputs_dir"]) / "runs"
    print(json.dumps(build(cfg, args.levels, args.halves, args.slots, d, M, out, args.lock), indent=1))


if __name__ == "__main__":
    main()
