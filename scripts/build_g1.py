#!/usr/bin/env python
"""Prepare exposed (d > 0) runs, e.g. G1's cells: training data + one run spec per (condition, d, seed slot).
Does NOT start training.

A cell with K = N_exp x d exposed examples trains on the first M - K examples of the frozen general
order (``general_order.jsonl``) followed by the K exposed ones (``contam.data.exposure``): exposure
replaces the tail of the d=0 data (prereg Section 2, D2). The trainer shuffles with the training
seed, so the position in the file is bookkeeping only. Every condition and d uses the same N_exp
exposed images of the split (``exposure_ids``, nested with the stored exposure sets); each unique
exposed row is repeated d times. Caption / Paired targets are the split's training references,
SameDist rows come from ``samedist_{split}.json`` (D4), Image targets from the base model's T=0
captions under p_cap (``--base-captions``, a ``contam.generate`` file of the untrained base).

The data file depends on (split, condition, d, M, N_exp) only, so every seed slot of a cell shares
it; the seed slot sets ``train_seed = training.seeds[slot]``. Before writing anything this checks
the manifest lock (the split, its SameDist file and general_order.jsonl must be locked and
unchanged), the annotations (same hash as the split, prereg reference normalization) and that every
exposed image decodes under paths.llava_images. Exposed NoCaps images are read as
``nocaps/<file>`` under that root; ``--link-nocaps`` creates the symlink
``<llava_images>/nocaps -> <images_dir>`` if it is missing (idempotent).

Writes, under --out (default <outputs_dir>/runs):
    data/{split}_{cond}_d{d}_M{M}_nexp{N}.json            LLaVA-format training file (M examples)
    data/{split}_{cond}_d{d}_M{M}_nexp{N}.exposure.json   the exposed rows + meta (contam.data.exposure)
    {gate}/{gate}_{Condition}_d{d}_s{slot}.json           one run spec per cell and seed slot
    EX/EX_{Condition}_d{d}_n{N}_s{slot}.json              --exploratory: the same, outside every gate

``--exploratory`` (docs/deviations.md) allows doses outside the prereg grid (e.g. 2) and any N_exp. Its
specs get gate ``EX``, run ids ``EX_{Condition}_d{d}_n{N_exp}_s{slot}`` (never parsed by
``contam.eval.gates.parse_run_id``, so no gate accepts them) and ``"exploratory": true``. Without it,
doses must be in the prereg grid and N_exp must be ``exposure.n_exp`` or a stored exposure-set size.

Example (G1's d=4 cells, first seed slot):
    python scripts/build_g1.py --M 130000 --n-exp 1000 --conditions Caption Paired SameDist --doses 4 --slots 0 --link-nocaps
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from contam.config import content_hash, file_sha256, load_config, load_prompts, prompt_sha, read_json, read_jsonl, write_json
from contam.data.exposure import base_captions, build_exposure, expand
from contam.data.freeze import DEFAULT_LOCK, verify
from contam.data.nocaps import load_annotations, load_annotations_meta
from contam.data.samedist import validate_samedist
from contam.data.splits import check_annotations
from contam.train import find_missing_images

# run-id spelling (contam.eval.gates.CONDITION_NAMES) -> contam.data.exposure condition
CONDITIONS = {"Caption": "caption", "Paired": "paired", "SameDist": "samedist", "Image": "image", "Shuffled": "shuffled"}
DOSES = (1, 4, 16, 64)
EXPLORATORY_GATE = "EX"
NOCAPS_IMAGE_CONDITIONS = ("Paired", "Image", "Shuffled")  # read NoCaps images (SameDist reads COCO, Caption none)
# prereg Section 2: training targets are the normalized references of annotations.json
REF_NORMALIZATION = {"strip_whitespace": True, "strip_trailing_not_sign": True}


def ensure_nocaps_link(cfg: dict, create: bool) -> Path:
    """``<llava_images>/nocaps`` must resolve to paths.images_dir; create the symlink if asked and missing."""
    link, target = Path(cfg["paths"]["llava_images"]) / "nocaps", Path(cfg["paths"]["images_dir"])
    if not link.exists() and not link.is_symlink() and create:
        os.symlink(target.resolve(), link)
        print(f"created symlink {link} -> {target.resolve()}")
    if not link.exists():
        raise SystemExit(f"{link} is missing: exposed NoCaps images are read as nocaps/<file> under {link.parent}. "
                         f"Rerun with --link-nocaps, or: ln -sfn {target.resolve()} {link}")
    if link.resolve() != target.resolve():
        raise SystemExit(f"{link} resolves to {link.resolve()}, not paths.images_dir {target.resolve()}")
    return link


def _write_once(obj, path: Path, indent: int | None = None) -> str:
    """Write ``obj`` to ``path`` unless a different file is already there (a run may train on it or record
    it); identical content is a no-op. Returns its sha256."""
    tmp = path.with_name(path.name + ".new")
    write_json(obj, tmp, indent=indent)
    sha = file_sha256(tmp)
    if path.exists() and file_sha256(path) != sha:
        tmp.unlink()
        raise SystemExit(f"{path} exists with different content (sha256 {file_sha256(path)}, new {sha}); move it away to rebuild")
    if path.exists():
        tmp.unlink()  # identical: leave the existing file (and its mtime) alone
    else:
        tmp.replace(path)
    return sha


def build(cfg: dict, split_name: str, conditions, doses, slots, M: int, n_exp: int, out: Path, gate: str = "G1",
          lock_path=DEFAULT_LOCK, base_captions_path=None, check_images: bool = True, exploratory: bool = False) -> dict:
    if exploratory:
        gate = EXPLORATORY_GATE
    elif gate == EXPLORATORY_GATE:
        raise SystemExit(f"gate {EXPLORATORY_GATE} is reserved for --exploratory runs")
    conditions, doses, slots = list(conditions), sorted(set(int(d) for d in doses)), sorted(set(int(s) for s in slots))
    bad = [c for c in conditions if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"unknown conditions {bad}; one of {list(CONDITIONS)}")
    if not doses or min(doses) < 1:
        raise SystemExit(f"doses must be >= 1 (d=0 is scripts/build_g0.py), got {doses}")
    seeds = list(cfg["training"]["seeds"])
    if not slots or not all(0 <= s < len(seeds) for s in slots):
        raise SystemExit(f"seed slots must be in 0..{len(seeds) - 1}, got {slots}")

    split_file, sd_file = f"split_{split_name}.json", f"samedist_{split_name}.json"
    required = [split_file, "general_order.jsonl"] + ([sd_file] if "SameDist" in conditions else [])
    try:
        lock = verify(cfg, lock_path, allow_unlocked=True, required=required)
    except RuntimeError as e:
        raise SystemExit(str(e))
    lock_sha = file_sha256(lock_path)
    splits_dir = Path(cfg["paths"]["splits_dir"])
    split = read_json(splits_dir / split_file)
    if lock["files"][split_file].get("content_hash") != split["content_hash"]:
        raise SystemExit(f"{split_file} content hash differs from the lock")

    ann_path = cfg["paths"]["annotations"]
    norm = load_annotations_meta(ann_path).get("ref_normalization")
    if norm != REF_NORMALIZATION:
        raise SystemExit(f"{ann_path}: ref_normalization {norm} is not the prereg's {REF_NORMALIZATION}")
    annotations = load_annotations(ann_path)
    try:
        check_annotations(split, annotations)
    except ValueError as e:
        raise SystemExit(str(e))

    samedist = None
    if "SameDist" in conditions:
        samedist = read_json(splits_dir / sd_file)
        validate_samedist(samedist, split)
    base_caps = None
    if "Image" in conditions:
        if not base_captions_path:
            raise SystemExit("the Image condition needs --base-captions (the untrained base model's T=0 captions under p_cap)")
        base_caps = base_captions(base_captions_path, prompt_sha(load_prompts()["caption"]))
    prompts = load_prompts()

    order = list(read_jsonl(splits_dir / "general_order.jsonl"))
    if not 0 < M <= len(order):
        raise SystemExit(f"M={M} must be in 1..{len(order)} (general pool size)")
    for d in doses:
        if not 0 < n_exp * d < M:
            raise SystemExit(f"K = N_exp x d = {n_exp * d} must be in 1..M-1 (M={M})")
    mix = read_json(cfg["paths"]["llava_mix"])

    eval_splits = {split_name: {"path": split_file, "content_hash": split["content_hash"]}}
    data_dir, spec_dir = out / "data", out / gate
    cells, runs = [], []
    for cond in conditions:
        for d in doses:
            K = n_exp * d
            ex = build_exposure(CONDITIONS[cond], split, annotations, n_exp, d, prompts, shuffle_seed=cfg["exposure"]["shuffle_seed"],
                                samedist=samedist, base_caps=base_caps)
            exposed = expand(ex)
            if len(exposed) != K or ex["meta"]["K"] != K:
                raise AssertionError(f"{cond} d={d}: {len(exposed)} exposed examples, expected K={K}")
            if check_images:
                uniq = [{"image": r["image"]} for r in ex["rows"] if r.get("image")]
                missing = find_missing_images(uniq, cfg["paths"]["llava_images"])
                if missing:
                    raise SystemExit(f"{cond} d={d}: {len(missing)} exposed images are missing or unreadable under {cfg['paths']['llava_images']}, "
                                     f"e.g. {uniq[missing[0]]['image']} (NoCaps: --link-nocaps)")
            ids = [r["nocaps_image_id"] for r in ex["rows"]]
            stem = f"{split_name}_{CONDITIONS[cond]}_d{d}_M{M}_nexp{n_exp}"
            exp_path, data_path = data_dir / f"{stem}.exposure.json", data_dir / f"{stem}.json"
            _write_once(ex, exp_path)
            data_sha = _write_once([mix[o["src_index"]] for o in order[: M - K]] + exposed, data_path)
            exposure = {"condition": CONDITIONS[cond], "split": split_name, "n_exp": n_exp, "d": d, "K": K,
                        "exposure_ids_sha256": content_hash(sorted(ids)), "exposure_content_hash": ex["content_hash"],
                        "exposure_path": str(exp_path.resolve()), "exposure_sha256": file_sha256(exp_path), "prompts": ex["meta"]["prompts"],
                        "samedist": {"path": sd_file, "sha256": lock["files"][sd_file]["sha256"], "content_hash": samedist["content_hash"]} if cond == "SameDist" else None,
                        "shuffle_seed": ex["meta"]["shuffle_seed"], "base_captions_hash": ex["meta"]["base_captions_hash"]}
            cells.append({"condition": cond, "d": d, "K": K, "general": M - K, "n_examples": M, "data": str(data_path), "data_sha256": data_sha,
                          "data_bytes": data_path.stat().st_size, "exposure_ids_sha256": exposure["exposure_ids_sha256"]})
            for k in slots:
                spec = {
                    "gate": gate,
                    "run_id": f"{gate}_{cond}_d{d}_n{n_exp}_s{k}" if exploratory else f"{gate}_{cond}_d{d}_s{k}",
                    "condition": cond,
                    "d": d,
                    "K": K,
                    "M": M,
                    "N_exp": n_exp,
                    "split": split_name,
                    "seed_slot": k,
                    "train_seed": int(seeds[k]),
                    "data": {"path": str(data_path.resolve()), "sha256": data_sha, "n_examples": M, "general_prefix": M - K, "exposed": K},
                    "exposure": exposure,
                    "model": cfg["models"]["main"],
                    "manifest_lock_sha256": lock_sha,
                    "manifest_entries": {n: lock["files"][n]["sha256"] for n in required},
                    "eval_splits": eval_splits,
                    "eval_temperatures": [0.0, 1.0],
                }
                if exploratory:
                    spec["exploratory"] = True
                _write_once(spec, spec_dir / f"{spec['run_id']}.json", indent=1)  # same bytes as write_json's default
                runs.append(spec["run_id"])
    return {"runs": runs, "cells": cells, "M": M, "N_exp": n_exp, "split": split_name, "manifest_lock_sha256": lock_sha, "exploratory": exploratory}


def check_prereg_grid(cfg: dict, split_name: str, doses, n_exp: int) -> None:
    """Without --exploratory: doses from the prereg grid, N_exp = exposure.n_exp or a stored exposure-set size."""
    bad = sorted(set(int(d) for d in doses) - set(DOSES))
    if bad:
        raise SystemExit(f"doses {bad} are not in the prereg grid {list(DOSES)}; exploratory doses need --exploratory")
    stored = {len(v) for v in read_json(Path(cfg["paths"]["splits_dir"]) / f"split_{split_name}.json")["exposure_sets"].values()}
    allowed = {int(cfg["exposure"]["n_exp"])} | stored
    if n_exp not in allowed:
        raise SystemExit(f"N_exp={n_exp} is not exposure.n_exp or a stored exposure-set size {sorted(allowed)}; exploratory N_exp needs --exploratory")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--M", type=int, required=True, help="total examples per run (D2)")
    ap.add_argument("--n-exp", type=int, default=None, help="exposed images (D2; default: config exposure.n_exp)")
    ap.add_argument("--split", default="A")
    ap.add_argument("--conditions", nargs="+", default=["Caption", "Paired", "SameDist"], choices=list(CONDITIONS))
    ap.add_argument("--doses", nargs="+", type=int, default=[4, 64], help=f"prereg grid {list(DOSES)}; other doses need --exploratory")
    ap.add_argument("--slots", nargs="+", type=int, default=[0, 1], help="training seed slots")
    ap.add_argument("--gate", default="G1", help="run-id prefix and spec subdirectory")
    ap.add_argument("--exploratory", action="store_true", help=f"allow doses / N_exp outside the prereg grid; forces gate {EXPLORATORY_GATE} and run ids "
                    f"{EXPLORATORY_GATE}_<Condition>_d<d>_n<N_exp>_s<slot> (docs/deviations.md)")
    ap.add_argument("--base-captions", default=None, help="Image condition: contam.generate file of the untrained base model")
    ap.add_argument("--link-nocaps", action="store_true", help="create <llava_images>/nocaps -> <images_dir> if missing")
    ap.add_argument("--out", default=None, help="default: <outputs_dir>/runs")
    ap.add_argument("--lock", default=str(DEFAULT_LOCK))
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.n_exp is None:
        args.n_exp = int(cfg["exposure"]["n_exp"])
    if args.exploratory:
        if args.gate not in ("G1", EXPLORATORY_GATE):
            raise SystemExit(f"--exploratory runs always use gate {EXPLORATORY_GATE}, not {args.gate}")
    else:
        check_prereg_grid(cfg, args.split, args.doses, args.n_exp)
    if set(args.conditions) & set(NOCAPS_IMAGE_CONDITIONS):
        ensure_nocaps_link(cfg, args.link_nocaps)
    out = Path(args.out) if args.out else Path(cfg["paths"]["outputs_dir"]) / "runs"
    print(json.dumps(build(cfg, args.split, args.conditions, args.doses, args.slots, args.M, args.n_exp, out, args.gate, args.lock, args.base_captions,
                           exploratory=args.exploratory), indent=1))


if __name__ == "__main__":
    main()
