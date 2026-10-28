#!/usr/bin/env python
"""Evaluate one checkpoint's T=0 generations: S/A/V/Q per group and reference configuration.

Optionally adds teacher-forced V metrics (``--logprobs`` from scripts/score_refs.py) and paired
comparisons against the d=0 baseline (``--baseline``, one metrics JSON per d=0 seed; pooled).
Every image is scored; group summaries and comparisons leave out the B1 analysis exclusion
(``analysis.exclude_image_ids_file``) unless ``--include-b1-overlap`` (robustness view, not gate-valid).

Example:
    python scripts/evaluate_checkpoint.py --checkpoint-id paired_d64_s0 \
        --generations outputs/gens/paired_d64_s0.jsonl --split data/splits/split_A.json \
        --logprobs outputs/logprobs/paired_d64_s0.jsonl \
        --baseline outputs/metrics/d0_s{0,1,2,3,4}.json --out outputs/metrics/paired_d64_s0.json
"""

from __future__ import annotations

import argparse

from contam.config import load_config, read_json, write_json
from contam.data.freeze import add_check_args, cli_check
from contam.data.nocaps import load_annotations
from contam.eval.evaluate import (
    compare,
    evaluate_checkpoint,
    generations_provenance,
    load_generations,
    load_analysis_exclusion,
    load_logprobs,
    logprobs_provenance,
    mark_validity,
    warn_partial,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint-id", required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--prompt-id", default="caption")
    ap.add_argument("--logprobs", default=None, help="teacher-forced log-probs JSONL for this checkpoint")
    ap.add_argument("--baseline", nargs="*", default=[], help="d=0 metrics JSON(s) (one per seed) for paired comparisons")
    ap.add_argument("--metrics", nargs="*", default=["cider", "spice"])
    ap.add_argument("--tokenizer", default="ptb", choices=["ptb", "simple"])
    ap.add_argument("--no-spice-cache", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="allow generations for a subset of split images (smoke tests only)")
    ap.add_argument("--pos", action="store_true", help="add spaCy POS mix to S")
    ap.add_argument("--include-b1-overlap", action="store_true", help="robustness view: keep the B1 analysis exclusion images in every result; output is NOT gate-valid")
    add_check_args(ap)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    frozen = cli_check(args, cfg, args.split, "evaluate_checkpoint")
    ann = load_annotations(args.annotations or cfg["paths"]["annotations"])
    split = read_json(args.split)
    excl = load_analysis_exclusion(cfg, args.include_b1_overlap)
    if args.allow_partial:
        warn_partial("evaluate_checkpoint")
    gens = load_generations(args.generations, args.prompt_id, checkpoint_id=args.checkpoint_id)
    prov = {"generations": generations_provenance(args.generations, args.prompt_id)}
    lp = None
    if args.logprobs:
        lp = load_logprobs(args.logprobs, args.prompt_id, checkpoint_id=args.checkpoint_id)
        prov["logprobs"] = logprobs_provenance(args.logprobs, args.prompt_id)
        for f in ("prompt_sha", "model", "adapter"):
            g, l = prov["generations"][f], prov["logprobs"][f]
            if g != l:
                raise SystemExit(f"generations and log-probs disagree on {f}: {g!r} vs {l!r}")
    res = evaluate_checkpoint(
        gens,
        ann,
        split,
        logprobs=lp,
        boilerplate_patterns=cfg["style"]["boilerplate_patterns"],
        metrics=args.metrics,
        tokenizer=args.tokenizer,
        spice_cache=None if args.no_spice_cache else cfg["paths"]["spice_cache"],
        require_all=not args.allow_partial,
        checkpoint_id=args.checkpoint_id,
        with_pos=args.pos,
        exclusion=excl,
    )
    res["prompt_id"] = args.prompt_id
    res["provenance"] = prov
    res.update(frozen)
    mark_validity(res, res["complete"] and frozen["manifest_split_sha256"] is not None, excl)
    if args.baseline:
        st = cfg["stats"]
        base = [read_json(p) for p in args.baseline]
        res["vs_d0"] = compare([res], base, split, n_boot=st["n_boot"], seed=st["boot_seed"], ci=st["ci"], allow_partial=args.allow_partial, exclusion=excl)
        res["vs_d0"]["baseline_files"] = list(args.baseline)
    write_json(res, args.out)
    s = res["summary"]
    keys = [k for k in ("cider_heldout", "spice_heldout", "cider_all10", "word_count", "boilerplate", "exact_match_train", "rmi_sum") if k in s["all"]]
    n_ex = res["analysis_exclusion"]["n_excluded"]["all"]
    print(f"{args.checkpoint_id}: n={res['n_images']} complete={res['complete']} B1 overlap ({n_ex} images) {'INCLUDED (robustness)' if args.include_b1_overlap else 'excluded from group means'} -> {args.out}")
    for g in ("seen", "unseen", "all"):
        print(f"  {g:7s} n={s[g]['n']:5d} " + " ".join(f"{k}={s[g][k]:.3f}" for k in keys))


if __name__ == "__main__":
    main()
