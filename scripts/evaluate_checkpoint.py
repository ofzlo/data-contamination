#!/usr/bin/env python
"""Evaluate one checkpoint's T=0 generations: S/A/V/Q per group and reference configuration.

Optionally adds teacher-forced V metrics (``--logprobs`` from scripts/score_refs.py) and paired
comparisons against the d=0 baseline (``--baseline``, one metrics JSON per d=0 seed; pooled).
Every image is scored; group summaries and comparisons leave out the B1 analysis exclusion
(``analysis.exclude_image_ids_file``) unless ``--include-b1-overlap`` (robustness view, not gate-valid).
The seen group is the exposure set of ``exposure.n_exp`` images (``--n-exp``; recorded as
``seen_definition``); seen_unexposed (the rest of the split's seen half) is reported alongside.

``--reuse-scores``: if ``--out`` already holds a complete evaluation of the same generations (same
checkpoint id, split, generations hash, tokenizer and provenance, newer than the generations and
log-probs files), its per-image CIDEr / SPICE scores are reused (no SPICE run, no GPU) and every other
field and every group statistic is recomputed, e.g. under a new seen definition or B1 exclusion.

Example:
    python scripts/evaluate_checkpoint.py --checkpoint-id paired_d64_s0 \
        --generations outputs/gens/paired_d64_s0.jsonl --split data/splits/split_A.json \
        --logprobs outputs/logprobs/paired_d64_s0.jsonl \
        --baseline outputs/metrics/d0_s{0,1,2,3,4}.json --out outputs/metrics/paired_d64_s0.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from contam.config import content_hash, load_config, read_json, write_json
from contam.data.freeze import add_check_args, cli_check
from contam.data.nocaps import load_annotations
from contam.eval.evaluate import (
    REPORT_GROUPS,
    compare,
    evaluate_checkpoint,
    generations_provenance,
    load_generations,
    load_analysis_exclusion,
    load_logprobs,
    load_seen_definition,
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
    ap.add_argument("--n-exp", type=int, default=None, help="seen group = the first N_exp exposed images of the split (default: exposure.n_exp; C2 arms: 563 / 141)")
    ap.add_argument("--reuse-scores", action="store_true", help="reuse the CIDEr / SPICE per-image scores of an existing --out made from the same generations")
    add_check_args(ap)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    frozen = cli_check(args, cfg, args.split, "evaluate_checkpoint")
    ann = load_annotations(args.annotations or cfg["paths"]["annotations"])
    split = read_json(args.split)
    excl = load_analysis_exclusion(cfg, args.include_b1_overlap)
    seen = load_seen_definition(cfg, split, ann, args.n_exp)
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
    reuse = reusable_scores(args, split, gens, prov, lp is not None, frozen) if args.reuse_scores else None
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
        seen=seen,
        reuse_q=reuse,
    )
    res["prompt_id"] = args.prompt_id
    res["provenance"] = prov
    res.update(frozen)
    mark_validity(res, res["complete"] and frozen["manifest_split_sha256"] is not None, excl)
    if args.baseline:
        st = cfg["stats"]
        base = [read_json(p) for p in args.baseline]
        res["vs_d0"] = compare([res], base, split, n_boot=st["n_boot"], seed=st["boot_seed"], ci=st["ci"], allow_partial=args.allow_partial, exclusion=excl, seen=seen)
        res["vs_d0"]["baseline_files"] = list(args.baseline)
    write_json(res, args.out)
    s = res["summary"]
    keys = [k for k in ("cider_heldout", "spice_heldout", "cider_all10", "word_count", "boilerplate", "exact_match_train", "rmi_sum") if k in s["all"]]
    n_ex = res["analysis_exclusion"]["n_excluded"]["all"]
    print(f"{args.checkpoint_id}: n={res['n_images']} complete={res['complete']} B1 overlap ({n_ex} images) {'INCLUDED (robustness)' if args.include_b1_overlap else 'excluded from group means'}; "
          f"seen = exposure set N_exp={seen['n_exp']} ({seen['ids_sha256'][:12]}){'; CIDEr / SPICE reused from ' + args.out if reuse is not None else ''} -> {args.out}")
    for g in (*REPORT_GROUPS, "all"):
        if s[g]["n"]:
            print(f"  {g:14s} n={s[g]['n']:5d} " + " ".join(f"{k}={s[g][k]:.3f}" for k in keys))


def reusable_scores(args, split: dict, gens: dict, prov: dict, has_lp: bool, frozen: dict) -> dict | None:
    """Per-image rows of an existing ``--out`` evaluation of these generations (None, with the reason
    printed, if there is none to reuse): the CIDEr / SPICE fields are taken from them."""
    out = Path(args.out)
    if not out.exists():
        print(f"--reuse-scores: no {out}; scoring from scratch")
        return None
    old = read_json(out)
    ids = sorted(gens)
    why = None
    if old.get("checkpoint_id") != args.checkpoint_id:
        why = f"checkpoint_id {old.get('checkpoint_id')!r}"
    elif old.get("split", {}).get("content_hash") != split["content_hash"]:
        why = "another split"
    elif old.get("generations_hash") != content_hash({str(i): gens[i] for i in ids}):
        why = "other generations"
    elif old.get("tokenizer") != args.tokenizer or old.get("prompt_id") != args.prompt_id or old.get("has_logprobs") is not has_lp:
        why = "another tokenizer / prompt / log-prob setting"
    elif old.get("provenance") != prov:
        why = "other generation / log-prob provenance"
    elif not old.get("complete") and not args.allow_partial:
        why = "a partial image set"
    elif old.get("manifest_split_sha256") != frozen["manifest_split_sha256"]:
        why = "another manifest lock entry"
    elif any(out.stat().st_mtime < Path(p).stat().st_mtime for p in (args.generations, args.logprobs) if p):
        why = "older than its generations / log-probs"
    if why:
        print(f"--reuse-scores: {out} was made from {why}; scoring from scratch")
        return None
    return {int(k): v for k, v in old["per_image"].items()}


if __name__ == "__main__":
    main()
