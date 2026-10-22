#!/usr/bin/env python
"""How much of the B3 CIDEr gap is the CIDEr-D length penalty? (input to the prereg Section 7 decision)

For each B3 prompt, scores the base model's T=0 captions against the 9 held-out references with:

* ``cider_d``: standard CIDEr-D (Gaussian length penalty, sigma = 6), as in the pipeline;
* ``cider_d_nolp``: CIDEr-D with the length penalty switched off (sigma -> inf; clipping kept);
* ``cider_d_first``: standard CIDEr-D on the first sentence of each caption only.

All 4,500 split images are scored as one corpus per variant (same IDF as the pipeline), and every
prompt is compared with ``neutral`` by the image-level paired bootstrap.

Example:
    python scripts/analysis/b3_length_control.py --split data/splits/split_A.json \
        --generations outputs/b3/base.jsonl --out outputs/b3/length_control.json
"""

from __future__ import annotations

import argparse
import re

import numpy as np

from contam.config import load_config, read_json, write_json
from contam.data.nocaps import load_annotations, refs_by_id
from contam.eval.caption_metrics import SCALE, refs_for_config, tokenize
from contam.eval.evaluate import load_generations
from contam.eval.stats import bootstrap_paired_diff

PROMPTS = ("neutral", "nocaps_like", "verbose")
NO_PENALTY_SIGMA = 1e9
_SENT_END = re.compile(r"(?<=[.!?])\s+")


def first_sentence(caption: str) -> str:
    return _SENT_END.split(" ".join(caption.split()), maxsplit=1)[0]


def cider_scores(hyps: dict[int, str], refs: dict[int, list[str]], sigma: float) -> dict[int, float]:
    """Per-image CIDEr-D (x100) of ``hyps`` scored as one corpus, with length-penalty width ``sigma``."""
    from pycocoevalcap.cider.cider import Cider

    ids = sorted(hyps)
    keys = {iid: f"{n:07d}" for n, iid in enumerate(ids)}
    tok = tokenize({**{"h" + keys[i]: [hyps[i]] for i in ids}, **{"r" + keys[i]: list(refs[i]) for i in ids}})
    gts = {keys[i]: tok["r" + keys[i]] for i in ids}
    res = {keys[i]: tok["h" + keys[i]] for i in ids}
    _, scores = Cider(sigma=sigma).compute_score(gts, res)
    by_key = dict(zip(gts.keys(), (float(s) for s in scores)))
    return {i: SCALE * by_key[keys[i]] for i in ids}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--generations", required=True, help="B3 generations JSONL (all prompts)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    st = cfg["stats"]
    split = read_json(args.split)
    ids = sorted(set(split["seen"]) | set(split["unseen"]))
    refs = refs_for_config({i: r for i, r in refs_by_id(load_annotations(cfg["paths"]["annotations"])).items() if i in set(ids)}, split["train_ref_idx"], "heldout")

    scores: dict[str, dict[str, np.ndarray]] = {}
    words: dict[str, dict[str, float]] = {}
    for pid in PROMPTS:
        gens = load_generations(args.generations, pid)
        gens = {i: gens[i] for i in ids}
        first = {i: first_sentence(c) for i, c in gens.items()}
        variants = {
            "cider_d": cider_scores(gens, refs, 6.0),
            "cider_d_nolp": cider_scores(gens, refs, NO_PENALTY_SIGMA),
            "cider_d_first": cider_scores(first, refs, 6.0),
        }
        scores[pid] = {v: np.array([s[i] for i in ids]) for v, s in variants.items()}
        words[pid] = {"full": float(np.mean([len(gens[i].split()) for i in ids])), "first": float(np.mean([len(first[i].split()) for i in ids]))}

    report = {"split": split["name"], "n_images": len(ids), "refs": "heldout", "words": words, "means": {}, "vs_neutral": {}}
    for pid in PROMPTS:
        report["means"][pid] = {v: float(a.mean()) for v, a in scores[pid].items()}
        if pid != "neutral":
            report["vs_neutral"][pid] = {
                v: bootstrap_paired_diff(scores[pid][v], scores["neutral"][v], n_boot=st["n_boot"], seed=st["boot_seed"], ci=st["ci"]).to_dict()
                for v in scores[pid]
            }
    write_json(report, args.out)

    print(f"{'prompt':12s} {'words':>6s} {'1st-sent':>8s} {'cider_d':>9s} {'no_lp':>9s} {'first':>9s}")
    for pid in PROMPTS:
        m = report["means"][pid]
        print(f"{pid:12s} {words[pid]['full']:6.1f} {words[pid]['first']:8.1f} {m['cider_d']:9.2f} {m['cider_d_nolp']:9.2f} {m['cider_d_first']:9.2f}")
    for pid, d in report["vs_neutral"].items():
        print(f"{pid}-neutral: " + "; ".join(f"{v} {x['estimate']:+.2f} [{x['ci_lo']:+.2f},{x['ci_hi']:+.2f}]" for v, x in d.items()))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
