#!/usr/bin/env python
"""B3 prompt intervention: how much CIDEr / SPICE move from prompt-induced style alone.

Models: the untrained base model and the d=0 model(s) (base + LoRA adapter). Prompts: the B3 list
in configs/prompts.yaml (neutral, nocaps_like, verbose; wording owned by Kaustav, placeholders
are marked there). T=0 generation, then Q (CIDEr/SPICE, all reference configs) and S per
(model, prompt), and the paired change of every prompt vs. ``neutral`` with bootstrap CIs.

Example:
    python scripts/b3_prompt_intervention.py generate --split data/splits/split_A.json \
        --model-tag base --model-tag d0_s0=/scratch/.../d0_s0
    python scripts/b3_prompt_intervention.py evaluate --split data/splits/split_A.json \
        --model-tag base --model-tag d0_s0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from contam.config import load_config, load_prompts, load_yaml, read_json, write_json
from contam.config import DEFAULT_PROMPTS
from contam.data.nocaps import ImageLoader, load_annotations
from contam.eval.evaluate import evaluate_checkpoint, load_generations
from contam.eval.stats import bootstrap_paired_diff

REPORT_FIELDS = ("cider_all10", "spice_all10", "cider_heldout", "spice_heldout", "word_count", "sentence_count", "boilerplate")


def parse_tags(tags):
    out = {}
    for t in tags:
        name, _, adapter = t.partition("=")
        out[name] = adapter or None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["generate", "evaluate"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--model-tag", action="append", required=True, help="NAME or NAME=ADAPTER_DIR (no adapter = untrained base)")
    ap.add_argument("--model", default=None, help="HF base model id (default: config models.main)")
    ap.add_argument("--out-dir", default=None, help="default: <outputs_dir>/b3")
    ap.add_argument("--max-images", type=int, default=None, help="first N split images (smoke tests; Q is then not comparable to full-set numbers)")
    ap.add_argument("--tokenizer", default="ptb", choices=["ptb", "simple"])
    ap.add_argument("--metrics", nargs="*", default=["cider", "spice"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    prompts = load_prompts(args.prompts)
    b3_ids = load_yaml(args.prompts or DEFAULT_PROMPTS)["b3"]["prompt_ids"]
    out_dir = Path(args.out_dir or cfg["paths"]["outputs_dir"] / "b3")
    ann = load_annotations(cfg["paths"]["annotations"])
    split = read_json(args.split)
    ids = sorted(set(split["seen"]) | set(split["unseen"]))[: args.max_images]
    tags = parse_tags(args.model_tag)

    if args.phase == "generate":
        from contam.generate import run_generation
        from contam.models import HFVLM

        loader = ImageLoader(ann, cfg["paths"]["images_dir"])
        g = cfg["generation"]
        for name, adapter in tags.items():
            vlm = HFVLM(args.model or cfg["models"]["main"], adapter=adapter, dtype=cfg["models"]["dtype"], end_token=cfg["models"]["end_token"])
            for pid in b3_ids:
                run_generation(vlm, loader, ids, pid, prompts[pid], out_dir / f"{name}.jsonl", name, 0.0, batch_size=g["batch_size"], max_new_tokens=g["max_new_tokens"])
            del vlm
        return

    st = cfg["stats"]
    report = {"split": split["name"], "n_images": len(ids), "prompts": {p: prompts[p] for p in b3_ids}, "models": {}}
    for name in tags:
        per_prompt = {}
        for pid in b3_ids:
            gens = load_generations(out_dir / f"{name}.jsonl", pid)
            gens = {i: gens[i] for i in ids}
            per_prompt[pid] = evaluate_checkpoint(
                gens, ann, split, boilerplate_patterns=cfg["style"]["boilerplate_patterns"], metrics=args.metrics, tokenizer=args.tokenizer,
                spice_cache=cfg["paths"]["spice_cache"], require_all=args.max_images is None, checkpoint_id=f"{name}/{pid}",
            )
        rows = {pid: {f: per_prompt[pid]["summary"]["all"].get(f) for f in REPORT_FIELDS} for pid in b3_ids}
        diffs = {}
        if "neutral" in per_prompt:
            ref = per_prompt["neutral"]["per_image"]
            for pid in b3_ids:
                if pid == "neutral":
                    continue
                cur = per_prompt[pid]["per_image"]
                diffs[f"{pid}-neutral"] = {
                    f: bootstrap_paired_diff(np.array([cur[str(i)][f] for i in ids]), np.array([ref[str(i)][f] for i in ids]), n_boot=st["n_boot"], seed=st["boot_seed"], ci=st["ci"]).to_dict()
                    for f in REPORT_FIELDS if f in ref[str(ids[0])]
                }
        report["models"][name] = {"summary_all": rows, "vs_neutral": diffs}
        print(f"== {name}")
        print("  prompt       " + " ".join(f"{f:>14s}" for f in REPORT_FIELDS))
        for pid in b3_ids:
            print(f"  {pid:12s} " + " ".join(f"{rows[pid][f]:14.3f}" if rows[pid][f] is not None else f"{'-':>14s}" for f in REPORT_FIELDS))
        for k, d in diffs.items():
            print(f"  {k}: " + "; ".join(f"{f} {v['estimate']:+.2f} [{v['ci_lo']:+.2f},{v['ci_hi']:+.2f}]" for f, v in d.items() if f.startswith(("cider", "spice"))))
    write_json(report, out_dir / "b3_report.json")
    print(f"-> {out_dir / 'b3_report.json'}")


if __name__ == "__main__":
    main()
