#!/usr/bin/env python
"""COCO image-study report: per-level Leak with CIs, per-model accuracies, POPE F1 (``contam.eval.coco_eval``).

Each ``--level NAME:RUN_A,RUN_B`` pairs the model trained on half A with the one trained on half B, for
example ``L3:C_L3_A_d16_s0,C_L3_B_d16_s0``. Scores are read from
``<scores-dir>/<run>.{vqa,pope}.jsonl`` (``scripts/coco_eval_score.py``).

Per level and cell:
* VQAv2 cells: mentioned / unmentioned / yes/no / number / all;
* POPE cells: positives mentioned / unmentioned, each negative type, all;
* Leak = ½[(acc(M_A, A) − acc(M_B, A)) + (acc(M_B, B) − acc(M_A, B))], with an image-level bootstrap CI
  within each half (B = 10,000, seed 0).

Every input must have been scored on the same question file.

Sanity check (guide Y-4 / M-4): ``--level base:base,base`` must give Leak = 0 exactly.

Example:
    python scripts/coco_leak_report.py --level L3:C_L3_A_d16_s0,C_L3_B_d16_s0 --level L2:C_L2_A_d16_s0,C_L2_B_d16_s0 \
        --level L1:C_L1_A_d16_s0,C_L1_B_d16_s0 --level Cap:C_Cap_A_d16_s0,C_Cap_B_d16_s0 --out outputs/coco_eval/leak_report.json
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from contam.config import load_config, write_json
from contam.eval import coco_eval as ce
from contam.eval import yesno


def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}"


def main(argv=None) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from coco_eval_score import load_question_file

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--questions", default="outputs/coco/coco_questions.json")
    ap.add_argument("--scores-dir", default="outputs/coco_eval")
    ap.add_argument("--level", action="append", required=True, help="NAME:RUN_A,RUN_B")
    ap.add_argument("--tasks", nargs="+", default=["vqa", "pope"], choices=["vqa", "pope"])
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    qf = load_question_file(args.questions, cfg)
    sha, sd = qf["content_hash"], Path(args.scores_dir)
    vqa_q, pope_q = qf["vqav2"], qf["pope"]

    report = {"questions": args.questions, "questions_content_hash": sha, "n_boot": args.n_boot, "boot_seed": args.boot_seed, "ci": args.ci, "levels": {}}
    lines = []
    for spec in args.level:
        name, runs = spec.split(":", 1)
        run_a, run_b = runs.split(",")
        lv = {"run_a": run_a, "run_b": run_b}
        if "vqa" in args.tasks:
            ca, _ = ce.load_vqa(sd / f"{run_a}.vqa.jsonl", vqa_q, sha)
            cb, _ = ce.load_vqa(sd / f"{run_b}.vqa.jsonl", vqa_q, sha)
            lv["vqa"] = {c: ce.leak(ca, cb, vqa_q, c, args.n_boot, args.boot_seed, args.ci) for c in (*ce.VQA_CELLS, None)}
            lv["vqa"]["all"] = lv["vqa"].pop(None)
        if "pope" in args.tasks:
            sa, _ = yesno.load_scores(sd / f"{run_a}.pope.jsonl", pope_q, sha)
            sb, _ = yesno.load_scores(sd / f"{run_b}.pope.jsonl", pope_q, sha)
            ca, cb = ce.pope_correct(sa, pope_q), ce.pope_correct(sb, pope_q)
            lv["pope"] = {c: ce.leak(ca, cb, pope_q, c, args.n_boot, args.boot_seed, args.ci) for c in (*ce.POPE_CELLS, None)}
            lv["pope"]["all"] = lv["pope"].pop(None)
            lv["pope_f1"] = {run: {h: ce.pope_f1(s, pope_q, h) for h in ("A", "B")} for run, s in ((run_a, sa), (run_b, sb))}
        report["levels"][name] = lv
        for task in ("vqa", "pope"):
            for cell, e in lv.get(task, {}).items():
                if e is None:
                    continue
                a = e["accuracy"]
                lines.append(f"{name:5s} {task:4s} {cell:16s} Leak={_f(e['estimate'])} [{_f(e['ci_lo'])}, {_f(e['ci_hi'])}]   "
                             f"acc A: M_A {a['M_A_on_A']:.3f} M_B {a['M_B_on_A']:.3f} | B: M_B {a['M_B_on_B']:.3f} M_A {a['M_A_on_B']:.3f}   "
                             f"(q A {e['n_questions']['A']}, B {e['n_questions']['B']})")
    write_json(report, args.out)
    print("Leak = 1/2[(acc(M_A,A) - acc(M_B,A)) + (acc(M_B,B) - acc(M_A,B))], image-level bootstrap within each half")
    print("\n".join(lines))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
