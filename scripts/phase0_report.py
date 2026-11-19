#!/usr/bin/env python
"""Phase 0 report: preregistered yes/no effects per run cell and the primary verdict (``contam.eval.yesno``).

Inputs are the per-checkpoint score files of ``scripts/phase0_score.py``: the five d=0 seeds
(``--d0``) and any exposed runs (``--runs``). Runs are grouped into cells by id (``<cell>_s<seed>``);
each cell uses the question file's group column of its N_exp (``_n<N>`` runs, else 1000), and the
margins δ of that column from the d=0 seeds. All files must share the question file, prompt, model,
batch size and token ids. The verdict is read only from ``yesno.PREREG_PRIMARY``; every other cell
is reported, not decided.

Example:
    python scripts/phase0_report.py --d0 outputs/phase0/G0_d0_s{0,1,2,3,4}.jsonl \
        --runs outputs/phase0/G1_*.jsonl outputs/phase0/G2_*.jsonl outputs/phase0/EX_*.jsonl \
        --out outputs/phase0/report.json
    python scripts/phase0_report.py --d0 outputs/phase0/G0_d0_s{0,1,2,3,4}.jsonl --out outputs/phase0/margins.json   # d=0 only: margins
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict

import numpy as np

from contam.config import resolve_path, write_json
from contam.eval import yesno

CHECK_FIELDS = ("model", "prompt_sha", "questions_sha256", "batch_size", "yes_ids", "no_ids")


def _load(paths, questions, qsha):
    out = {}
    for p in paths:
        scores, prov = yesno.load_scores(p, questions, qsha)
        cid = prov["checkpoint_id"]
        if cid in out:
            raise SystemExit(f"checkpoint {cid} given twice")
        out[cid] = (scores, prov, str(p))
    return out


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Phase 0 preregistered yes/no report.")
    ap.add_argument("--questions", default=yesno.PREREG_QUESTIONS)
    ap.add_argument("--questions-sha256", default=yesno.PREREG_QUESTIONS_SHA256)
    ap.add_argument("--d0", nargs="+", required=True, help="the five d=0 seeds' score files")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--primary", default=yesno.PREREG_PRIMARY)
    ap.add_argument("--n-boot", type=int, default=yesno.PREREG_N_BOOT)
    ap.add_argument("--boot-seed", type=int, default=yesno.PREREG_BOOT_SEED)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--exclude-body-parts", action="store_true", help="robustness view: also leave out the body-part yes questions (not the primary analysis)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    all_questions, qsha = yesno.load_questions(resolve_path(args.questions), args.questions_sha256)
    d0 = _load(args.d0, all_questions, qsha)
    runs = _load(args.runs, all_questions, qsha)
    questions = yesno.analysis_questions(all_questions, exclude_body_parts=args.exclude_body_parts)
    if sorted(d0) != [f"G0_d0_s{k}" for k in range(5)]:
        raise SystemExit(f"--d0 must be G0_d0_s0..s4, got {sorted(d0)}")
    if set(d0) & set(runs):
        raise SystemExit(f"in both --d0 and --runs: {sorted(set(d0) & set(runs))}")
    allp = [v[1] for v in list(d0.values()) + list(runs.values())]
    for f in CHECK_FIELDS:
        vals = {repr(p[f]) for p in allp}
        if len(vals) > 1:
            raise SystemExit(f"score files differ in {f}: {sorted(vals)}")
    adapters = [p["adapter"] for p in allp]
    if None in adapters or len(set(adapters)) != len(adapters):
        raise SystemExit("every checkpoint needs its own adapter (no base model, no adapter used twice)")

    d0_scores = {k: v[0] for k, v in d0.items()}
    d0_pooled = {q["qid"]: float(np.mean([s[q["qid"]] for s in d0_scores.values()])) for q in questions}
    cells = defaultdict(dict)
    for cid, (scores, _, _) in runs.items():
        name, seed = yesno.run_cell(cid)
        cells[name][seed] = scores

    report = {"prereg": {"questions": str(args.questions), "questions_sha256": qsha, "prompt_template": yesno.PROMPT_TEMPLATE,
                         **{f: allp[0][f] for f in CHECK_FIELDS if f != "questions_sha256"},
                         "primary": args.primary, "exclude_body_part_no": yesno.PREREG_EXCLUDE_BODY_PART_NO, "exclude_body_parts": args.exclude_body_parts, "n_questions": len(questions), "n_boot": args.n_boot, "boot_seed": args.boot_seed, "ci": args.ci},
              "inputs": {k: v[2] for k, v in list(d0.items()) + list(runs.items())},
              "answer_mass": {k: {"mean": v[1]["mean_mass"], "min": v[1]["min_mass"]} for k, v in list(d0.items()) + list(runs.items())},
              "margins": {}, "d0_accuracy": {}, "cells": {}, "primary": None}
    low = {k: m["mean"] for k, m in report["answer_mass"].items() if m["mean"] < 0.9}
    if low:
        print(f"WARNING: mean P(yes) + P(no) below 0.9 (the model often answers with another token): {low}", file=sys.stderr)
    columns = sorted({yesno.n_exp_column(f"{c}_s0") for c in cells} | {"1000"}, key=int, reverse=True)
    for col in columns:
        report["margins"][col] = yesno.margins(questions, d0_scores, col)
        acc = {}
        for cell in yesno.CELLS:
            for g in ("seen", "unseen"):
                qs = [q for q in questions if q["cell"] == cell and q["group"][col] == g]
                acc[f"{cell}|{g}"] = {k: yesno.accuracy(s, qs) for k, s in d0_scores.items()}
        report["d0_accuracy"][col] = acc

    lines = []
    for name in sorted(cells):
        seeds = cells[name]
        col = yesno.n_exp_column(f"{name}_s0")
        eff = {}
        for cell in yesno.CELLS:
            eff[cell] = yesno.cell_effects(questions, {str(k): v for k, v in sorted(seeds.items())}, d0_pooled, col, cell,
                                           args.n_boot, args.boot_seed, args.ci)
        m = report["margins"][col]
        acc = {}
        for cell in yesno.CELLS:
            for g in ("seen", "unseen"):
                qs = [q for q in questions if q["cell"] == cell and q["group"][col] == g]
                acc[f"{cell}|{g}"] = {str(k): yesno.accuracy(s, qs) for k, s in sorted(seeds.items())}
        entry = {"column": col, "seeds": sorted(seeds), "effects": eff, "accuracy": acc,
                 "direction": {cell: ("+" if yesno.directional_pass(eff[cell], m[cell]["delta"], +1) else
                                      "-" if yesno.directional_pass(eff[cell], m[cell]["delta"], -1) else "0") for cell in yesno.CELLS}}
        report["cells"][name] = entry
        for cell in yesno.CELLS:
            e = eff[cell]
            if e is None:
                lines.append(f"{name:28s} {col:>5s} {cell:15s} n/a (no seen or unseen questions)")
                continue
            p = e["pooled"]
            seeds_txt = " ".join(f"s{k}={_fmt(v)}" for k, v in e["per_seed"].items())
            lines.append(f"{name:28s} {col:>5s} {cell:15s} Δ={_fmt(p['estimate'])} [{_fmt(p['ci_lo'])}, {_fmt(p['ci_hi'])}] "
                         f"δ={_fmt(m[cell]['delta'])} {entry['direction'][cell]}  {seeds_txt}  (seen {e['n_questions']['seen']} q)")

    if args.primary in cells:
        pc = cells[args.primary]
        if sorted(pc) != [0, 1]:
            raise SystemExit(f"primary {args.primary} needs seeds s0 and s1, got {sorted(pc)}")
        if yesno.n_exp_column(f"{args.primary}_s0") != "1000":
            raise SystemExit(f"primary {args.primary} is not an N_exp=1000 run")
        report["primary"] = {"robustness_view": args.exclude_body_parts, "cell": args.primary, **yesno.primary_verdict(report["cells"][args.primary]["effects"], report["margins"]["1000"])}
    write_json(report, args.out)

    print("margins δ (2 × SD of the d=0 seeds' seen − unseen gap, log-odds):")
    for col in columns:
        print(f"  N_exp {col:>4s}: " + "  ".join(f"{c}={_fmt(report['margins'][col][c]['delta'])}" for c in yesno.CELLS))
    if lines:
        print("\nΔ = (seen − unseen)_run − (seen − unseen)_d0, pooled over seeds; direction + / − / 0 by the G1(b)-style rule:")
        print("\n".join(lines))
    if report["primary"]:
        p = report["primary"]
        print(f"\nPRIMARY {p['cell']}: {p['verdict']}   (P1 control {'pass' if p['p1_control_pass'] else 'FAIL'}; interpretable={p['interpretable']})")
    elif runs:
        print(f"\nprimary {args.primary} not among --runs: no verdict", file=sys.stderr)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
