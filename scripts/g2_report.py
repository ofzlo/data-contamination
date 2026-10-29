#!/usr/bin/env python
"""G2 gate report: τ_style, τ_memory, equivalence at d_style, τ-difference CI. Rules: contam.eval.gates.

δ_S comes from the G0 margins (--margins), δ_V from the G1 gate report (--g1-report). Checkpoints are
metrics JSONs whose checkpoint_id names the cell, ``{condition}_d{d}_s{seed slot}`` with an optional
``G<n>_`` prefix (e.g. ``paired_d16_s2``, ``G2_Caption_d8_s0``), or are given explicitly with
``--cell COND:D=path[,path...]`` (seed slots 0, 1, ... in the order given). The B1 analysis exclusion
is left out; ``--include-b1-overlap`` gives the robustness view (from robustness margins and G1
report; not gate-valid). The seen group is the exposure set (``exposure.n_exp`` / ``--n-exp``); every
input must record it.

Example:
    python scripts/g2_report.py --split data/splits/split_A.json --margins outputs/g0/margins_A.json \
        --g1-report outputs/g1_report.json --baseline outputs/g0/G0_d0_s{0,1,2,3,4}_metrics_v_A.json \
        --checkpoints outputs/metrics/{caption,paired}_d{1,4,16,64}_s{0,1,2}.json outputs/metrics/samedist_d16_s{0,1,2}.json \
        --out outputs/g2_report_A.json          # also writes outputs/g2_report_A.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from contam.config import load_config, read_json, write_json
from contam.data.freeze import add_check_args, cli_check
from contam.data.nocaps import load_annotations
from contam.eval.evaluate import load_analysis_exclusion, load_seen_definition, mark_validity
from contam.eval.gates import CONDITION_NAMES, g2_report, g2_text, parse_run_id, validate_g2_inputs


def parse_cell(spec: str) -> list[dict]:
    head, sep, paths = spec.partition("=")
    cond, _, d = head.partition(":")
    if not sep or cond.lower() not in CONDITION_NAMES or not d.isdigit() or not paths:
        raise SystemExit(f"--cell expects COND:D=path[,path...] with COND in {sorted(set(CONDITION_NAMES.values()))}, got {spec!r}")
    return [{"condition": CONDITION_NAMES[cond.lower()], "d": int(d), "seed": k, "metrics": p} for k, p in enumerate(paths.split(","))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--margins", required=True, help="G0 margins JSON (scripts/compute_margins.py): delta_S")
    ap.add_argument("--g1-report", required=True, help="G1 gate report JSON (scripts/gate_report.py): delta_V")
    ap.add_argument("--baseline", nargs="+", required=True, help="metrics JSON of each d=0 seed (the five G0 seeds)")
    ap.add_argument("--checkpoints", nargs="*", default=[], help="metrics JSONs whose checkpoint_id names the cell")
    ap.add_argument("--cell", action="append", default=[], help="COND:D=path[,path...] for metrics whose checkpoint_id does not name the cell")
    ap.add_argument("--allow-partial", action="store_true", help="smoke tests only: allow partial metrics / incomplete grid; output is NOT gate-valid")
    ap.add_argument("--n-exp", type=int, default=None, help="seen group = the first N_exp exposed images of the split (default: exposure.n_exp; C2 arms: 563 / 141)")
    ap.add_argument("--annotations", default=None, help="needed only for an --n-exp that is not a stored exposure set (default: paths.annotations)")
    add_check_args(ap)
    ap.add_argument("--include-b1-overlap", action="store_true", help="robustness view: keep the B1 analysis exclusion images in every result; output is NOT gate-valid")
    ap.add_argument("--out", required=True, help="report JSON; the text report goes next to it (.txt)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    frozen = cli_check(args, cfg, args.split, "g2_report")
    split = read_json(args.split)
    margins = read_json(args.margins)
    g1 = read_json(args.g1_report)
    base = [read_json(p) for p in args.baseline]
    cks, cms = [], []
    for p in args.checkpoints:
        m = read_json(p)
        cell = parse_run_id(m.get("checkpoint_id"))
        if cell is None:
            raise SystemExit(f"{p}: checkpoint_id {m.get('checkpoint_id')!r} does not name a G2 cell ({{condition}}_d{{d}}_s{{seed}}); pass it with --cell COND:D=path")
        cks.append({"condition": cell[0], "d": cell[1], "seed": cell[2], "metrics": p})
        cms.append(m)
    for spec in args.cell:
        for c in parse_cell(spec):
            cks.append(c)
            cms.append(read_json(c["metrics"]))
    excl = load_analysis_exclusion(cfg, args.include_b1_overlap)
    seen = load_seen_definition(cfg, split, load_annotations(args.annotations) if args.annotations else None, args.n_exp)
    validate_g2_inputs(split, margins, args.margins, g1, args.baseline, base, cks, cms, cfg, args.allow_partial, frozen["manifest_split_sha256"], excl, seen)
    report = g2_report(split, margins, g1, base, cks, cms, cfg, args.allow_partial, excl, seen)
    report.update(frozen)
    mark_validity(report, not args.allow_partial and frozen["manifest_split_sha256"] is not None, excl)
    report["inputs"] = {"margins": args.margins, "g1_report": args.g1_report, "baseline": list(args.baseline), "checkpoints": cks}
    write_json(report, args.out)
    text = g2_text(report)
    Path(args.out).with_suffix(".txt").write_text(text + "\n")
    print(text)
    print(f"-> {args.out} (+ .txt)")


if __name__ == "__main__":
    main()
