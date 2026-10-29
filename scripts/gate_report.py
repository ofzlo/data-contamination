#!/usr/bin/env python
"""G1 pilot gate report (criteria (a) and (b); (c) compute is manual). Rules: contam.eval.gates.

Manifest (YAML or JSON):
    split: data/splits/split_A.json
    margins: outputs/margins_A.json
    baseline: [outputs/metrics/d0_s0.json, ...]          # the five d=0 seeds
    checkpoints:
      - {condition: Paired, d: 64, seed: 0, metrics: outputs/metrics/paired_d64_s0.json}
      ...

The B1 analysis exclusion is left out of every result; ``--include-b1-overlap`` gives the robustness
view (needs robustness margins, compute_margins.py --include-b1-overlap; not gate-valid). The seen
group is the exposure set (``exposure.n_exp`` / ``--n-exp``); every input must record it.
"""

from __future__ import annotations

import argparse
import os
import sys

from contam.config import load_config, load_yaml, read_json, write_json
from contam.data.freeze import add_check_args, cli_check
from contam.data.nocaps import load_annotations
from contam.eval.evaluate import load_analysis_exclusion, load_seen_definition, mark_validity
from contam.eval.gates import g1_report, spec_exposure_problems, validate_inputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--allow-partial", action="store_true", help="smoke tests only: allow partial metrics / non-standard baseline; output is NOT gate-valid")
    ap.add_argument("--n-exp", type=int, default=None, help="seen group = the first N_exp exposed images of the split (default: exposure.n_exp; C2 arms: 141, the shared 3.125% set, in every arm)")
    ap.add_argument("--annotations", default=None, help="needed only for an --n-exp that is not a stored exposure set (default: paths.annotations)")
    add_check_args(ap)
    ap.add_argument("--include-b1-overlap", action="store_true", help="robustness view: keep the B1 analysis exclusion images in every result; output is NOT gate-valid")
    ap.add_argument("--specs", default=None, help="build_g1.py run specs to cross-check exposure sets against (default: <outputs_dir>/runs/G1; skipped if absent)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    man = load_yaml(args.manifest)
    frozen = cli_check(args, cfg, man["split"], "gate_report")
    split = read_json(man["split"])
    margins = read_json(man["margins"])
    base = [read_json(p) for p in man["baseline"]]
    cks = man["checkpoints"]
    cms = [read_json(c["metrics"]) for c in cks]
    excl = load_analysis_exclusion(cfg, args.include_b1_overlap)
    seen = load_seen_definition(cfg, split, load_annotations(args.annotations) if args.annotations else None, args.n_exp)
    validate_inputs(split, margins, man["baseline"], base, cks, cms, cfg["gates"]["g1_seeds_per_cell"], args.allow_partial, frozen["manifest_split_sha256"], excl, seen)
    specs = args.specs or os.path.join(cfg["paths"]["outputs_dir"], "runs", "G1")
    problems = spec_exposure_problems(specs, cks, cms, seen, split["name"])
    if problems:
        msg = "checkpoints not trained on the analysis seen set:\n  " + "\n  ".join(problems)
        if not args.allow_partial:
            raise SystemExit(msg + "\n(--allow-partial: warn only, output NOT gate-valid)")
        print(f"WARNING: {msg}", file=sys.stderr)
    report = g1_report(split, margins, base, cks, cms, cfg, args.allow_partial, excl, seen)
    report.update(frozen)
    mark_validity(report, not args.allow_partial and frozen["manifest_split_sha256"] is not None, excl)
    # realpaths, so later checks against these files (scripts/g2_report.py) do not depend on the cwd
    report["margins_file"] = os.path.realpath(man["margins"])
    report["baseline_files"] = [os.path.realpath(p) for p in man["baseline"]]
    write_json(report, args.out)

    a, b = report["criteria"]["a"], report["criteria"]["b"]
    print(f"G1 report (split {split['name']}){'' if report['gate_valid'] else '  [NOT GATE-VALID]'}{'  [ROBUSTNESS: B1 overlap included]' if report['robustness'] else ''}")
    print(f"(a) {a['cell']}: {a['verdict']}")
    if "delta_rmi_seen" in a:
        r = a["delta_rmi_seen"]["rmi_sum"]
        print(f"    mean seen dRMI(sum) = {r['estimate']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]; delta_V = {a['delta_V']}")
    if "exact_match_change_seen" in a:
        e = a["exact_match_change_seen"]
        print(f"    seen exact-match change = {e['estimate']:.4f} [{e['ci_lo']:.4f}, {e['ci_hi']:.4f}]")
    print(f"(b) d={b['dose']}: {b['verdict']}  ({b['rule']})")
    for name, v in b["tests"].items():
        seeds = ", ".join(f"{x:.3f}" for x in v["per_seed_estimates"])
        print(f"    [{v['role']:9s}] {name:22s} effect={v['estimate']:.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}] seeds=({seeds}) delta_S={v['delta_S']:.4f} {'pass' if v['pass'] else 'no'}")
    print("(c) MANUAL: confirm the P0 grid fits compute")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
