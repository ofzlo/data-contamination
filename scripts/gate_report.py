#!/usr/bin/env python
"""G1 pilot gate report (criteria (a) and (b); (c) compute is manual). Rules: contam.eval.gates.

Manifest (YAML or JSON):
    split: data/splits/split_A.json
    margins: outputs/margins_A.json
    baseline: [outputs/metrics/d0_s0.json, ...]          # the five d=0 seeds
    checkpoints:
      - {condition: Paired, d: 64, seed: 0, metrics: outputs/metrics/paired_d64_s0.json}
      ...
"""

from __future__ import annotations

import argparse
import os

from contam.config import load_config, load_yaml, read_json, write_json
from contam.data.freeze import add_check_args, cli_check
from contam.eval.gates import g1_report, validate_inputs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--allow-partial", action="store_true", help="smoke tests only: allow partial metrics / non-standard baseline; output is NOT gate-valid")
    add_check_args(ap)
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
    validate_inputs(split, margins, man["baseline"], base, cks, cms, cfg["gates"]["g1_seeds_per_cell"], args.allow_partial, frozen["manifest_split_sha256"])
    report = g1_report(split, margins, base, cks, cms, cfg, args.allow_partial)
    report.update(frozen)
    report["gate_valid"] = report["gate_valid"] and frozen["manifest_split_sha256"] is not None
    # realpaths, so later checks against these files (scripts/g2_report.py) do not depend on the cwd
    report["margins_file"] = os.path.realpath(man["margins"])
    report["baseline_files"] = [os.path.realpath(p) for p in man["baseline"]]
    write_json(report, args.out)

    a, b = report["criteria"]["a"], report["criteria"]["b"]
    print(f"G1 report (split {split['name']}){'' if report['gate_valid'] else '  [NOT GATE-VALID]'}")
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
