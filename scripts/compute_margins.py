#!/usr/bin/env python
"""G0: decision margins δ_S, δ_A from the five d=0 seeds, plus the base-model CIDEr sanity check.

δ_S(metric) = 2 × SD (ddof=1) across d=0 seeds of the UNSEEN-group mean of the S metric.
δ_A(metric) = 2 × SD (ddof=1) across d=0 seeds of the seen-minus-unseen gap of group means.
δ_V is set at G1 from the Paired d=64 effect (scripts/gate_report.py); its floors are recorded here.

Example:
    python scripts/compute_margins.py --d0 outputs/metrics/d0_s{0,1,2,3,4}.json \
        --base outputs/metrics/base.json --out outputs/margins_A.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from contam.config import load_config, read_json, write_json
from contam.data.freeze import add_check_args, cli_check
from contam.eval.evaluate import warn_partial
from contam.eval.gates import N_BASELINE_SEEDS
from contam.eval.stats import association_margin, style_margin


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--d0", nargs="+", required=True, help="metrics JSON of each d=0 seed")
    ap.add_argument("--base", default=None, help="metrics JSON of the untrained base model")
    ap.add_argument("--split", default=None, help="split JSON the d=0 metrics were computed on (default: <splits_dir>/split_<name>.json)")
    ap.add_argument("--allow-partial", action="store_true", help="smoke tests only; output is NOT gate-valid")
    add_check_args(ap)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["margins"]
    k = mcfg["sd_multiplier"]
    d0 = [read_json(p) for p in args.d0]
    if len({m["split"]["content_hash"] for m in d0}) != 1:
        sys.exit("d=0 metrics come from different splits")
    split_path = args.split or Path(cfg["paths"]["splits_dir"]) / f"split_{d0[0]['split']['name']}.json"
    frozen = cli_check(args, cfg, split_path, "compute_margins")
    cur = frozen["manifest_split_sha256"]
    if read_json(split_path)["content_hash"] != d0[0]["split"]["content_hash"]:
        sys.exit(f"d=0 metrics were not computed on {split_path}")
    d0_shas = {m.get("manifest_split_sha256") for m in d0}
    if len({os.path.realpath(p) for p in args.d0}) != len(args.d0) or len({m["generations_hash"] for m in d0}) != len(d0):
        sys.exit("duplicate d=0 inputs (same file or identical generations)")
    if args.allow_partial:
        warn_partial("compute_margins")
    elif len(d0) != N_BASELINE_SEEDS:
        sys.exit(f"D5 specifies {N_BASELINE_SEEDS} d=0 seeds; got {len(d0)}")
    elif not all(m.get("complete") for m in d0):
        sys.exit("some d=0 metrics were computed on a partial image set")
    elif None in d0_shas or len(d0_shas) != 1:
        sys.exit(f"d=0 metrics must share one recorded manifest_split_sha256; got {sorted(map(str, d0_shas))}")
    elif cur is not None and d0_shas != {cur}:
        sys.exit(f"d=0 metrics were checked against split sha256 {d0_shas.pop()}, but the current lock entry for {Path(split_path).name} is {cur}")

    out = {"gate_valid": not args.allow_partial and cur is not None, "split": d0[0]["split"], "d0_files": [os.path.realpath(p) for p in args.d0],
           "manifest_split_sha256": cur if d0_shas == {cur} else None, "manifest_lock_sha256": frozen["manifest_lock_sha256"], "sd_multiplier": k, "delta_S": {}, "delta_A": {}, "per_seed": {}}
    for f in cfg["style"]["primary_metrics"]:
        vals = [m["summary"]["unseen"][f] for m in d0]
        out["delta_S"][f] = style_margin(vals, k)
        out["per_seed"][f"unseen_mean/{f}"] = vals
    a_fields = sorted({mcfg["a_metric"], "cider_heldout", "spice_heldout"} & set(d0[0]["summary"]["seen"]))
    for f in a_fields:
        seen = [m["summary"]["seen"][f] for m in d0]
        unseen = [m["summary"]["unseen"][f] for m in d0]
        out["delta_A"][f] = association_margin(seen, unseen, k)
        out["per_seed"][f"seen_minus_unseen/{f}"] = list(np.subtract(seen, unseen))
    out["delta_A_primary_metric"] = mcfg["a_metric"]
    out["delta_V_rule"] = {"rel": mcfg["delta_v_rel"], "floor": mcfg["delta_v_floor"], "set_at": "G1 (Paired d=64)"}

    print(f"G0 margins on split {out['split']['name']} ({len(d0)} d=0 seeds, multiplier {k}):")
    for f, v in out["delta_S"].items():
        print(f"  delta_S[{f}] = {v:.4f}   unseen means per seed: {np.round(out['per_seed'][f'unseen_mean/{f}'], 4).tolist()}")
    for f, v in out["delta_A"].items():
        print(f"  delta_A[{f}] = {v:.4f}   seen-unseen per seed: {np.round(out['per_seed'][f'seen_minus_unseen/{f}'], 4).tolist()}")
    d0c = [m["summary"]["all"].get("cider_all10") for m in d0]
    print(f"  d=0 CIDEr(all10, all images) per seed: {d0c}")

    if args.base:
        base = read_json(args.base)
        c = base["summary"]["all"].get("cider_all10")
        lo, hi = mcfg["base_cider_band"]
        ok = c is not None and lo <= c <= hi
        out["base_cider_check"] = {"cider_all10": c, "band": [lo, hi], "in_band": ok, "prompt_id": base.get("prompt_id")}
        msg = f"  base CIDEr(all10) = {c:.2f}; reported-range band [{lo}, {hi}]"
        print(msg + (" OK" if ok else "  <-- WARNING: outside band; check prompt, decoding and tokenization before G1"))
        if base.get("prompt_id") != "caption":
            print("  note: the reported range assumes a one-sentence caption prompt; compare with the matching B3 prompt", file=sys.stderr)
    print("  G0 also requires: GPU hours per run confirmed (manual, Minho).")
    write_json(out, args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
