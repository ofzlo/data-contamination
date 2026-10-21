"""Gate logic (G1) and gate-input validation. CLI: scripts/gate_report.py.

G1 criteria (prereg.md, Section 6):

(a) Paired d=64: the 95% CI lower bound of the pooled mean seen ΔRMI (summed) is > 0 AND the CI
    lower bound of the seen exact-match change vs. d=0 is > 0. Sets
    δ_V = max(0.10 × pooled mean seen ΔRMI, floor).
(b) ONE preregistered primary test (proposed by Yoorhim, needs team sign-off): condition
    ``gates.g1_b_primary.condition`` (Paired) × S metric ``gates.g1_b_primary.metric``
    (word count) at d=4. PASS iff the directional unseen effect (sign from
    ``style.primary_metrics``) has pooled CI lower bound > δ_S AND the per-seed point estimate
    exceeds δ_S in every training seed. Every other condition × primary S metric is reported as
    secondary and does not gate (an any-of-4 rule roughly doubles the null false-PASS rate;
    scripts/sim_g1b_null.py).
(c) Manual (compute).
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Sequence

from contam.eval.evaluate import compare, warn_partial
from contam.eval.stats import memory_margin

N_BASELINE_SEEDS = 5
# Generation provenance that must be identical across baseline and checkpoints (T=0 batch
# composition depends on batch_size, truncation on max_new_tokens).
GATE_PROVENANCE_FIELDS = ("prompt_sha", "model", "batch_size", "max_new_tokens")


def directional(ci: dict, direction: float) -> dict:
    lo, hi = sorted((direction * ci["ci_lo"], direction * ci["ci_hi"]))
    return {"estimate": direction * ci["estimate"], "ci_lo": lo, "ci_hi": hi}


def _distinct(items: Sequence, what: str) -> None:
    seen = set()
    for x in items:
        if x in seen:
            raise ValueError(f"duplicate {what}: {x!r}")
        seen.add(x)


def validate_inputs(
    split: dict,
    margins: dict,
    baseline_paths: Sequence[str],
    baseline: Sequence[dict],
    checkpoints: Sequence[dict],
    cell_metrics: Sequence[dict],
    seeds_per_cell: int,
    allow_partial: bool = False,
) -> None:
    """Refuse gate inputs that would silently corrupt a verdict.

    * margins must have been computed on the same split, be gate-valid, and come from exactly the
      given baseline files;
    * the baseline must be N_BASELINE_SEEDS distinct d=0 runs (distinct files, checkpoint ids and
      generations), and every cell must have ``seeds_per_cell`` seeds;
    * no metrics file or checkpoint id may appear twice across baseline and checkpoints, no
      (condition, d, seed) may appear twice, and no two seeds of a cell may have identical generations;
    * every run's generations must carry the same, recorded GATE_PROVENANCE_FIELDS.
    The gate-validity, file-identity and seed-count rules are relaxed only with ``allow_partial``
    (the report is then not gate-valid). Split / completeness / image-set agreement of every
    metrics file is checked by ``compare``.
    """
    if margins["split"]["content_hash"] != split["content_hash"]:
        raise ValueError(f"margins were computed on split {margins['split']['name']}, not {split['name']}")
    if not allow_partial:
        if margins.get("gate_valid") is not True:
            raise ValueError("margins are not gate-valid (computed with --allow-partial?)")
        if sorted(map(os.path.realpath, margins.get("d0_files", []))) != sorted(map(os.path.realpath, baseline_paths)):
            raise ValueError(f"margins were computed from d=0 files {margins.get('d0_files')}, not the gate baseline {list(baseline_paths)}")
    _distinct([m["generations_hash"] for m in baseline], "baseline generations (identical d=0 outputs)")
    if len(baseline) != N_BASELINE_SEEDS and not allow_partial:
        raise ValueError(f"baseline must be {N_BASELINE_SEEDS} distinct d=0 seeds, got {len(baseline)}")
    _distinct([os.path.realpath(p) for p in list(baseline_paths) + [c["metrics"] for c in checkpoints]], "metrics file (baseline or checkpoint)")
    _distinct([m.get("checkpoint_id") for m in list(baseline) + list(cell_metrics)], "checkpoint_id (baseline or checkpoint)")
    _distinct([(c["condition"], int(c["d"]), int(c["seed"])) for c in checkpoints], "(condition, d, seed)")
    by_cell = defaultdict(list)
    for c, m in zip(checkpoints, cell_metrics):
        by_cell[(c["condition"], int(c["d"]))].append(m["generations_hash"])
    for cell, hashes in by_cell.items():
        _distinct(hashes, f"generations within cell {cell}")
        if len(hashes) != seeds_per_cell and not allow_partial:
            raise ValueError(f"cell {cell} has {len(hashes)} seeds, expected {seeds_per_cell}")
    for f in GATE_PROVENANCE_FIELDS:
        vals = {m.get("provenance", {}).get("generations", {}).get(f) for m in list(baseline) + list(cell_metrics)}
        if None in vals or len(vals) != 1:
            raise ValueError(f"baseline and checkpoint generations must share one recorded {f}; got {sorted(map(str, vals))}")


def g1_report(split: dict, margins: dict, baseline: Sequence[dict], checkpoints: Sequence[dict], cell_metrics: Sequence[dict], cfg: dict, allow_partial: bool = False) -> dict:
    """Compute the G1 report dict (inputs validated by the caller with ``validate_inputs``)."""
    st, gcfg, mcfg = cfg["stats"], cfg["gates"], cfg["margins"]
    if allow_partial:
        warn_partial("gate_report")
    kw = dict(n_boot=st["n_boot"], seed=st["boot_seed"], ci=st["ci"], allow_partial=allow_partial)
    cells = defaultdict(list)
    for c, m in zip(checkpoints, cell_metrics):
        cells[(c["condition"], int(c["d"]))].append(m)
    report = {"split": split["name"], "split_hash": split["content_hash"], "gate_valid": not allow_partial, "cells": {}, "criteria": {}}
    for key in sorted(cells):
        ms = cells[key]
        report["cells"][f"{key[0]}/d{key[1]}"] = {"pooled": compare(ms, baseline, split, **kw), "per_seed": [compare([m], baseline, split, **kw) for m in ms], "n_seeds": len(ms)}

    # (a)
    name_a = f"Paired/d{gcfg['g1_a_dose']}"
    a = {"cell": name_a}
    if name_a not in report["cells"]:
        a["verdict"] = "NOT EVALUABLE (no Paired d=64 checkpoints)"
    else:
        V = report["cells"][name_a]["pooled"]["V"]
        em = V["exact_match_train"]["seen"]
        a["exact_match_change_seen"] = em
        a["exact_match_ok"] = em["ci_lo"] > 0
        if "rmi_sum" in V:
            a["delta_rmi_seen"] = {f: V[f]["seen"] for f in ("rmi_sum", "rmi_norm")}
            a["delta_rmi_unseen_control"] = {f: V[f]["unseen"] for f in ("rmi_sum", "rmi_norm")}
            a["delta_rmi_ok"] = V["rmi_sum"]["seen"]["ci_lo"] > 0
            a["delta_V"] = {f: memory_margin(V[f]["seen"]["estimate"], mcfg["delta_v_floor"][f], mcfg["delta_v_rel"]) for f in ("rmi_sum", "rmi_norm")}
            a["verdict"] = "PASS" if (a["delta_rmi_ok"] and a["exact_match_ok"]) else "FAIL"
        else:
            a["verdict"] = "NOT EVALUABLE (no teacher-forced log-probs)"
    report["criteria"]["a"] = a

    # (b)
    dose = gcfg["g1_b_dose"]
    prim = gcfg["g1_b_primary"]
    b = {"dose": dose, "delta_S": margins["delta_S"], "primary": dict(prim), "tests": {}}
    for (cond, d) in sorted(cells):
        if d != dose:
            continue
        cell = report["cells"][f"{cond}/d{d}"]
        for f, direction in cfg["style"]["primary_metrics"].items():
            eff = directional(cell["pooled"]["S"][f]["unseen"], direction)
            per_seed = [direction * ps["S"][f]["unseen"]["estimate"] for ps in cell["per_seed"]]
            delta = margins["delta_S"][f]
            ok = eff["ci_lo"] > delta and all(x > delta for x in per_seed)
            b["tests"][f"{cond}/{f}"] = {
                **eff,
                "per_seed_estimates": per_seed,
                "delta_S": delta,
                "pass": ok,
                "role": "primary" if (cond == prim["condition"] and f == prim["metric"]) else "secondary",
            }
    pkey = f"{prim['condition']}/{prim['metric']}"
    b["verdict"] = ("PASS" if b["tests"][pkey]["pass"] else "FAIL") if pkey in b["tests"] else "NOT EVALUABLE"
    b["rule"] = f"{pkey} at d={dose}: pooled CI_lo > delta_S and every seed's estimate > delta_S (others secondary)"
    report["criteria"]["b"] = b
    report["criteria"]["c"] = {"verdict": "MANUAL", "note": "P0 grid fits compute: confirm from the W1 timing run (Minho)"}
    if allow_partial:
        for k in ("a", "b"):
            report["criteria"][k]["verdict"] = "NOT GATE-VALID (--allow-partial): " + report["criteria"][k]["verdict"]
    return report
