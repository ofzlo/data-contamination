"""Gate logic (G1, G2) and gate-input validation. CLIs: scripts/gate_report.py, scripts/g2_report.py.

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

G2 (``g2_report``; τ procedure in contam.eval.stats):

* τ_style: directional unseen effect of ``gates.tau_style_metric`` (word count) vs δ_S (G0 margins);
  τ_memory: mean seen ΔRMI of ``gates.tau_memory_metric`` (summed) vs δ_V (the G1 report). Both are
  estimated per condition (Caption, Paired) over the supplied dose grid; the verdict uses
  ``gates.g2_tau_condition`` (Paired): d_style = τ̂_style, d_memory = τ̂_memory.
* Primary (``gates.g2_equivalence_rule``, default ``two_level_bootstrap``): the two-sided CI upper
  bound of the V effect (seen ΔRMI) at d_style is < δ_V, with the CI from a two-level bootstrap
  (the cell's seeds and the base seeds resampled, then images), so it includes training-seed
  variance. The image bootstrap alone does not (scripts/sim_g2_null.py). Alternatives, all
  reported: ``pooled`` (image-bootstrap CI only), ``pooled_and_every_seed`` (that AND every seed's
  estimate < δ_V, as in G1(b)), ``seed_t`` (t interval over the per-seed estimates).
  τ̂_style = +inf ("no d_style") fails it.
* Secondary: the CI of log₂ τ*_memory − log₂ τ*_style is above 0 (censoring fractions reported).
* The overall verdict is the primary one; the plan's fallback is reported when both effects start
  at the same d (τ̂_style = τ̂_memory).

Both gates leave out the B1 analysis exclusion (``exclusion``; contam.eval.evaluate) and require
every input to record the same one; with it not applied they give the robustness view, which is
not gate-valid. Their seen group is the exposure set (``seen``; contam.eval.evaluate.load_seen_definition),
and every input must record the same seen definition; the seen_unexposed group is reported only.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Sequence

import numpy as np

from contam.eval import stats
from contam.eval.association import aligned
from contam.eval.evaluate import (
    analysis_ids,
    check_comparable,
    check_exclusion,
    check_seen,
    compare,
    exclusion_record,
    is_valid,
    mark_validity,
    per_image_field,
    seen_record,
    warn_partial,
)
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
    split_sha: str | None = None,
    exclusion: dict | None = None,
    seen: dict | None = None,
) -> None:
    """Refuse gate inputs that would silently corrupt a verdict.

    * margins must have been computed on the same split, be gate-valid, carry the G0 base CIDEr check
      (``base_cider_check``; its band is a warning, so only its presence is required), and come from
      exactly the given baseline files;
    * the baseline must be N_BASELINE_SEEDS distinct d=0 runs (distinct files, checkpoint ids and
      generations), and every cell must have ``seeds_per_cell`` seeds;
    * no metrics file or checkpoint id may appear twice across baseline and checkpoints, no
      (condition, d, seed) may appear twice, and no two seeds of a cell may have identical generations;
    * every run's generations must carry the same, recorded GATE_PROVENANCE_FIELDS;
    * margins, baseline and checkpoint metrics must carry one recorded manifest_split_sha256 (the split's
      manifest lock entry when they were evaluated), equal to ``split_sha`` (the current entry) if given;
    * margins, baseline and checkpoint metrics must record the B1 exclusion ``exclusion``, and the
      margins must have applied it as this run does (``check_exclusion``; never relaxed);
    * margins, baseline and checkpoint metrics must record the seen definition ``seen`` (``check_seen``;
      never relaxed).
    The gate-validity, manifest-lock, file-identity and seed-count rules are relaxed only with ``allow_partial``
    (the report is then not gate-valid). Split / completeness / image-set agreement of every
    metrics file is checked by ``compare``.
    """
    if margins["split"]["content_hash"] != split["content_hash"]:
        raise ValueError(f"margins were computed on split {margins['split']['name']}, not {split['name']}")
    check_exclusion([("margins", margins)], exclusion, aggregates=True)
    check_exclusion([(f"metrics {m.get('checkpoint_id')!r}", m) for m in [*baseline, *cell_metrics]], exclusion)
    check_seen([("margins", margins), *((f"metrics {m.get('checkpoint_id')!r}", m) for m in [*baseline, *cell_metrics])], seen)
    if not allow_partial:
        if not is_valid(margins, exclusion):
            raise ValueError("margins are not gate-valid (computed with --allow-partial?)")
        if "base_cider_check" not in margins:
            raise ValueError("margins carry no G0 base CIDEr check (compute_margins.py without --base); G0 is incomplete, hold G1")
        if sorted(map(os.path.realpath, margins.get("d0_files", []))) != sorted(map(os.path.realpath, baseline_paths)):
            raise ValueError(f"margins were computed from d=0 files {margins.get('d0_files')}, not the gate baseline {list(baseline_paths)}")
        shas = {x.get("manifest_split_sha256") for x in [margins, *baseline, *cell_metrics]}
        if None in shas or len(shas) != 1:
            raise ValueError(f"margins, baseline and checkpoint metrics must share one recorded manifest_split_sha256; got {sorted(map(str, shas))}")
        if split_sha is not None and shas != {split_sha}:
            raise ValueError(f"inputs were evaluated on split sha256 {shas.pop()}, but the current manifest lock entry is {split_sha}")
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


def g1_report(
    split: dict, margins: dict, baseline: Sequence[dict], checkpoints: Sequence[dict], cell_metrics: Sequence[dict], cfg: dict, allow_partial: bool = False, exclusion: dict | None = None,
    seen: dict | None = None,
) -> dict:
    """Compute the G1 report dict (inputs validated by the caller with ``validate_inputs``)."""
    st, gcfg, mcfg = cfg["stats"], cfg["gates"], cfg["margins"]
    if allow_partial:
        warn_partial("gate_report")
    kw = dict(n_boot=st["n_boot"], seed=st["boot_seed"], ci=st["ci"], allow_partial=allow_partial, exclusion=exclusion, seen=seen)
    cells = defaultdict(list)
    for c, m in zip(checkpoints, cell_metrics):
        cells[(c["condition"], int(c["d"]))].append(m)
    report = {"split": split["name"], "split_hash": split["content_hash"], "analysis_exclusion": exclusion_record(exclusion, split, map(int, baseline[0]["per_image"]), seen),
              "seen_definition": seen_record(seen), "cells": {}, "criteria": {}}
    mark_validity(report, not allow_partial, exclusion)
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
        if "seen_unexposed" in V["exact_match_train"]:  # reported only
            a["exact_match_change_seen_unexposed"] = V["exact_match_train"]["seen_unexposed"]
        if "rmi_sum" in V:
            a["delta_rmi_seen"] = {f: V[f]["seen"] for f in ("rmi_sum", "rmi_norm")}
            a["delta_rmi_unseen_control"] = {f: V[f]["unseen"] for f in ("rmi_sum", "rmi_norm")}
            if "seen_unexposed" in V["rmi_sum"]:  # reported only
                a["delta_rmi_seen_unexposed"] = {f: V[f]["seen_unexposed"] for f in ("rmi_sum", "rmi_norm")}
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
    for k in ("a", "b"):
        report["criteria"][k]["verdict"] = _verdict_prefix(allow_partial, report["robustness"]) + report["criteria"][k]["verdict"]
    return report


def _verdict_prefix(allow_partial: bool, robustness: bool) -> str:
    return ("NOT GATE-VALID (--allow-partial): " if allow_partial else "") + ("ROBUSTNESS (B1 overlap included), not a gate decision: " if robustness else "")


# ----------------------------------------------------------------------------- G2

G2_TAU_CONDITIONS = ("Caption", "Paired")
G2_FALLBACK = 'both effects start at the same d: reframe as "dose-response per component" and skip C3 (plan, G2)'
G2_EQUIVALENCE_RULES = ("pooled_and_every_seed", "pooled", "two_level_bootstrap", "seed_t")
CONDITION_NAMES = {"caption": "Caption", "paired": "Paired", "samedist": "SameDist", "image": "Image"}
_RUN_ID = re.compile(r"(?:G\d+_)?(?P<cond>[A-Za-z]+)_d(?P<d>0|[1-9]\d*)_s(?P<seed>0|[1-9]\d*)")


def parse_run_id(checkpoint_id: str | None) -> tuple[str, int, int] | None:
    """(condition, d, seed slot) from a run id like ``paired_d64_s0`` or ``G2_Caption_d4_s2``; None if not parseable.

    The whole id must match (no trailing characters), and d / seed are not zero-padded.
    """
    m = _RUN_ID.fullmatch(checkpoint_id or "")
    if not m or m["cond"].lower() not in CONDITION_NAMES:
        return None
    return CONDITION_NAMES[m["cond"].lower()], int(m["d"]), int(m["seed"])


def validate_g2_inputs(
    split: dict,
    margins: dict,
    margins_path: str,
    g1: dict,
    baseline_paths: Sequence[str],
    baseline: Sequence[dict],
    checkpoints: Sequence[dict],
    cell_metrics: Sequence[dict],
    cfg: dict,
    allow_partial: bool = False,
    split_sha: str | None = None,
    exclusion: dict | None = None,
    seen: dict | None = None,
) -> None:
    """G1's ``validate_inputs`` (with ``gates.g2_seeds_per_cell``) plus the G2 grid and the G1 report.

    * Caption and Paired share one dose grid, drawn from ``gates.g2_doses`` + ``g2_optional_doses``
      and containing every ``g2_doses`` entry; SameDist only at ``g2_samedist_dose``; seed slots of
      every cell are 0 .. g2_seeds_per_cell − 1;
    * a metrics file whose checkpoint_id names a cell (``parse_run_id``) is used for that cell only;
    * the G1 report (source of δ_V) was made on this split, is gate-valid, under the inputs' split
      sha, B1 exclusion (applied the same way) and seen definition, from the same baseline and margins files, and its
      criterion (a) passed;
    * every metrics file has the τ fields (the memory field needs teacher-forced log-probs).
    The grid, seed-slot and G1-validity rules are relaxed with ``allow_partial`` (not gate-valid).
    """
    gcfg = cfg["gates"]
    n = gcfg["g2_seeds_per_cell"]
    for c, m in zip(checkpoints, cell_metrics):
        named = parse_run_id(m.get("checkpoint_id"))
        if named is not None and named != (c["condition"], int(c["d"]), int(c["seed"])):
            raise ValueError(f"{c['metrics']}: checkpoint_id {m['checkpoint_id']!r} is {named}, but it is given as {(c['condition'], int(c['d']), int(c['seed']))}")
    validate_inputs(split, margins, baseline_paths, baseline, checkpoints, cell_metrics, n, allow_partial, split_sha, exclusion, seen)
    check_exclusion([("the G1 report", g1)], exclusion, aggregates=True)
    check_seen([("the G1 report", g1)], seen)
    allowed = set(gcfg["g2_doses"]) | set(gcfg["g2_optional_doses"])
    grid, slots = defaultdict(set), defaultdict(set)
    for c in checkpoints:
        cond, d = c["condition"], int(c["d"])
        if cond in G2_TAU_CONDITIONS:
            if d not in allowed:
                raise ValueError(f"{cond} d={d} is not a G2 dose (allowed: {sorted(allowed)})")
        elif cond == "SameDist":
            if d != gcfg["g2_samedist_dose"]:
                raise ValueError(f"SameDist d={d}: G2 has SameDist only at d={gcfg['g2_samedist_dose']}")
        else:
            raise ValueError(f"condition {cond!r} is not part of G2 (Caption, Paired, SameDist)")
        grid[cond].add(d)
        slots[(cond, d)].add(int(c["seed"]))
    if gcfg["g2_equivalence_rule"] not in G2_EQUIVALENCE_RULES:
        raise ValueError(f"gates.g2_equivalence_rule must be one of {G2_EQUIVALENCE_RULES}, got {gcfg['g2_equivalence_rule']!r}")
    if gcfg["g2_tau_condition"] not in grid:
        raise ValueError(f"no {gcfg['g2_tau_condition']} checkpoints (gates.g2_tau_condition)")
    if not allow_partial:
        if grid["Caption"] != grid["Paired"]:
            raise ValueError(f"Caption and Paired dose grids differ: {sorted(grid['Caption'])} vs {sorted(grid['Paired'])}")
        missing = set(gcfg["g2_doses"]) - grid["Paired"]
        if missing:
            raise ValueError(f"G2 grid is missing d={sorted(missing)} for Caption and Paired")
        if "SameDist" not in grid:
            raise ValueError(f"no SameDist d={gcfg['g2_samedist_dose']} checkpoints")
        for cell, s in slots.items():
            if s != set(range(n)):
                raise ValueError(f"cell {cell} has seed slots {sorted(s)}, expected {list(range(n))}")
    if g1.get("split_hash") != split["content_hash"]:
        raise ValueError(f"the G1 report was made on split {g1.get('split')}, not {split['name']}")
    if not allow_partial:
        if not is_valid(g1, exclusion):
            raise ValueError("the G1 report is not gate-valid; delta_V cannot be taken from it")
        if g1.get("manifest_split_sha256") != margins.get("manifest_split_sha256"):
            raise ValueError(f"the G1 report was made under split sha256 {g1.get('manifest_split_sha256')}, the G2 inputs under {margins.get('manifest_split_sha256')}")
        if not str(g1["criteria"]["a"].get("verdict", "")).removeprefix(_verdict_prefix(False, g1.get("robustness") is True)).startswith("PASS"):
            raise ValueError(f"G1(a) did not pass ({g1['criteria']['a'].get('verdict')}); delta_V is not set")
        if sorted(map(os.path.realpath, g1.get("baseline_files", []))) != sorted(map(os.path.realpath, baseline_paths)):
            raise ValueError(f"the G1 report used d=0 files {g1.get('baseline_files')}, not the G2 baseline {list(baseline_paths)}")
        if os.path.realpath(g1.get("margins_file") or "") != os.path.realpath(margins_path):
            raise ValueError(f"the G1 report used margins {g1.get('margins_file')}, not {margins_path}")
    if gcfg["tau_memory_metric"] not in g1["criteria"]["a"].get("delta_V", {}):
        raise ValueError(f"the G1 report has no delta_V[{gcfg['tau_memory_metric']}]")
    for m in list(baseline) + list(cell_metrics):
        row = next(iter(m["per_image"].values()))
        for f in (gcfg["tau_style_metric"], gcfg["tau_memory_metric"]):
            if f not in row:
                raise ValueError(f"metrics {m.get('checkpoint_id')!r} have no per-image {f} (teacher-forced log-probs missing?)")


def g2_report(
    split: dict, margins: dict, g1: dict, baseline: Sequence[dict], checkpoints: Sequence[dict], cell_metrics: Sequence[dict], cfg: dict, allow_partial: bool = False,
    exclusion: dict | None = None, seen: dict | None = None,
) -> dict:
    """Compute the G2 report dict (inputs validated by the caller with ``validate_g2_inputs``). Every
    seen-group quantity (τ_memory, the equivalence test and its two-level bootstrap) uses ``seen``."""
    st, gcfg = cfg["stats"], cfg["gates"]
    if allow_partial:
        warn_partial("g2_report")
    check_comparable(list(baseline) + list(cell_metrics), split, allow_partial, exclusion, seen)
    kw = dict(n_boot=st["n_boot"], seed=st["boot_seed"], ci=st["ci"])
    sf, mf = gcfg["tau_style_metric"], gcfg["tau_memory_metric"]
    direction = cfg["style"]["primary_metrics"][sf]
    delta_s, delta_v = float(margins["delta_S"][sf]), float(g1["criteria"]["a"]["delta_V"][mf])
    avail = [int(k) for k in baseline[0]["per_image"]]
    seen_ids, unseen, unexp = (analysis_ids(split, g, avail, exclusion, seen) for g in ("seen", "unseen", "seen_unexposed"))
    cells = defaultdict(list)
    for c, m in zip(checkpoints, cell_metrics):
        cells[(c["condition"], int(c["d"]))].append(m)

    def arr(ms, field, ids):
        return aligned(per_image_field(ms, field), ids)

    base_s, base_m = arr(baseline, sf, unseen), arr(baseline, mf, seen_ids)
    report = {
        "split": split["name"], "split_hash": split["content_hash"], "analysis_exclusion": exclusion_record(exclusion, split, avail, seen), "seen_definition": seen_record(seen),
        "style_metric": sf, "memory_metric": mf, "delta_S": delta_s, "delta_V": delta_v,
        "tau_condition": gcfg["g2_tau_condition"], "conditions": {}, "samedist": {}, "criteria": {},
    }
    for cond in G2_TAU_CONDITIONS:
        doses = sorted(d for c, d in cells if c == cond)
        if not doses:
            continue
        data = {"unseen": {"b": base_s, **{f"d{d}": arr(cells[(cond, d)], sf, unseen) for d in doses}},
                "seen": {"b": base_m, **{f"d{d}": arr(cells[(cond, d)], mf, seen_ids) for d in doses}}}

        def style_fn(D, doses=doses):
            return np.array([direction * np.mean(D["unseen"][f"d{d}"] - D["unseen"]["b"]) for d in doses])

        def memory_fn(D, doses=doses):
            return np.array([np.mean(D["seen"][f"d{d}"] - D["seen"]["b"]) for d in doses])

        taus = stats.estimate_taus(style_fn, memory_fn, data, doses, delta_s, delta_v, **kw)
        report["conditions"][cond] = {"doses": doses, "n_seeds": {d: len(cells[(cond, d)]) for d in doses}, **taus}
    for (cond, d), ms in sorted(cells.items()):
        if cond == "SameDist":
            report["samedist"][f"d{d}"] = {
                "style_unseen": directional(stats.bootstrap_paired_diff(arr(ms, sf, unseen), base_s, **kw).to_dict(), direction),
                "memory_seen": stats.bootstrap_paired_diff(arr(ms, mf, seen_ids), base_m, **kw).to_dict(),
                "n_seeds": len(ms),
            }
            if unexp:  # reported only
                report["samedist"][f"d{d}"]["memory_seen_unexposed"] = stats.bootstrap_paired_diff(arr(ms, mf, unexp), arr(baseline, mf, unexp), **kw).to_dict()

    tc = report["conditions"][gcfg["g2_tau_condition"]]
    t_style, t_mem = float(tc["style"]["tau"]), float(tc["memory"]["tau"])  # "inf" -> inf
    report["d_style"], report["d_memory"] = tc["style"]["tau"], tc["memory"]["tau"]
    same_start = bool(np.isfinite(t_style) and t_style == t_mem)
    rule = gcfg["g2_equivalence_rule"]
    rule_text = {
        "pooled_and_every_seed": f"pooled two-sided {st['ci']:.0%} CI upper bound < delta_V AND every seed's estimate < delta_V",
        "pooled": f"pooled two-sided {st['ci']:.0%} CI upper bound < delta_V (image bootstrap only)",
        "two_level_bootstrap": f"two-level (seeds, then images) bootstrap {st['ci']:.0%} CI upper bound < delta_V",
        "seed_t": f"seed-level t {st['ci']:.0%} interval upper bound < delta_V (df = n_seeds - 1)",
    }
    prim = {"equivalence_rule": rule, "rule": f"{gcfg['g2_tau_condition']}, mean seen dRMI({mf}) at d_style: {rule_text[rule]}", "delta_V": delta_v}
    if not np.isfinite(t_style):
        prim["verdict"] = "FAIL"
        prim["note"] = f"no d_style on this grid (tau_style = +inf: the unseen {sf} effect's CI lower bound never exceeds delta_S): primary not evaluable -> FAIL"
    else:
        d = int(t_style)
        cell = cells[(gcfg["g2_tau_condition"], d)]
        boot = stats.bootstrap_paired_diff(arr(cell, mf, seen_ids), base_m, **kw)
        per_seed = [float(np.mean(arr([m], mf, seen_ids) - base_m)) for m in cell]
        two = stats.bootstrap_two_level([arr([m], mf, seen_ids) for m in cell], [arr([m], mf, seen_ids) for m in baseline], **kw)
        t_int = stats.seed_t_interval(per_seed, st["ci"])
        pooled_ok = stats.equivalence_below(boot, delta_v)
        rules = {
            "pooled_and_every_seed": {"pass": pooled_ok and all(x < delta_v for x in per_seed)},
            "pooled": {**boot.to_dict(), "pass": pooled_ok},
            "two_level_bootstrap": {**two.to_dict(), "pass": stats.equivalence_below(two, delta_v)},
            "seed_t": {**t_int, "pass": t_int["ci_hi"] < delta_v},
        }
        prim.update(d_style=d, v_effect=boot.to_dict(), per_seed_estimates=per_seed, rules=rules, verdict="PASS" if rules[rule]["pass"] else "FAIL")
        if unexp:  # reported only, never part of the verdict
            prim["v_effect_seen_unexposed"] = stats.bootstrap_paired_diff(arr(cell, mf, unexp), arr(baseline, mf, unexp), **kw).to_dict()
    diff = tc["log2_tau_diff"]
    sec = {"rule": "CI of log2 tau*_memory - log2 tau*_style above 0", **diff, "verdict": "PASS" if diff["ci_above_zero"] else "FAIL"}
    report["criteria"] = {"primary": prim, "secondary": sec}
    report["same_start"] = same_start
    report["fallback"] = G2_FALLBACK if same_start else None
    report["verdict"] = prim["verdict"]
    mark_validity(report, not allow_partial, exclusion)
    prefix = _verdict_prefix(allow_partial, report["robustness"])
    for k in ("primary", "secondary"):
        report["criteria"][k]["verdict"] = prefix + report["criteria"][k]["verdict"]
    report["verdict"] = prefix + report["verdict"]
    return report


def _fmt_tau(t) -> str:
    return "never crosses (+inf)" if t == "inf" else f"d={t:g}"


def _f(x) -> str:
    return x if isinstance(x, str) else f"{x:.3f}"  # "inf" / "-inf" from stats._json_float


def g2_text(report: dict) -> str:
    """Readable G2 report."""
    sd = report.get("seen_definition")
    L = [f"G2 report (split {report['split']}){'' if report['gate_valid'] else '  [NOT GATE-VALID]'}{'  [ROBUSTNESS: B1 overlap included]' if report.get('robustness') else ''}"
         + (f"; seen = exposure set N_exp={sd['n_exp']}" if sd else ""),
         f"delta_S[{report['style_metric']}] = {report['delta_S']:.4f}   delta_V[{report['memory_metric']}] = {report['delta_V']:.4f}"]
    for cond, c in report["conditions"].items():
        role = "verdict" if cond == report["tau_condition"] else "reported only"
        L.append(f"== {cond} ({role}); seeds per dose {c['n_seeds']}")
        for comp, label in (("style", f"unseen {report['style_metric']} (directional) vs delta_S"), ("memory", f"seen dRMI {report['memory_metric']} vs delta_V")):
            t = c[comp]
            L.append(f"  {comp:6s} {label}: tau_hat = {_fmt_tau(t['tau'])}; replicates never crossing {t['frac_replicates_never_cross']:.3f}")
            L.append("    d      effect    fitted   [fit CI lo, hi]")
            for d, e, f, lo, hi in zip(t["doses"], t["effect"], t["fitted"], t["ci_lo"], t["ci_hi"]):
                L.append(f"    {d:<5g} {e:9.4f} {f:9.4f}   [{lo:.4f}, {hi:.4f}]{'  > delta' if lo > t['delta'] else ''}")
        x = c["log2_tau_diff"]
        L.append(f"  log2 tau_mem - log2 tau_style: tau_hat difference {_f(x['estimate'])}; CI of interpolated crossings [{_f(x['ci_lo'])}, {_f(x['ci_hi'])}]; "
                 f"censored replicates: memory-only {x['frac_memory_only_censored']:.3f}, style-only {x['frac_style_only_censored']:.3f}, both {x['frac_both_censored']:.3f}")
    for d, s in report["samedist"].items():
        a, b = s["style_unseen"], s["memory_seen"]
        L.append(f"== SameDist {d} (reported only): style {a['estimate']:.4f} [{a['ci_lo']:.4f}, {a['ci_hi']:.4f}]; memory {b['estimate']:.4f} [{b['ci_lo']:.4f}, {b['ci_hi']:.4f}]")
    p, s = report["criteria"]["primary"], report["criteria"]["secondary"]
    L.append(f"d_style = {_fmt_tau(report['d_style'])}; d_memory = {_fmt_tau(report['d_memory'])}  ({report['tau_condition']})")
    L.append(f"Primary   {p['verdict']}: {p['rule']}")
    if "v_effect" in p:
        v = p["v_effect"]
        L.append(f"          V effect at d={p['d_style']}: {v['estimate']:.4f} [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}] vs delta_V {p['delta_V']:.4f}; "
                 f"per seed: {', '.join(f'{x:.4f}' for x in p['per_seed_estimates'])}")
        if "v_effect_seen_unexposed" in p:
            u = p["v_effect_seen_unexposed"]
            L.append(f"          seen_unexposed (reported only): {u['estimate']:.4f} [{u['ci_lo']:.4f}, {u['ci_hi']:.4f}]")
        for name, r in p["rules"].items():
            bound = f" upper {r['ci_hi']:.4f}" if "ci_hi" in r else ""
            L.append(f"          {'*' if name == p['equivalence_rule'] else ' '} {name:22s}{bound:>16s}  {'pass' if r['pass'] else 'no'}")
    if "note" in p:
        L.append(f"          {p['note']}")
    L.append(f"Secondary {s['verdict']}: {s['rule']}: CI [{_f(s['ci_lo'])}, {_f(s['ci_hi'])}]")
    L.append("          (the tau_hat difference uses the grid lower-bound rule and can fall outside this CI; the gate uses the CI, prereg Section 5)")
    if report["fallback"]:
        L.append(f"Fallback: {report['fallback']}")
    L.append(f"G2 verdict (primary): {report['verdict']}; gate_valid = {report['gate_valid']}")
    return "\n".join(L)
