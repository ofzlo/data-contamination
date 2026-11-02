#!/usr/bin/env python
"""Exploratory dose / K analysis of trained checkpoints vs. the five d=0 seeds (docs/deviations.md).

EXPLORATORY: nothing here is a gate input (every output records ``exploratory: true``,
``gate_valid: false``); the prereg margins and gate rules are used as they are, never changed.

For every checkpoint metrics file (``scripts/evaluate_checkpoint.py`` output) this compares the
checkpoint with the d=0 seeds via ``contam.eval.evaluate.compare`` (B1 exclusion, seen definition;
partial per-image tables only with ``--allow-partial``) and writes ``<out-dir>/<checkpoint_id>[_short].{json,md}`` with:

(a) seen / unseen / seen_unexposed levels of CIDEr / SPICE (held-out, all10, train), word count,
    boilerplate and exact match for the checkpoint, d=0 (and base), model − d0 with CIs, and A Δ;
(b) ΔRMI summed and per token (seen, unseen, seen_unexposed) with the prereg floors (0.5 / 0.02);
(d) diversity on unseen and seen (``contam.eval.diversity``): distinct-1/2 (corpus level), duplicate
    rate, nouns per caption; levels and model − d0 with paired image-bootstrap CIs.

A cell with several seeds gets a seed-pooled file too (``..._pooled``); dose series, τ and plots use
the pooled cells. Run ids: ``G<k>_<Cond>_d<d>_s<slot>`` or the exploratory ``EX_<Cond>_d<d>_n<N_exp>_s<slot>``;
N_exp and the exposure ids sha256 are read from the run spec (``--runs-dir``/<GATE>/<id>.json) and must
match the seen definition used. The seen group of a run is ITS exposure set (N_exp images; the
sets nest, so a K-sweep run's seen set is inside the main one): the per-image tables do not depend
on the seen definition, so metrics evaluated under the default one are regrouped (``rebind_seen``).

(e) τ split, per prompt and condition, over the doses at the main N_exp (``tau_<prompt>.{json,md}``):
    τ_verbatim (seen exact-match change; no prereg margin: δ_verbatim = 0.05 absolute, CI lower bound
    > 0.05, docs/deviations.md 2026-11-01; the δ = 0 onset is reported as a secondary value),
    τ_latent (seen ΔRMI summed and per token; δ_V from the G1 report ``criteria.a.delta_V``, else
    max(0.10 × Paired d=64 seen ΔRMI, floor), or the floor alone, pending), and τ_style for reference
    (unseen word count, directional, δ_S from the G0 margins). Prereg τ procedure:
    ``contam.eval.stats.estimate_tau`` (isotonic over log2 d, pointwise percentile bounds).

Short prompt (``--short-*``): checkpoints generated with ``nocaps_like`` are compared with the
short-prompt d=0 seeds (or, without them, the short-prompt untrained base): does exposure still raise
scores over a prompt-matched baseline? No log-probs there (Q, S, exact match, diversity).

``summary_<prompt>.{json,md}``: model − d0 (95% CI) per condition × dose × N_exp for every metric
above, marking changes whose CI lies entirely beyond their margin (δ_S, δ_verbatim, δ_V, δ_A).

Figures (``<fig-dir>/<fig-tag>_dose.{png,pdf}``, ``..._ksweep``, ``..._short_dose``): dose: x = log2 d
(d=0 at the left), one line per condition; K sweep: x = N_exp at d=1 (log), Paired vs SameDist; error
bars are 95% CIs of levels (ΔRMI: of model − d0), δ_V per token dashed. Outputs go to a new directory
(``<repo>/outputs/explore/<tag>/``); every target path is checked before anything is written, and an
existing file is never overwritten.

Example:
    python scripts/analysis/explore_dose.py --split data/splits/split_A.json --tag full_20261101 \\
        --checkpoints outputs/g1/G1_*_d4_s0_metrics_v_A.json outputs/g2/G2_*_d1_s0_metrics_v_A.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from contam.config import content_hash, load_config, read_json, write_json
from contam.eval import diversity, stats
from contam.eval.association import aligned
from contam.eval.evaluate import (
    REPORT_GROUPS,
    analysis_ids,
    compare,
    load_analysis_exclusion,
    load_generations,
    load_seen_definition,
    per_image_field,
    seen_matches,
    seen_record,
)

RUN_ID = re.compile(r"(?P<gate>G\d+|EX)_(?P<cond>[A-Za-z]+)_d(?P<d>0|[1-9]\d*)(?:_n(?P<n>[1-9]\d*))?_s(?P<slot>0|[1-9]\d*)")
CONDITIONS = ("Paired", "Caption", "SameDist", "Image", "Shuffled")
COLORS = {"Paired": "#2a78d6", "Caption": "#eb6834", "SameDist": "#1baf7a", "Image": "#6250d6", "Shuffled": "#e87ba4"}
MARKERS = {"Paired": "o", "Caption": "s", "SameDist": "^", "Image": "D", "Shuffled": "v"}
LEVEL_FIELDS = ("word_count", "boilerplate", "exact_match_train", "cider_heldout", "spice_heldout", "cider_all10", "spice_all10", "cider_train", "spice_train")
RMI_FIELDS = ("rmi_sum", "rmi_norm")
DIV_GROUPS = ("unseen", "seen")
D0_X = -1.5  # where d=0 sits on the log2 d axis
DUP_NOTE = ("Duplicate rate (share of captions in the group that occur more than once in it) depends on the group size: "
            "it is not comparable across groups of different size (unseen vs seen, or seen across K / N_exp).")


# ----------------------------------------------------------------------------- inputs


def parse_run(checkpoint_id: str) -> dict | None:
    """``{gate, condition, d, n_exp, slot}`` of a G / EX run id (``n_exp``: spelled in EX ids, None in G ids:
    it comes from the run spec, ``load_entry``); None otherwise (d=0, base)."""
    m = RUN_ID.fullmatch(checkpoint_id or "")
    if not m or m["cond"] not in CONDITIONS:
        return None
    if (m["gate"] == "EX") != (m["n"] is not None):
        return None  # N_exp is spelled in (and only in) exploratory ids
    return {"gate": m["gate"], "condition": m["cond"], "d": int(m["d"]), "n_exp": int(m["n"]) if m["n"] else None, "slot": int(m["slot"])}


def run_from_spec(run: dict, checkpoint_id: str, runs_dir) -> dict:
    """``run`` completed from its spec ``<runs_dir>/<gate>/<id>.json`` (scripts/build_g1.py): N_exp and the
    exposure ids sha256 (the seen group of the run); the spec must agree with the id."""
    p = Path(runs_dir) / run["gate"] / f"{checkpoint_id}.json"
    if not p.exists():
        raise SystemExit(f"{checkpoint_id}: run spec {p} not found (N_exp and the exposure set are read from it)")
    spec = read_json(p)
    n_exp, sha = int(spec["N_exp"]), spec["exposure"]["exposure_ids_sha256"]
    got = (spec["run_id"], spec["condition"], int(spec["d"]), int(spec["seed_slot"]), int(spec["exposure"]["n_exp"]))
    want = (checkpoint_id, run["condition"], run["d"], run["slot"], n_exp)
    if got != want or (run["n_exp"] is not None and run["n_exp"] != n_exp):
        raise SystemExit(f"{p}: spec (run_id, condition, d, slot, exposure.n_exp) = {got} and N_exp {n_exp} disagree with run id {checkpoint_id}")
    return {**run, "n_exp": n_exp, "exposure_ids_sha256": sha, "spec": str(p)}


def find_generations(metrics_path, m: dict) -> Path | None:
    """The T=0 generations next to a metrics file: ``<checkpoint_id>_<prompt_id>.jsonl`` (run_gen_job /
    run_short_eval_job naming); None if absent."""
    p = Path(metrics_path).parent / f"{m['checkpoint_id']}_{m.get('prompt_id', 'caption')}.jsonl"
    return p if p.exists() else None


def load_entry(path, runs_dir=None) -> dict:
    """Metrics file + parsed run id (completed from its run spec under ``runs_dir``) + its captions
    (checked against ``generations_hash``)."""
    m = read_json(path)
    run = parse_run(m["checkpoint_id"])
    if run is not None:
        if runs_dir is None:
            raise SystemExit(f"{path}: run {m['checkpoint_id']} needs its run spec (runs_dir)")
        run = run_from_spec(run, m["checkpoint_id"], runs_dir)
    entry = {"path": str(path), "m": m, "id": m["checkpoint_id"], "prompt_id": m.get("prompt_id", "caption"), "run": run, "gens": None}
    g = find_generations(path, m)
    if g is not None:
        gens = load_generations(g, entry["prompt_id"], checkpoint_id=m["checkpoint_id"])
        if content_hash({str(i): gens[i] for i in sorted(gens)}) != m["generations_hash"]:
            raise ValueError(f"{g} does not hold the generations {path} was computed from")
        entry["gens"], entry["gens_path"] = gens, str(g)
    return entry


class SeenDefs:
    """``load_seen_definition`` per N_exp, cached."""

    def __init__(self, cfg, split, ann):
        self.cfg, self.split, self.ann, self.cache = cfg, split, ann, {}

    def __call__(self, n_exp: int) -> dict:
        if n_exp not in self.cache:
            self.cache[n_exp] = load_seen_definition(self.cfg, self.split, self.ann, n_exp)
        return self.cache[n_exp]


def rebind_seen(m: dict, seen: dict, seen_defs: SeenDefs) -> dict:
    """``m`` regrouped under ``seen``. ``per_image`` does not depend on the seen definition, so only the
    record changes; ``m`` must record a valid exposure-set definition of this split (checked)."""
    rec = m.get("seen_definition") or {}
    if rec.get("kind") != "exposure" or not seen_matches(m, seen_defs(int(rec["n_exp"]))):
        raise ValueError(f"metrics {m.get('checkpoint_id')!r} record seen definition {rec or None}, not an exposure set of this split")
    out = dict(m, seen_definition=seen_record(seen))
    out.pop("summary", None)  # group means of the old definition
    return out


# ----------------------------------------------------------------------------- one cell vs. d=0


class Nouns:
    """Nouns per caption of a generations file (``diversity.noun_counts``), cached on disk by requested
    backend and generations hash (new cache files only)."""

    def __init__(self, cache_dir: Path, backend: str = "auto"):
        self.dir, self.backend, self.mem, self.used = Path(cache_dir), backend, {}, set()

    def __call__(self, entry: dict) -> dict[int, float]:
        h = entry["m"]["generations_hash"]
        if h not in self.mem:
            p = self.dir / f"nouns_{self.backend}_{h[:16]}.json"
            if p.exists():
                c = read_json(p)
                if c["generations_hash"] != h or (self.backend != "auto" and not c["backend"].startswith(self.backend)):
                    raise ValueError(f"noun cache {p} holds backend {c['backend']} / generations {c['generations_hash'][:16]}")
            else:
                ids = sorted(entry["gens"])
                counts, used = diversity.noun_counts([entry["gens"][i] for i in ids], self.backend)
                c = {"backend": used, "generations_hash": h, "counts": {str(i): float(x) for i, x in zip(ids, counts)}}
                write_json(c, p)
            self.used.add(c["backend"])
            if len(self.used) > 1:
                raise ValueError(f"noun counts from several backends {sorted(self.used)} would be mixed")
            self.mem[h] = {int(k): v for k, v in c["counts"].items()}
        return self.mem[h]


def _boot_levels(model_arr, base_arr, kw) -> dict:
    return {"model": stats.bootstrap_mean(model_arr, **kw).to_dict(), "d0": stats.bootstrap_mean(base_arr, **kw).to_dict(),
            "diff": stats.bootstrap_paired_diff(model_arr, base_arr, **kw).to_dict()}


def analyze(models: list[dict], d0: list[dict], split: dict, excl: dict, seen: dict, seen_defs: SeenDefs, kw: dict,
            base: dict | None = None, nouns: Nouns | None = None, floors: dict | None = None, allow_partial: bool = False) -> dict:
    """(a), (b), (d) of the module docstring for one cell (``models``: its seeds' entries) vs. ``d0``."""
    ms = [rebind_seen(e["m"], seen, seen_defs) for e in models]
    bs = [rebind_seen(e["m"], seen, seen_defs) for e in d0]
    cmp = compare(ms, bs, split, allow_partial=allow_partial, exclusion=excl, seen=seen, **kw)
    avail = [int(k) for k in ms[0]["per_image"]]
    grp = {g: analysis_ids(split, g, avail, excl, seen) for g in REPORT_GROUPS}
    fields = set.intersection(*(set(next(iter(m["per_image"].values()))) for m in ms + bs))

    levels: dict = {}
    for f in LEVEL_FIELDS:
        if f not in fields:
            continue
        levels[f] = {}
        for g, ids in grp.items():
            if not ids:
                continue
            levels[f][g] = _boot_levels(aligned(per_image_field(ms, f), ids), aligned(per_image_field(bs, f), ids), kw)
            if base is not None and f in next(iter(base["m"]["per_image"].values())):
                levels[f][g]["base"] = float(np.mean([base["m"]["per_image"][str(i)][f] for i in ids]))

    out = {
        "exploratory": True,
        "gate_valid": False,
        "checkpoint_ids": [e["id"] for e in models],
        "metrics_files": [e["path"] for e in models],
        "run": models[0]["run"],
        "prompt_id": models[0]["prompt_id"],
        "baseline": {"checkpoint_ids": [e["id"] for e in d0], "files": [e["path"] for e in d0]},
        "base_file": base["path"] if base else None,
        "seen_definition": seen_record(seen),
        "analysis_exclusion": cmp["analysis_exclusion"],
        "n": {g: len(ids) for g, ids in grp.items()},
        "levels": levels,
        "A": {f: cmp["A"][f] for f in sorted(cmp["A"]) if f.endswith(("heldout", "all10", "train"))},
        "rmi": {f: {**cmp["V"][f], "floor": (floors or {}).get(f)} for f in RMI_FIELDS if f in cmp["V"]},
        "compare": cmp,
    }

    if all(e["gens"] is not None for e in models + d0):
        div: dict = {}
        for g in DIV_GROUPS:
            ids = grp[g]
            if not ids:
                continue
            texts = lambda es: [[e["gens"][i] for i in ids] for e in es]  # noqa: E731
            mt, bt = texts(models), texts(d0)
            div[g] = {f"distinct_{n}": diversity.bootstrap_corpus([diversity.ngram_matrix(t, n) for t in mt], [diversity.ngram_matrix(t, n) for t in bt], **kw)
                      for n in (1, 2)}
            dup = lambda ts: np.mean([diversity.duplicate_flags(t) for t in ts], axis=0)  # noqa: E731
            div[g]["duplicate_rate"] = _boot_levels(dup(mt), dup(bt), kw)
            if nouns is not None:
                pooled = lambda es: np.mean([[nouns(e)[i] for i in ids] for e in es], axis=0)  # noqa: E731
                div[g]["nouns_per_caption"] = _boot_levels(pooled(models), pooled(d0), kw)
        out["diversity"] = div
        out["nouns_backend"] = sorted(nouns.used) if nouns is not None else None
    else:
        out["diversity"] = None
        out["diversity_skipped"] = "generations file not found next to every metrics file"
    return out


# ----------------------------------------------------------------------------- τ split


def _pooled_seen(entries, field, ids):
    return aligned(per_image_field([e["m"] for e in entries], field), ids)


def tau_split(series: dict[int, list[dict]], d0: list[dict], split: dict, excl: dict, seen: dict, kw: dict, deltas: dict) -> dict:
    """τ per component over the doses of ``series`` (``{d: seed entries}``, d > 0) against ``d0``.

    ``deltas``: ``{component: (delta, note)}``; components whose field is missing at some dose are skipped."""
    doses = sorted(series)
    avail = [int(k) for k in d0[0]["m"]["per_image"]]
    comps = {  # component -> (field, group, direction)
        "verbatim_exact_match": ("exact_match_train", "seen", 1.0),
        "verbatim_exact_match_delta0": ("exact_match_train", "seen", 1.0),
        "latent_rmi_sum": ("rmi_sum", "seen", 1.0),
        "latent_rmi_norm": ("rmi_norm", "seen", 1.0),
        "style_word_count": ("word_count", "unseen", -1.0),
    }
    out = {"doses": doses, "seen_definition": seen_record(seen), "components": {}}
    for name, (field, group, direction) in comps.items():
        if name not in deltas:
            continue
        if not all(field in next(iter(e["m"]["per_image"].values())) for es in [*series.values(), d0] for e in es):
            continue
        ids = analysis_ids(split, group, avail, excl, seen)
        data = {"g": {"d0": _pooled_seen(d0, field, ids), **{f"d{d}": _pooled_seen(series[d], field, ids) for d in doses}}}

        def effect(x, doses=doses, direction=direction):
            return np.array([direction * np.mean(x["g"][f"d{d}"] - x["g"]["d0"]) for d in doses])

        delta, note = deltas[name]
        res = stats.estimate_tau(effect, data, doses, delta, **kw).to_dict()
        res["raw"] = {str(d): stats.bootstrap_paired_diff(data["g"][f"d{d}"], data["g"]["d0"], direction=direction, **kw).to_dict() for d in doses}
        res.update(field=field, group=group, direction=direction, delta_note=note, n_seeds={str(d): len(series[d]) for d in doses})
        out["components"][name] = res
    return out


def delta_v(paired_d64: list[dict] | None, d0: list[dict], split, excl, seen, cfg, g1_report: str | None = None) -> dict:
    """δ_V per RMI variant. With ``g1_report`` (``scripts/g1_report`` output): its ``criteria.a.delta_V``, the
    value set at G1 (prereg Section 4), with this analysis's own recomputation from ``paired_d64`` recorded
    next to it. Without: max(rel × pooled Paired d=64 seen ΔRMI, floor); the floor alone, pending, without d=64."""
    mg = cfg["margins"]
    out = {}
    rep = read_json(g1_report) if g1_report else None
    for f in RMI_FIELDS:
        floor = float(mg["delta_v_floor"][f])
        has = paired_d64 and all(f in next(iter(e["m"]["per_image"].values())) for e in paired_d64 + d0)
        if rep is not None:
            out[f] = {"delta": float(rep["criteria"]["a"]["delta_V"][f]), "floor": floor, "status": f"set at G1 ({Path(g1_report).name} criteria.a.delta_V)",
                      "source": str(g1_report)}
            if has:
                ids = analysis_ids(split, "seen", [int(k) for k in d0[0]["m"]["per_image"]], excl, seen)
                E = float(np.mean(_pooled_seen(paired_d64, f, ids) - _pooled_seen(d0, f, ids)))
                out[f]["recomputed"] = {"delta": stats.memory_margin(E, floor, mg["delta_v_rel"]), "E_paired_d64": E, "paired_d64_ids": [e["id"] for e in paired_d64]}
        elif has:
            ids = analysis_ids(split, "seen", [int(k) for k in d0[0]["m"]["per_image"]], excl, seen)
            E = float(np.mean(_pooled_seen(paired_d64, f, ids) - _pooled_seen(d0, f, ids)))
            out[f] = {"delta": stats.memory_margin(E, floor, mg["delta_v_rel"]), "E_paired_d64": E, "floor": floor, "status": "set (Paired d=64, exploratory seeds)",
                      "paired_d64_ids": [e["id"] for e in paired_d64]}
        else:
            out[f] = {"delta": floor, "E_paired_d64": None, "floor": floor, "status": "pending: Paired d=64 not available, floor only"}
    return out


# ----------------------------------------------------------------------------- markdown


def _ci(r, scale=1.0, fmt="{:.2f}"):
    if r is None:
        return "–"
    return f"{fmt.format(scale * r['estimate'])} [{fmt.format(scale * r['ci_lo'])}, {fmt.format(scale * r['ci_hi'])}]"


def cell_markdown(res: dict, title: str) -> str:
    L = [f"# {title}", "", f"EXPLORATORY (not a gate input). Prompt `{res['prompt_id']}`; checkpoint(s) {', '.join(res['checkpoint_ids'])}; "
         f"baseline {', '.join(res['baseline']['checkpoint_ids'])} (pooled); seen = exposure set N_exp={res['seen_definition']['n_exp']} "
         f"(`{res['seen_definition']['ids_sha256'][:12]}`); n = " + ", ".join(f"{g} {n}" for g, n in res["n"].items()) + ". 95% paired image-bootstrap CIs.", ""]
    L += ["## Levels (model vs d=0)", "", "| metric | group | base | d=0 | model | model − d0 |", "|---|---|---|---|---|---|"]
    for f, by_g in res["levels"].items():
        scale = 100.0 if f in ("exact_match_train", "boilerplate") else 1.0
        for g, r in by_g.items():
            base = f"{scale * r['base']:.2f}" if "base" in r else "–"
            L.append(f"| {f}{' (%)' if scale == 100 else ''} | {g} | {base} | {_ci(r['d0'], scale)} | {_ci(r['model'], scale)} | {_ci(r['diff'], scale)} |")
    L += ["", "## A Δ (seen − unseen change vs d=0)", "", "| metric | Δ |", "|---|---|"]
    L += [f"| {f} | {_ci(r)} |" for f, r in res["A"].items()]
    if res["rmi"]:
        L += ["", "## ΔRMI (model − d0, nats)", "", "| variant | floor | seen | unseen | seen_unexposed |", "|---|---|---|---|---|"]
        for f, r in res["rmi"].items():
            fmt = "{:.3f}" if f == "rmi_norm" else "{:.2f}"
            L.append(f"| {f} | {r['floor']} | {_ci(r.get('seen'), fmt=fmt)} | {_ci(r.get('unseen'), fmt=fmt)} | {_ci(r.get('seen_unexposed'), fmt=fmt)} |")
    if res.get("diversity"):
        L += ["", f"## Diversity (nouns: {', '.join(res['nouns_backend'] or ['–'])})", "", "| metric | group | d=0 | model | model − d0 |", "|---|---|---|---|---|"]
        for g, by_m in res["diversity"].items():
            for k, r in by_m.items():
                fmt = "{:.2f}" if k == "nouns_per_caption" else "{:.4f}"
                L.append(f"| {k} | {g} | {_ci(r['d0'], fmt=fmt)} | {_ci(r['model'], fmt=fmt)} | {_ci(r['diff'], fmt=fmt)} |")
        L += ["", "distinct-n: corpus level, CIs bias-shifted (contam.eval.diversity); d=0 = mean of the seeds' values.", "",
              DUP_NOTE]
    elif res.get("diversity_skipped"):
        L += ["", f"Diversity skipped: {res['diversity_skipped']}."]
    return "\n".join(L) + "\n"


def tau_markdown(taus: dict, dv: dict, prompt_id: str, deltas: dict | None = None) -> str:
    L = [f"# τ split (EXPLORATORY), prompt `{prompt_id}`", "",
         "Prereg τ procedure (isotonic over log2 d, pointwise 95% bounds); τ̂ = smallest d whose lower bound > δ (inf: never). "
         "Verbatim has no prereg margin: primary δ_verbatim = 0.05 (5 percentage points of seen exact-match change vs d=0, docs/deviations.md "
         "2026-11-01); `verbatim_exact_match_delta0` is the secondary δ = 0 onset (CI lower bound > 0).", "",
         "δ_V: " + "; ".join(f"{f} = {r['delta']:.4g} ({r['status']})" for f, r in dv.items()), ""]
    if deltas:
        L += ["| component | δ | note |", "|---|---|---|"] + [f"| {k} | {v['delta']:.4g} | {v['note']} |" for k, v in deltas.items()] + [""]
    for cond, t in taus.items():
        L += [f"## {cond}", "", "seeds per dose: " + ", ".join(f"d={d}: {n}" for d, n in next(iter(t["components"].values()))["n_seeds"].items()) if t["components"] else "", "",
              "| component | δ | τ̂ | " + " | ".join(f"d={d} raw [CI]" for d in t["doses"]) + " | " + " | ".join(f"d={d} fit lower" for d in t["doses"]) + " |",
              "|---|---|---|" + "---|" * (2 * len(t["doses"]))]
        for name, r in t["components"].items():
            raw = " | ".join(_ci(r["raw"][str(d)], fmt="{:.3f}") for d in t["doses"])
            low = " | ".join(f"{x:.3f}" for x in r["ci_lo"])
            L.append(f"| {name} | {r['delta']:.4g} | {r['tau']} | {raw} | {low} |")
        L.append("")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- summary across cells


SUMMARY_ROWS = (  # (key, source, field, group, scale, margin key)
    ("word_count_unseen", "levels", "word_count", "unseen", 1.0, "delta_S_word_count"),
    ("boilerplate_unseen_pct", "levels", "boilerplate", "unseen", 100.0, None),
    ("exact_match_seen_pct", "levels", "exact_match_train", "seen", 100.0, "delta_verbatim"),
    ("rmi_sum_seen", "rmi", "rmi_sum", "seen", 1.0, "delta_V_rmi_sum"),
    ("rmi_norm_seen", "rmi", "rmi_norm", "seen", 1.0, "delta_V_rmi_norm"),
    ("rmi_sum_unseen", "rmi", "rmi_sum", "unseen", 1.0, None),
    ("rmi_norm_unseen", "rmi", "rmi_norm", "unseen", 1.0, None),
    *((f"{m}_{v}_{g}", "levels", f"{m}_{v}", g, 1.0, None) for g in ("seen", "unseen") for m in ("cider", "spice") for v in ("heldout", "train", "all10")),
    *((f"A_{m}_{v}", "A", f"{m}_{v}", None, 1.0, f"delta_A_{m}_heldout" if v == "heldout" else None) for m in ("cider", "spice") for v in ("heldout", "train", "all10")),
    *((f"{k}_{g}", "diversity", k, g, 1.0, None) for g in ("unseen", "seen") for k in ("distinct_1", "distinct_2", "duplicate_rate", "nouns_per_caption")),
)


def _delta_of(res: dict, source: str, field: str, group: str | None) -> dict | None:
    if source == "levels":
        r = res["levels"].get(field, {}).get(group)
        return r and r["diff"]
    if source == "rmi":
        return res["rmi"].get(field, {}).get(group)
    if source == "A":
        return res["A"].get(field)
    r = (res.get("diversity") or {}).get(group, {}).get(field)
    return r and r["diff"]


def exceeds(r: dict, delta: float, scale: float = 1.0) -> bool:
    """95% CI of the change entirely beyond ±δ (either direction)."""
    return bool(scale * r["ci_lo"] > delta or scale * r["ci_hi"] < -delta)


def summarize(pooled: dict[tuple, dict], margins: dict, tau_doc: dict) -> dict:
    """Machine-readable Δ vs d0 (95% CI) per condition × dose × N_exp, with margin marks."""
    rows = []
    for (cond, d, n), res in sorted(pooled.items(), key=lambda kv: (CONDITIONS.index(kv[0][0]), kv[0][2], kv[0][1])):
        row = {"condition": cond, "d": d, "n_exp": n, "K": n * d, "checkpoint_ids": res["checkpoint_ids"], "n_seeds": len(res["checkpoint_ids"]),
               "seen_definition": res["seen_definition"], "n": res["n"], "metrics": {}}
        for key, src, field, group, scale, mkey in SUMMARY_ROWS:
            r = _delta_of(res, src, field, group)
            if r is None:
                continue
            v = {"estimate": scale * r["estimate"], "ci_lo": scale * r["ci_lo"], "ci_hi": scale * r["ci_hi"]}
            if src == "levels":
                lv = res["levels"][field][group]
                v["model"], v["d0"] = scale * lv["model"]["estimate"], scale * lv["d0"]["estimate"]
            if mkey and margins.get(mkey) is not None:
                v["margin"] = scale * margins[mkey]  # in the row's unit (exact match: pp)
                v["exceeds_margin"] = exceeds(r, margins[mkey])
            row["metrics"][key] = v
        rows.append(row)
    taus = {c: {k: {"tau": r["tau"], "delta": r["delta"]} for k, r in t["components"].items()} for c, t in tau_doc["conditions"].items()}
    return {"exploratory": True, "gate_valid": False, "prompt_id": tau_doc["prompt_id"], "margins": margins,
            "margin_rule": "exceeds_margin: the 95% CI of model − d0 lies entirely beyond ±margin (exact match: δ_verbatim is a proportion, 0.05 = 5 pp)",
            "duplicate_rate_note": DUP_NOTE, "rows": rows, "tau": taus}


def _cell_fmt(v: dict | None, fmt: str) -> str:
    if v is None:
        return "–"
    s = f"{fmt.format(v['estimate'])} [{fmt.format(v['ci_lo'])}, {fmt.format(v['ci_hi'])}]"
    return f"**{s}** ▲" if v.get("exceeds_margin") else s


def summary_markdown(summ: dict, baseline_label: str) -> str:
    mg = summ["margins"]
    L = [f"# Exploratory summary, prompt `{summ['prompt_id']}` (Δ = model − {baseline_label}, 95% paired image-bootstrap CIs)", "",
         "EXPLORATORY, not a gate input. Seeds are pooled where a cell has 2 (column `seeds`). Seen = the run's own exposure set (N_exp images). "
         "**Bold ▲** = the CI lies entirely beyond ±margin: " + ", ".join(f"{k} = {v:.4g}" for k, v in mg.items() if v is not None) + ".", ""]

    def table(title, cols, fmt_of, note=None):
        nonlocal L
        L += [f"## {title}", "", "| condition | d | N_exp | seeds | " + " | ".join(c for c, _ in cols) + " |", "|---|---|---|---|" + "---|" * len(cols)]
        for r in summ["rows"]:
            L.append(f"| {r['condition']} | {r['d']} | {r['n_exp']} | {r['n_seeds']} | " + " | ".join(_cell_fmt(r["metrics"].get(k), fmt_of(k)) for _, k in cols) + " |")
        L += ([""] + [note] if note else []) + [""]

    f2 = lambda k: "{:.3f}" if "norm" in k else "{:.2f}"  # noqa: E731
    table("Style and memory", [("word count (unseen)", "word_count_unseen"), ("boilerplate % (unseen)", "boilerplate_unseen_pct"),
                               ("exact match pp (seen)", "exact_match_seen_pct"), ("ΔRMI sum (seen)", "rmi_sum_seen"), ("ΔRMI/token (seen)", "rmi_norm_seen"),
                               ("ΔRMI sum (unseen)", "rmi_sum_unseen"), ("ΔRMI/token (unseen)", "rmi_norm_unseen")], f2)
    for g in ("seen", "unseen"):
        table(f"Quality, {g} (Δ)", [(f"{m.upper()} {v}", f"{m}_{v}_{g}") for m in ("cider", "spice") for v in ("heldout", "all10", "train")], f2)
    L += ["## Quality levels (model; d=0 in the last row of each block is the baseline)", "",
          "| condition | d | N_exp | " + " | ".join(f"{g} {m.upper()} {v}" for g in ("seen", "unseen") for m in ("cider", "spice") for v in ("heldout", "all10")) + " |",
          "|---|---|---|" + "---|" * 8]
    keys = [f"{m}_{v}_{g}" for g in ("seen", "unseen") for m in ("cider", "spice") for v in ("heldout", "all10")]
    for r in summ["rows"]:
        L.append(f"| {r['condition']} | {r['d']} | {r['n_exp']} | " + " | ".join(f"{r['metrics'][k]['model']:.2f}" if k in r["metrics"] else "–" for k in keys) + " |")
        L.append(f"| ↳ {baseline_label} | | (same ids) | " + " | ".join(f"{r['metrics'][k]['d0']:.2f}" if k in r["metrics"] else "–" for k in keys) + " |")
    L.append("")
    table("A Δ (seen − unseen change vs baseline)", [(f"{m.upper()} {v}", f"A_{m}_{v}") for m in ("cider", "spice") for v in ("heldout", "all10", "train")], f2)
    table("Diversity (Δ)", [(f"{k} {g}", f"{k}_{g}") for g in ("unseen", "seen") for k in ("distinct_1", "distinct_2", "duplicate_rate", "nouns_per_caption")],
          lambda k: "{:.2f}" if "nouns" in k else "{:.4f}", DUP_NOTE)
    if summ["tau"]:
        comps = list(dict.fromkeys(k for t in summ["tau"].values() for k in t))
        L += ["## τ split (doses at the main N_exp)", "", "| condition | " + " | ".join(f"{k} (δ)" for k in comps) + " |", "|---|" + "---|" * len(comps)]
        for c, t in summ["tau"].items():
            L.append(f"| {c} | " + " | ".join(f"{t[k]['tau']} ({t[k]['delta']:.4g})" if k in t else "–" for k in comps) + " |")
        L.append("")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- figures


PANELS = (  # (title, y label, source, field, group, scale, reference line key)
    ("Word count (unseen)", "words per caption", "levels", "word_count", "unseen", 1.0, None),
    ("Exact match (seen)", "% captions = training ref", "levels", "exact_match_train", "seen", 100.0, None),
    ("ΔRMI per token (seen)", "model − d0 (nats / token)", "rmi", "rmi_norm", "seen", 1.0, "rmi_norm"),
    ("CIDEr held-out (unseen)", "CIDEr", "levels", "cider_heldout", "unseen", 1.0, None),
    ("SPICE held-out (unseen)", "SPICE", "levels", "spice_heldout", "unseen", 1.0, None),
)
INK, MUTED, GRID = "#52514e", "#8a8984", "#e5e4e0"


def panel_value(res: dict, source: str, field: str, group: str, which: str = "model") -> dict | None:
    """``{estimate, ci_lo, ci_hi}`` of a panel for a cell (``which``: model or d0); ΔRMI is a difference (d0 = 0)."""
    if source == "levels":
        return res["levels"].get(field, {}).get(group, {}).get(which)
    r = res["rmi"].get(field, {}).get(group)
    if r is None:
        return None
    return r if which == "model" else {"estimate": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}


def _errbar(ax, xs, rs, scale, **kw):
    y = np.array([scale * r["estimate"] for r in rs])
    err = np.array([[scale * (r["estimate"] - r["ci_lo"]) for r in rs], [scale * (r["ci_hi"] - r["estimate"]) for r in rs]])
    kw = {"capsize": 3, "lw": 2, "ms": 8, "elinewidth": 1.2, **kw}
    ax.errorbar(xs, y, yerr=np.clip(err, 0, None), **kw)


def _style(ax, title, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)


def _ref_line(ax, ref, refs, x):
    if ref and refs and refs.get(ref) is not None:
        ax.axhline(refs[ref], color=INK, lw=1.2, ls="--")
        ax.annotate(f"δ_V = {refs[ref]:.3g}", (x, refs[ref]), textcoords="offset points", xytext=(2, 4), fontsize=8, color=INK)


def _legend(fig, axes, title_note):
    leg = axes.flat[-1]
    handles, labels = next(a for a in axes.flat if a.get_visible() and a.get_legend_handles_labels()[0]).get_legend_handles_labels()
    leg.axis("off")
    leg.legend(handles, labels, loc="center left", frameon=False, fontsize=10, title=title_note, title_fontsize=9, alignment="left")


def _cond_label(cond: str, by_x: dict, x: str = "d") -> str:
    pooled = [k for k, r in sorted(by_x.items()) if len(r["checkpoint_ids"]) > 1]
    return cond + (f" (2 seeds pooled at {x}={', '.join(map(str, pooled))})" if pooled else "")


def plot_dose(cells: dict[str, dict[int, dict]], path_stem: Path, title: str, refs: dict | None = None, baseline_label: str = "d=0") -> list[str]:
    """``cells``: ``{condition: {d: analysis result}}`` at one N_exp. ``refs``: dashed reference lines (δ_V). Returns the written files."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    doses = sorted({d for by_d in cells.values() for d in by_d})
    any_res = next(iter(next(iter(cells.values())).values()))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8))
    for ax, (ptitle, ylabel, src, field, group, scale, ref) in zip(axes.flat, PANELS):
        d0r = panel_value(any_res, src, field, group, "d0")
        if d0r is None:
            ax.set_visible(False)
            continue
        _errbar(ax, [D0_X], [d0r], scale, color=INK, marker="D", ls="none", label=baseline_label)
        for cond in (c for c in CONDITIONS if c in cells):
            pts = [(d, panel_value(cells[cond][d], src, field, group)) for d in sorted(cells[cond])]
            pts = [(d, r) for d, r in pts if r is not None]
            if not pts:
                continue
            xs = [np.log2(d) for d, _ in pts]
            _errbar(ax, xs, [r for _, r in pts], scale, color=COLORS[cond], marker=MARKERS[cond], label=_cond_label(cond, cells[cond]))
            ax.plot([D0_X, xs[0]], [scale * d0r["estimate"], scale * pts[0][1]["estimate"]], color=COLORS[cond], lw=1, ls=":")
        _ref_line(ax, ref, refs, D0_X)
        ax.axvline(D0_X / 2, color=GRID, lw=1, ls=(0, (2, 2)))
        ax.set_xticks([D0_X] + [np.log2(d) for d in doses], ["0"] + [str(d) for d in doses])
        ax.set_xlabel("dose d (log₂ scale; d=0 left of the break)", fontsize=9, color=INK)
        _style(ax, ptitle, ylabel)
    _legend(fig, axes, "error bars: 95% CI\n(ΔRMI: of model − d0; dashed: δ_V)")
    fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, path_stem)


def plot_ksweep(breadth: dict[str, dict[int, dict]], path_stem: Path, title: str, refs: dict | None = None, baseline_label: str = "d=0") -> list[str]:
    """x = K = N_exp at d=1 (log scale), Paired vs SameDist; seen = each run's own exposure set, so the d=0
    reference of a seen panel is drawn per K (on the same images)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8))
    Ks = sorted({K for by_k in breadth.values() for K in by_k})
    for ax, (ptitle, ylabel, src, field, group, scale, ref) in zip(axes.flat, PANELS):
        first = next(iter(breadth.values()))
        if panel_value(next(iter(first.values())), src, field, group, "d0") is None:
            ax.set_visible(False)
            continue
        if src == "levels":
            d0s = [(K, panel_value(first[K], src, field, group, "d0")) for K in sorted(first)]
            _errbar(ax, [K for K, _ in d0s], [r for _, r in d0s], scale, color=INK, marker="D", ls="--", lw=1, ms=6, mfc="white", label=f"{baseline_label}, same images")
        else:
            ax.axhline(0, color=MUTED, lw=1)
        for cond in (c for c in ("Paired", "SameDist") if c in breadth):
            pts = [(K, panel_value(breadth[cond][K], src, field, group)) for K in sorted(breadth[cond])]
            pts = [(K, r) for K, r in pts if r is not None]
            if pts:
                _errbar(ax, [K for K, _ in pts], [r for _, r in pts], scale, color=COLORS[cond], marker=MARKERS[cond], label=_cond_label(cond, breadth[cond], "K"))
        _ref_line(ax, ref, refs, Ks[0])
        ax.set_xscale("log")
        ax.set_xticks(Ks, [str(k) for k in Ks])
        ax.minorticks_off()
        ax.set_xlabel("K = N_exp exposed images (d = 1, log scale)", fontsize=9, color=INK)
        _style(ax, ptitle.replace("(seen)", "(seen = own exposure set)"), ylabel)
    _legend(fig, axes, "error bars: 95% CI\n(ΔRMI: of model − d0; dashed: δ_V)")
    fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, path_stem)


def _save(fig, stem: Path) -> list[str]:
    import matplotlib.pyplot as plt

    out = []
    for ext in ("png", "pdf"):
        p = stem.with_name(stem.name + f".{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        out.append(str(p))
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------- driver


def cells_of(entries: list[dict]) -> dict[tuple, list[dict]]:
    """``{(condition, d, n_exp): seed entries}`` (sorted by slot); entries without a parsable run id are refused."""
    cells = defaultdict(list)
    for e in entries:
        if e["run"] is None:
            raise SystemExit(f"{e['path']}: checkpoint id {e['id']!r} is not a G<k>_/EX_ run id")
        r = e["run"]
        cells[(r["condition"], r["d"], r["n_exp"])].append(e)
    for k, es in cells.items():
        slots = [e["run"]["slot"] for e in es]
        if len(set(slots)) != len(slots):
            raise SystemExit(f"cell {k}: duplicate seed slots {slots}")
        es.sort(key=lambda e: e["run"]["slot"])
    return dict(cells)


def plan_prompt(label: str, entries: list[dict], ctx: dict) -> dict:
    """Every output of ``run_prompt`` for ``label``, before anything is computed: ``{"cells": [(key, [(stem, seed entries)])],
    "tau", "summary", "dose_fig", "ksweep_fig", "paths"}``."""
    main_n = int(ctx["cfg"]["exposure"]["n_exp"])
    suffix = "" if label == "caption" else f"_{label}"
    out_dir, cells = ctx["out_dir"], cells_of(entries)
    plan: dict = {"cells": [], "suffix": suffix, "by_key": cells}
    for key, es in sorted(cells.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        items = [(out_dir / f"{e['id']}{suffix}", [e]) for e in es]
        if len(es) > 1:
            items.append((out_dir / f"{es[0]['id'].rsplit('_s', 1)[0]}_pooled{suffix}", es))
        plan["cells"].append((key, items))
    plan["tau"] = out_dir / f"tau{suffix or '_caption'}"
    plan["summary"] = out_dir / f"summary{suffix or '_caption'}"
    tag = ctx["fig_tag"] + ("_short" if label != "caption" else "")
    plan["dose_fig"] = ctx["fig_dir"] / f"{tag}_dose" if any(k[2] == main_n for k in cells) else None
    breadth = {k[0] for k in cells if k[1] == 1}
    plan["ksweep_fig"] = ctx["fig_dir"] / f"{tag}_ksweep" if any(k[2] != main_n for k in cells) and ({"Paired", "SameDist"} & breadth) else None
    stems = [st for _, items in plan["cells"] for st, _ in items] + [plan["tau"], plan["summary"]]
    plan["paths"] = [st.with_name(st.name + x) for st in stems for x in (".json", ".md")]
    plan["paths"] += [st.with_name(st.name + x) for st in (plan["dose_fig"], plan["ksweep_fig"]) if st for x in (".png", ".pdf")]
    return plan


def run_prompt(label: str, entries, d0, base, ctx: dict, plan: dict | None = None) -> dict:
    """Per-checkpoint and pooled-cell analyses, τ split, summary and figures for one prompt."""
    cfg, split, excl, seen_defs, kw, writer = ctx["cfg"], ctx["split"], ctx["excl"], ctx["seen_defs"], ctx["kw"], ctx["write"]
    main_n = int(cfg["exposure"]["n_exp"])
    floors = {f: float(cfg["margins"]["delta_v_floor"][f]) for f in RMI_FIELDS}
    for e in [*entries, *d0, *([base] if base else [])]:
        if e["prompt_id"] != label:
            raise SystemExit(f"{e['path']} is prompt {e['prompt_id']!r}, expected {label!r}")
    shas = {e["m"]["provenance"]["generations"]["prompt_sha"] for e in [*entries, *d0]}
    if len(shas) != 1:
        raise SystemExit(f"prompt {label}: checkpoints and baseline were generated from different prompt texts {sorted(shas)}")
    if plan is None:
        plan = plan_prompt(label, entries, ctx)
        ctx["check_new"](plan["paths"])
    baseline_label = ctx.get("baseline_label", {}).get(label, f"d=0 ({len(d0)} seeds)")
    analyze_kw = dict(base=base, nouns=ctx["nouns"], floors=floors, allow_partial=ctx.get("allow_partial", False))
    cells = plan["by_key"]
    pooled: dict[tuple, dict] = {}
    for key, items in plan["cells"]:
        seen = seen_defs(key[2])
        for e in cells[key]:
            if e["run"]["exposure_ids_sha256"] != seen["ids_sha256"]:
                raise SystemExit(f"{e['id']}: run spec exposure ids sha256 {e['run']['exposure_ids_sha256'][:12]} != seen definition N_exp={key[2]} "
                                 f"({seen['ids_sha256'][:12]})")
        for stem, es in items:
            res = analyze(es, d0, split, excl, seen, seen_defs, kw, **analyze_kw)
            title = f"{stem.name} (seeds {[e['run']['slot'] for e in es]}, pooled)" if len(es) > 1 else stem.name
            writer(stem, res, cell_markdown(res, f"{title} vs {baseline_label} ({label})"))
            pooled[key] = res  # the pooled item comes last

    # τ split over the doses at the main N_exp
    seen = seen_defs(main_n)
    paired64 = cells.get(("Paired", 64, main_n)) if label == "caption" else None
    dv = delta_v(paired64, d0, split, excl, seen, cfg, ctx.get("g1_report") if label == "caption" else None)
    dS, dverb = ctx["delta_S"], ctx.get("delta_verbatim", 0.05)
    deltas = {"verbatim_exact_match": (dverb, f"δ_verbatim = {dverb} absolute (seen exact-match change vs d=0; docs/deviations.md 2026-11-01)"),
              "verbatim_exact_match_delta0": (0.0, "secondary: δ = 0 onset (CI lower bound > 0)"),
              "latent_rmi_sum": (dv["rmi_sum"]["delta"], dv["rmi_sum"]["status"]),
              "latent_rmi_norm": (dv["rmi_norm"]["delta"], dv["rmi_norm"]["status"] + " (per-token analogue of the prereg rule)")}
    if dS is not None:
        deltas["style_word_count"] = (dS, "δ_S word count from the G0 margins" + ("" if label == "caption" else " (full-prompt δ_S, not prompt-matched)"))
    taus = {}
    for cond in CONDITIONS:
        series = {k[1]: es for k, es in cells.items() if k[0] == cond and k[2] == main_n}
        if series:
            taus[cond] = tau_split(series, d0, split, excl, seen, kw, deltas)
    tau_doc = {"exploratory": True, "gate_valid": False, "prompt_id": label, "baseline": baseline_label, "delta_v": dv,
               "deltas": {k: {"delta": v[0], "note": v[1]} for k, v in deltas.items()}, "conditions": taus}
    writer(plan["tau"], tau_doc, tau_markdown(taus, dv, label, tau_doc["deltas"]))

    margins = {"delta_S_word_count": dS, "delta_verbatim": dverb, "delta_V_rmi_sum": dv["rmi_sum"]["delta"], "delta_V_rmi_norm": dv["rmi_norm"]["delta"],
               **{f"delta_A_{k}": v for k, v in (ctx.get("delta_A") or {}).items()}}
    summ = summarize(pooled, margins, tau_doc)
    summ["baseline"] = baseline_label
    writer(plan["summary"], summ, summary_markdown(summ, baseline_label))

    figs = []
    refs = {"rmi_norm": dv["rmi_norm"]["delta"]} if label == "caption" else None
    by_cond, breadth = defaultdict(dict), defaultdict(dict)
    for (cond, d, n), res in pooled.items():
        if n == main_n:
            by_cond[cond][d] = res
        if d == 1 and cond in ("Paired", "SameDist"):
            breadth[cond][n] = res
    if plan["dose_fig"]:
        figs += plot_dose(dict(by_cond), plan["dose_fig"], f"Exploratory dose series, split {split['name']}, prompt {label}: N_exp = {main_n}, vs {baseline_label}",
                          refs, baseline_label)
    if plan["ksweep_fig"]:
        figs += plot_ksweep(dict(breadth), plan["ksweep_fig"], f"Exploratory K sweep at d = 1, split {split['name']}, prompt {label}, vs {baseline_label}", refs, baseline_label)
    return {"cells": sorted(pooled), "tau": tau_doc, "summary": summ, "figures": figs}


def main() -> None:
    import datetime

    from contam.config import REPO_ROOT

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--d0", nargs="+", default=None, help="d=0 metrics (default: <repo>/outputs/g0/G0_d0_s*_metrics_v_<split>.json)")
    ap.add_argument("--base", default=None, help="untrained base metrics (levels only; default <repo>/outputs/g0/base_metrics_v.json if present)")
    ap.add_argument("--checkpoints", nargs="*", default=[], help="full-prompt metrics files of trained checkpoints")
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "outputs/runs"), help="run specs <runs-dir>/<GATE>/<id>.json (N_exp, exposure ids sha256)")
    ap.add_argument("--short-d0", nargs="*", default=[], help="short-prompt (nocaps_like) d=0 metrics")
    ap.add_argument("--short-base", default=None, help="short-prompt untrained base metrics (baseline when --short-d0 is empty)")
    ap.add_argument("--short-checkpoints", nargs="*", default=[], help="short-prompt metrics files of trained checkpoints")
    ap.add_argument("--short-prompt-id", default="nocaps_like")
    ap.add_argument("--margins", default=None, help="G0 margins JSON for δ_S / δ_A (default <repo>/outputs/g0/margins_<split>.json if present)")
    ap.add_argument("--g1-report", default=None, help="G1 report JSON whose criteria.a.delta_V is δ_V (default <repo>/outputs/g1/g1_report_<split>.json if present)")
    ap.add_argument("--delta-verbatim", type=float, default=0.05, help="δ_verbatim, absolute seen exact-match change (docs/deviations.md 2026-11-01)")
    ap.add_argument("--tag", default=None, help="output tag (default run_<YYYYMMDD>): tables in <repo>/outputs/explore/<tag>/, figures explore_<split>_<tag>_*")
    ap.add_argument("--out-dir", default=None, help="default <repo>/outputs/explore/<tag>")
    ap.add_argument("--fig-dir", default=str(REPO_ROOT / "outputs/figures"))
    ap.add_argument("--fig-tag", default=None, help="figure file prefix (default explore_<split>_<tag>)")
    ap.add_argument("--cache-dir", default=str(REPO_ROOT / "outputs/explore/cache"), help="noun-count cache (new files only)")
    ap.add_argument("--nouns-backend", default="auto", choices=["auto", "spacy", "nltk"])
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--allow-partial", action="store_true", help="compare checkpoints with partial per-image tables (contam.eval.evaluate.compare)")
    args = ap.parse_args()

    from contam.data.nocaps import load_annotations

    cfg = load_config(args.config)
    split = read_json(args.split)
    sname = split["name"]
    ann = load_annotations(cfg["paths"]["annotations"])
    st = cfg["stats"]
    tag = args.tag or f"run_{datetime.date.today():%Y%m%d}"
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "outputs/explore" / tag
    fig_dir = Path(args.fig_dir)

    def check_new(paths):
        paths = [str(p) for p in paths]
        clash = [p for p in paths if Path(p).exists()]
        dup = sorted({p for p in paths if paths.count(p) > 1})
        if clash or dup:
            raise SystemExit(f"refusing to write: existing {clash}, duplicate targets {dup} (choose a new --tag / --out-dir)")

    def write(stem: Path, obj: dict, md: str):
        j, m = stem.with_name(stem.name + ".json"), stem.with_name(stem.name + ".md")
        check_new([j, m])
        write_json(obj, j)
        m.write_text(md)
        print(f"wrote {j} / {m.name}", flush=True)

    g0 = REPO_ROOT / "outputs/g0"
    d0_paths = args.d0 or sorted(glob.glob(str(g0 / f"G0_d0_s*_metrics_v_{sname}.json")))
    base_path = args.base or (str(g0 / "base_metrics_v.json") if (g0 / "base_metrics_v.json").exists() else None)
    margins_path = args.margins or (str(g0 / f"margins_{sname}.json") if (g0 / f"margins_{sname}.json").exists() else None)
    g1_path = args.g1_report or (str(REPO_ROOT / f"outputs/g1/g1_report_{sname}.json") if (REPO_ROOT / f"outputs/g1/g1_report_{sname}.json").exists() else None)
    margins = read_json(margins_path) if margins_path else None
    load = lambda ps: [load_entry(p, args.runs_dir) for p in ps]  # noqa: E731
    ctx = {"cfg": cfg, "split": split, "excl": load_analysis_exclusion(cfg), "seen_defs": SeenDefs(cfg, split, ann),
           "kw": dict(n_boot=args.n_boot or st["n_boot"], seed=st["boot_seed"], ci=st["ci"]), "write": write, "check_new": check_new,
           "nouns": Nouns(Path(args.cache_dir), args.nouns_backend), "out_dir": out_dir, "fig_dir": fig_dir, "fig_tag": args.fig_tag or f"explore_{sname}_{tag}",
           "delta_S": margins["delta_S"]["word_count"] if margins else None, "delta_A": margins["delta_A"] if margins else None,
           "delta_verbatim": args.delta_verbatim, "g1_report": g1_path, "allow_partial": args.allow_partial, "baseline_label": {}}

    jobs = []  # (label, entries, baseline, base)
    if args.checkpoints:
        jobs.append(("caption", load(args.checkpoints), load(d0_paths), load_entry(base_path) if base_path else None))
    if args.short_checkpoints:
        short_base = load_entry(args.short_base) if args.short_base else None
        baseline = load(args.short_d0) or ([short_base] if short_base else [])
        if not baseline:
            raise SystemExit("--short-checkpoints need --short-d0 or --short-base")
        if not args.short_d0:
            ctx["baseline_label"][args.short_prompt_id] = "untrained base (short prompt; no trained d=0 baseline)"
        jobs.append((args.short_prompt_id, load(args.short_checkpoints), baseline, short_base if args.short_d0 else None))
    plans = {label: plan_prompt(label, entries, ctx) for label, entries, _, _ in jobs}
    check_new([p for pl in plans.values() for p in pl["paths"]])  # every target is new before anything is written
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    report = {label: run_prompt(label, entries, baseline, base, ctx, plans[label]) for label, entries, baseline, base in jobs}
    for label, r in report.items():
        print(f"[{label}] nouns backend {sorted(ctx['nouns'].used)}; figures: {r['figures']}")
    print(json.dumps({"exploratory": True, "prompts": list(report), "out_dir": str(out_dir), "n_boot": ctx["kw"]["n_boot"]}))


if __name__ == "__main__":
    main()
