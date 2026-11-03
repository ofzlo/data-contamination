#!/usr/bin/env python
"""Format-matched style vs. vocabulary transfer on short-prompt (``nocaps_like``) generations.

EXPLORATORY (docs/deviations.md, 2026-11-02): no gate input; every output records
``exploratory: true``, ``gate_valid: false``. Stands in for the pending B4 style classifier.
Metrics: ``contam.eval.style_transfer``. Groups: ``contam.eval.evaluate.analysis_ids`` (seen = the
N_exp exposure set, B1 analysis exclusion applied); every metric is on UNSEEN images.

(a) Stylistic distance: Jensen–Shannon divergence (base 2) of the outputs' first-two-word,
    function-word and POS-bigram distributions to (1) the NoCaps references of the seen_unexposed
    images only (never evaluated here, never trained on) and (2) a fixed random sample of COCO
    train2017 captions. Δ vs. the five d=0 seeds.
(b) Vocabulary transfer: Open Images class names mentioned (as nouns) per caption, the subset also
    named in that image's NoCaps references, precision (matched / mentioned) and recall (matched /
    named in the references), for all classes, the non-COCO classes and the rare non-COCO classes,
    overall and per NoCaps domain; Δ vs. d=0 and Paired − SameDist (seed-matched). Non-COCO (primary) excludes
    COCO names, aliases, human body parts and near-aliases (``st.OI_COCO_NEAR_ALIASES``); ``*_old`` = the
    first, wider set without the body parts / near-aliases (sensitivity).

Bootstrap: one shared matrix of image multiplicities (unseen images resampled, B replicates,
``stats.boot_seed``) applied to every corpus, so every Δ is paired; levels average per-seed corpus
statistics; bias-shifted percentile CIs (``contam.eval.diversity``). References are fixed.

Outputs ``<out-dir>/<stem>.{json,md}``; refuses to overwrite.

Example:
    python scripts/analysis/style_transfer.py --split data/splits/split_A.json \\
        --out-dir outputs/explore/style_20261102 --stem style_transfer_A
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from contam.config import load_config, read_json, write_json
from contam.data.nocaps import load_annotations
from contam.data.splits import normalize_domain
from contam.eval import style_transfer as st
from contam.eval.evaluate import analysis_ids, load_analysis_exclusion, load_generations, load_seen_definition, seen_record

PROMPT = "nocaps_like"
RUN_ID = re.compile(r"^(?P<gate>G\d+|EX)_(?P<cond>[A-Za-z]+)_d(?P<d>0|[1-9]\d*)(?:_n(?P<n>[1-9]\d*))?_s(?P<slot>0|[1-9]\d*)$")
EXPECTED = (
    ["base"] + [f"G0_d0_s{s}" for s in range(5)]
    + [f"G2_{c}_d1_s0" for c in ("Paired", "Caption", "SameDist")] + ["G2_Paired_d1_s1"]
    + [f"G1_{c}_d{d}_s{s}" for c in ("Paired", "Caption", "SameDist") for d in (4, 64) for s in (0, 1)]
    + [f"EX_{c}_d2_n1000_s0" for c in ("Paired", "Caption")]
    + [f"EX_{c}_d1_n{n}_s0" for c in ("Paired", "SameDist") for n in (30, 100, 300)]
)
D0_KEY = ("G0", "d0", 0, None)
RARE_PER_1000 = 1.0  # a non-COCO class is "rare" if named (as a noun) in < 1 per 1,000 COCO sample captions


def cell_key(run_id: str):
    m0 = re.match(r"^G0_d0_s(0|[1-9]\d*)$", run_id)
    if m0:
        return D0_KEY, int(m0[1])
    m = RUN_ID.match(run_id)
    if not m:
        return None
    return (m["gate"], m["cond"], int(m["d"]), int(m["n"]) if m["n"] else None), int(m["slot"])


def cell_name(k) -> str:
    gate, cond, d, n = k
    return f"{gate}_{cond}_d{d}" + (f"_n{n}" if n else "")


def fmt(x, nd=4) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def fci(s, nd=4) -> str:
    if s["estimate"] is None:
        return "–"
    short = f" (B={s['n_boot_used']})" if s.get("n_boot_used", s["n_boot"]) < s["n_boot"] else ""  # CI from fewer finite replicates
    return f"{s['estimate']:+.{nd}f} [{s['ci_lo']:+.{nd}f}, {s['ci_hi']:+.{nd}f}]" + (" *" if s["ci_lo"] > 0 or s["ci_hi"] < 0 else "") + short


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--short-dir", default="outputs/short")
    ap.add_argument("--oi-classes", default=None, help="Open Images class-descriptions-boxable.csv (default: <data_root>/openimages/class-descriptions-boxable.csv)")
    ap.add_argument("--coco-sample", type=int, default=30000, help="COCO train2017 captions sampled for the COCO reference")
    ap.add_argument("--samedist", default=None, help="SameDist file (default: samedist_<split>.json next to --split), for the COCO-reference overlap count")
    ap.add_argument("--n-boot", type=int, default=None, help="default: stats.n_boot")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default="style_transfer_A")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    targets = [out_dir / f"{args.stem}.json", out_dir / f"{args.stem}.md"]
    for t in targets:
        if t.exists():
            raise SystemExit(f"refusing to overwrite {t}")
    cfg = load_config(args.config)
    st_cfg = cfg["stats"]
    n_boot = int(args.n_boot or st_cfg["n_boot"])
    split = read_json(args.split)
    ann = load_annotations(cfg["paths"]["annotations"])
    by_id = {int(a["image_id"]): a for a in ann}
    excl = load_analysis_exclusion(cfg)
    seen = load_seen_definition(cfg, split, ann)

    # ---- generations
    files = {Path(p).name[: -len(f"_{PROMPT}.jsonl")]: p for p in sorted(glob.glob(os.path.join(args.short_dir, f"*_{PROMPT}.jsonl")))}
    runs = {r: p for r, p in files.items() if r == "base" or cell_key(r)}
    gens = {r: load_generations(p, PROMPT, checkpoint_id=None if r == "base" else r) for r, p in runs.items()}
    unseen = analysis_ids(split, "unseen", by_id, excl, seen)
    incomplete = sorted(r for r, g in gens.items() if not set(unseen) <= set(g))  # still generating: left out
    gens = {r: g for r, g in gens.items() if r not in incomplete}
    missing = [r for r in EXPECTED if r not in gens]
    unexp = analysis_ids(split, "seen_unexposed", by_id, excl, seen)
    if not unseen or not unexp:
        raise SystemExit("empty unseen / seen_unexposed group")
    domain = np.array([normalize_domain(by_id[i]["domain"]) for i in unseen])
    print(f"{len(gens)} complete runs ({len(missing)} expected missing; incomplete, skipped: {incomplete}); unseen n={len(unseen)}; seen_unexposed refs from {len(unexp)} images")

    # ---- references
    nocaps_refs = [r for i in unexp for r in by_id[i]["refs"]]
    coco = read_json(cfg["paths"]["coco_captions"])["annotations"]
    rng = np.random.default_rng(st_cfg["boot_seed"])
    coco_pick = sorted(rng.choice(len(coco), size=min(args.coco_sample, len(coco)), replace=False))
    coco_refs = [coco[i]["caption"].strip() for i in coco_pick]
    coco_ann_ids = {int(coco[i]["id"]) for i in coco_pick}
    del coco
    sd_path = args.samedist or str(Path(args.split).with_name(f"samedist_{split['name']}.json"))
    sd_overlap = None
    if Path(sd_path).exists():  # SameDist training captions that are also in the COCO reference sample (upper bound: every SameDist row)
        sd_ids = {int(r["coco_ann_id"]) for r in read_json(sd_path)["rows"]}
        sd_overlap = {"file": sd_path, "n_samedist_rows": len(sd_ids), "n_in_coco_sample": len(sd_ids & coco_ann_ids)}

    # ---- POS tagging (distinct texts, once)
    t0 = time.time()
    unseen_refs = {i: by_id[i]["refs"] for i in unseen}
    texts = [g[i] for g in gens.values() for i in unseen] + nocaps_refs + coco_refs + [r for rs in unseen_refs.values() for r in rs]
    tagged = st.tag_texts(texts)
    print(f"tagged {len(tagged)} distinct texts in {time.time() - t0:.0f}s")

    # ---- distribution features
    feats = {r: [st.feature_counters(g[i], tagged[g[i]]) for i in unseen] for r, g in gens.items()}
    ref_feats = {"nocaps_ref": [st.feature_counters(t, tagged[t]) for t in nocaps_refs], "coco": [st.feature_counters(t, tagged[t]) for t in coco_refs]}
    vocabs = {f: st.Vocab() for f in st.FEATURES}
    for f, v in vocabs.items():
        for rows in [*feats.values(), *ref_feats.values()]:
            v.add(c[f] for c in rows)
    ref_vec = {f: {name: vocabs[f].vector(c[f] for c in rows) for name, rows in ref_feats.items()} for f in st.FEATURES}

    # ---- vocabulary
    oi_path = args.oi_classes or os.path.join(cfg["paths"]["data_root"], "openimages", "class-descriptions-boxable.csv")
    names = st.load_oi_classes(oi_path)
    matcher = st.ClassMatcher(names)
    noncoco = {n for n in names if not st.coco_overlap(n)}
    noncoco_old = {n for n in names if not st.coco_overlap(n, near_aliases=False)}
    coco_hits = defaultdict(int)
    for t in coco_refs:
        for n in matcher(tagged[t]):
            coco_hits[n] += 1
    is_rare = lambda n: 1000.0 * coco_hits[n] / len(coco_refs) < RARE_PER_1000  # noqa: E731
    rare, rare_old = {n for n in noncoco if is_rare(n)}, {n for n in noncoco_old if is_rare(n)}
    class_sets = {"oi_all": set(names), "oi_noncoco": noncoco, "oi_rare": rare, "oi_noncoco_old": noncoco_old, "oi_rare_old": rare_old}
    ref_named = [set().union(*(matcher(tagged[r]) for r in unseen_refs[i])) for i in unseen]

    # ---- replicates (one shared weight matrix)
    W = st.boot_weights(len(unseen), n_boot, st_cfg["boot_seed"])
    reps = {}
    for r in gens:
        d = {}
        for f in st.FEATURES:
            for ref, v in st.js_replicates(vocabs[f].matrix([c[f] for c in feats[r]]), ref_vec[f], W).items():
                d[f"js/{f}/{ref}"] = v
        mentioned = [matcher(tagged[gens[r][i]]) for i in unseen]
        d.update(st.vocab_replicates(st.vocab_rows(mentioned, ref_named, class_sets), domain, W))
        reps[r] = d
    ci = st_cfg["ci"]

    # ---- cells and comparisons
    cells = defaultdict(dict)
    for r in gens:
        if r != "base":
            k, slot = cell_key(r)
            cells[k][slot] = r
    d0_runs = [cells[D0_KEY][s] for s in sorted(cells.get(D0_KEY, {}))]
    if len(d0_runs) != 5:
        print(f"WARNING: {len(d0_runs)} d=0 seeds (expected 5)")
    d0 = [reps[r] for r in d0_runs]
    out = {
        "exploratory": True, "gate_valid": False, "prompt_id": PROMPT, "split": split["name"], "split_hash": split["content_hash"],
        "seen_definition": seen_record(seen), "analysis_exclusion": {"file": str(excl["file"]), "sha256": excl["sha256"], "n_ids": len(excl["image_ids"])},
        "n_unseen": len(unseen), "n_unseen_by_domain": {dm: int((domain == dm).sum()) for dm in st.DOMAINS},
        "n_boot": n_boot, "boot_seed": st_cfg["boot_seed"], "ci": ci, "ci_method": "bias-shifted percentile (as contam.eval.diversity)",
        "references": {
            "nocaps_ref": {"what": "NoCaps references of the seen_unexposed images (B1 exclusion applied)", "n_images": len(unexp), "n_captions": len(nocaps_refs)},
            "coco": {"what": f"random sample of COCO train2017 captions (seed {st_cfg['boot_seed']})", "n_captions": len(coco_refs), "file": str(cfg["paths"]["coco_captions"])},
        },
        "oi_classes": {"file": str(oi_path), "n": len(names), "n_noncoco": len(noncoco), "n_rare": len(rare), "rare_rule": f"non-COCO, named as a noun in < {RARE_PER_1000} per 1,000 COCO sample captions",
                       "n_noncoco_old": len(noncoco_old), "n_rare_old": len(rare_old),
                       "moved_to_coco_overlap": sorted(noncoco_old - noncoco), "near_aliases": st.OI_COCO_NEAR_ALIASES,
                       "primary": "oi_noncoco / oi_rare (COCO names, aliases, body parts and near-aliases removed); *_old = first, wider set (sensitivity)"},
        "samedist_in_coco_reference": sd_overlap,
        "pos_tagger": "spacy en_core_web_sm (UPOS)",
        "runs_used": sorted(gens), "runs_missing": missing, "runs_incomplete_skipped": incomplete, "d0_runs": d0_runs,
        "levels": {"base": st.level([reps["base"]], ci) if "base" in reps else None, "d0": st.level(d0, ci),
                   "d0_per_seed_point": {r: {k: float(v[0]) for k, v in reps[r].items()} for r in d0_runs}},
        "vs_d0": {}, "paired_minus_samedist": {},
    }
    for k in sorted(cells, key=lambda k: (k[0], k[1], k[2], k[3] or 0)):
        if k == D0_KEY:
            continue
        rs = [cells[k][s] for s in sorted(cells[k])]
        out["vs_d0"][cell_name(k)] = {"runs": rs, **st.compare_replicates([reps[r] for r in rs], d0, ci)}
    for k in sorted(cells, key=lambda k: (k[0], k[2], k[3] or 0)):
        if k[1] != "Paired":
            continue
        ks = (k[0], "SameDist", k[2], k[3])
        if ks not in cells:
            continue
        slots = sorted(set(cells[k]) & set(cells[ks]))
        if not slots:
            continue
        name = cell_name(k).replace("_Paired", "")
        out["paired_minus_samedist"][name] = {"seeds": slots, **st.compare_replicates([reps[cells[k][s]] for s in slots], [reps[cells[ks][s]] for s in slots], ci)}
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out, targets[0])
    targets[1].write_text(render_md(out))
    print(f"-> {targets[0]}\n-> {targets[1]}")


INTERPRETATION = """\
**Interpretation rule (fixed before reading the numbers, 2026-11-02).** Compare the seed-matched
Paired − SameDist unseen difference in the vocabulary metrics across NoCaps domains:

* gain concentrated in **out-domain** (and near-domain), little or none in-domain → **vocabulary
  transfer** (Paired teaches NoCaps object names that generalize to unseen images);
* gain spread **evenly** across domains (and mirrored in the JS distance to the NoCaps references) →
  **style** (format / phrasing, not object vocabulary);
* **in-domain** images contain COCO classes, and SameDist was trained on COCO captions, so in-domain
  may favour SameDist; this is flagged separately and not read as evidence against vocabulary
  transfer.
"""


def render_md(o: dict) -> str:
    L = ["# Format-matched style vs. vocabulary transfer (short prompt `nocaps_like`, split A)", "",
         "EXPLORATORY: not a gate input (`gate_valid: false`); stands in for the pending B4 style classifier. "
         "All metrics on **unseen** images (seen = N_exp = 1,000 exposure set; B1 exclusion applied); "
         f"n = {o['n_unseen']} ({', '.join(f'{k} {v}' for k, v in o['n_unseen_by_domain'].items())}).",
         f"Image bootstrap: B = {o['n_boot']:,} (seed {o['boot_seed']}), one shared resampling of unseen images for every corpus, so every Δ is paired; "
         f"{o['ci']:.0%} {o['ci_method']} CIs; levels are means over training seeds of per-seed corpus statistics; d=0 = the five d=0 seeds. "
         "References are fixed (not resampled). `*` = CI excludes 0.", "",
         f"References: NoCaps refs of {o['references']['nocaps_ref']['n_images']} seen_unexposed images ({o['references']['nocaps_ref']['n_captions']} captions; disjoint from the evaluated images); "
         f"COCO train2017 sample ({o['references']['coco']['n_captions']} captions). Open Images classes: {o['oi_classes']['n']} boxable "
         f"({o['oi_classes']['n_noncoco']} non-COCO, {o['oi_classes']['n_rare']} rare: {o['oi_classes']['rare_rule']}); matched as nouns, plural-tolerant; COCO overlap is name/alias based (approximate).", ""]
    if o["runs_missing"]:
        L += [f"Runs missing at this run ({len(o['runs_missing'])}): {', '.join(o['runs_missing'])}", ""]
    oc = o["oi_classes"]
    L += [f"Non-COCO class set (corrected, primary): {oc['n_noncoco']} classes ({oc['n_rare']} rare); the first set (`*_old`, sensitivity: "
          f"{oc['n_noncoco_old']} / {oc['n_rare_old']} rare) also counted these {len(oc['moved_to_coco_overlap'])} human body parts and near-aliases "
          f"of COCO classes as non-COCO: {', '.join(oc['moved_to_coco_overlap'])}.", ""]
    sd = o.get("samedist_in_coco_reference")
    L += ["Caveats:", "",
          "* **COCO reference vs SameDist.** The COCO reference is a sample of COCO train2017 captions, the source of SameDist's training captions; "
          + (f"{sd['n_in_coco_sample']} of the {sd['n_samedist_rows']} SameDist captions (all rows, an upper bound for any one run) are in the "
             f"{o['references']['coco']['n_captions']}-caption sample. " if sd else "the overlap was not counted. ")
          + "SameDist's JS distance to `coco` can be lowered by having trained on (part of) that reference; read SameDist vs `coco` with this in mind.",
          "* **first2 carries content.** The first two words are often a determiner + object (\"a dog\", \"two men\"), so the first-two-word "
          "distribution mixes phrasing with object content; it is not a pure style feature (function words and POS bigrams are closer to one).",
          "* CIs: `n_boot_used` in the JSON = number of finite (non-NaN) bootstrap replicates each CI uses (precision / recall are NaN in a replicate "
          "with no mentions / no reference names).", ""]
    L += [INTERPRETATION, "## (a) Stylistic distance: JS divergence to the reference distributions", ""]
    js_keys = [f"js/{f}/{r}" for f in st.FEATURES for r in ("nocaps_ref", "coco")]
    lv = o["levels"]
    L += ["Levels (JS, base 2):", "", "| corpus | " + " | ".join(k[3:] for k in js_keys) + " |", "|---" * (len(js_keys) + 1) + "|"]
    if lv["base"]:
        L.append("| base | " + " | ".join(fmt(lv["base"][k]["estimate"]) for k in js_keys) + " |")
    L.append("| d=0 (5 seeds) | " + " | ".join(fmt(lv["d0"][k]["estimate"]) for k in js_keys) + " |")
    sds = {k: float(np.std([v[k] for v in lv["d0_per_seed_point"].values()], ddof=1)) for k in js_keys} if len(lv["d0_per_seed_point"]) > 1 else {}
    if sds:
        L.append("| d=0 seed SD | " + " | ".join(fmt(sds[k]) for k in js_keys) + " |")
    for name, c in o["vs_d0"].items():
        L.append(f"| {name} ({len(c['runs'])}s) | " + " | ".join(fmt(c[k]["model"]["estimate"]) for k in js_keys) + " |")
    L += ["", "Δ vs d=0 (model − d0; negative = closer to the reference):", "", "| cell | " + " | ".join(k[3:] for k in js_keys) + " |", "|---" * (len(js_keys) + 1) + "|"]
    for name, c in o["vs_d0"].items():
        L.append(f"| {name} | " + " | ".join(fci(c[k]["diff"]) for k in js_keys) + " |")
    L += ["", "Paired − SameDist (seed-matched):", "", "| dose | " + " | ".join(k[3:] for k in js_keys) + " |", "|---" * (len(js_keys) + 1) + "|"]
    for name, c in o["paired_minus_samedist"].items():
        L.append(f"| {name} (s{','.join(map(str, c['seeds']))}) | " + " | ".join(fci(c[k]["diff"]) for k in js_keys) + " |")

    L += ["", "## (b) Vocabulary transfer (Open Images class names, unseen images)", "",
          "`mentions` = class names named per caption; `matched` = of those, also named in that image's NoCaps references (per caption); "
          "`precision` = matched / mentioned; `recall` = matched / named in the references.", ""]
    for cs in ("oi_noncoco", "oi_rare", "oi_all", "oi_noncoco_old", "oi_rare_old"):
        L += [f"### {cs}" + (" (SENSITIVITY: first, wider non-COCO set)" if cs.endswith("_old") else " (primary)" if cs != "oi_all" else ""), "", "Levels (all / in / near / out):", "", "| corpus | stat | all | in | near | out |", "|---|---|---|---|---|---|"]
        for stat in st.VOCAB_STATS:
            if lv["base"]:
                L.append(f"| base | {stat} | " + " | ".join(fmt(lv["base"][f"{sl}/{cs}/{stat}"]["estimate"]) for sl in st.SLICES) + " |")
            L.append(f"| d=0 | {stat} | " + " | ".join(fmt(lv["d0"][f"{sl}/{cs}/{stat}"]["estimate"]) for sl in st.SLICES) + " |")
        for stat in ("mentions", "matched", "precision"):
            L += ["", f"Δ vs d=0, `{stat}`:", "", "| cell | all | in | near | out |", "|---|---|---|---|---|"]
            for name, c in o["vs_d0"].items():
                L.append(f"| {name} | " + " | ".join(fci(c[f"{sl}/{cs}/{stat}"]["diff"]) for sl in st.SLICES) + " |")
            L += ["", f"Paired − SameDist (seed-matched), `{stat}`:", "", "| dose | all | in | near | out |", "|---|---|---|---|---|"]
            for name, c in o["paired_minus_samedist"].items():
                L.append(f"| {name} | " + " | ".join(fci(c[f"{sl}/{cs}/{stat}"]["diff"]) for sl in st.SLICES) + " |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
