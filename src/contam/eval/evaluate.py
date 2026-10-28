"""Per-checkpoint evaluation (S, A, V, Q) and paired comparisons against the d=0 baseline.

``evaluate_checkpoint`` scores ONE checkpoint's T=0 generations and returns a metrics dict with
group summaries and a ``per_image`` table (keyed by ``str(image_id)``). Everything that compares
checkpoints (S change, Δ, Γ, ΔRMI, margins, gates) works from these per-image tables so the
paired bootstrap can resample the same images on both sides.

B1 analysis exclusion (prereg Section 2; ``analysis.exclude_image_ids_file``): the listed images are
scored with the rest (one CIDEr IDF corpus, ``per_image`` covers every image), but every group mean,
Δ, CI, margin and gate decision takes its images from ``analysis_ids``, which leaves them out. With
``load_analysis_exclusion(cfg, include=True)`` (``--include-b1-overlap``) the same functions give the
robustness view, which keeps them and is never gate-valid. Every output records the exclusion
(``analysis_exclusion``), and inputs recording another one (or none: metrics from before the
exclusion) are refused (``check_exclusion``).

Per-image fields
----------------
Q:  ``{cider,spice}_{heldout,train,all10}``          (x100; seen+unseen scored jointly)
S:  ``word_count, sentence_count, boilerplate, function_word_ratio``
V:  ``exact_match_train, exact_match_heldout_any, lsn_train, lsn_heldout_max, lsn_heldout_mean``
    and, with teacher-forced log-probs, ``rmi_sum, rmi_norm, train_rank_sum, train_rank_norm,
    logp_train_sum, ntok_train``.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Iterable, Mapping, Sequence

import numpy as np

from contam.config import content_hash, file_sha256, read_jsonl, resolve_path
from contam.data.splits import check_annotations, group_ids
from contam.eval import caption_metrics, memory, stats, style
from contam.eval.association import aligned
from contam.eval.detectors import load_exclusion

GROUPS = ("seen", "unseen", "shared_seen", "all")


# ----------------------------------------------------------------------------- loading


PROVENANCE_FIELDS = ("checkpoint_id", "prompt_sha", "model", "adapter", "batch_size", "max_new_tokens")


def _check_provenance(rows: list[dict], path, expected_checkpoint_id: str | None) -> dict:
    """Each provenance field must take ONE value across ``rows`` (missing fields read as None)."""
    prov = {}
    for f in PROVENANCE_FIELDS:
        vals = {r.get(f) for r in rows}
        if len(vals) > 1:
            raise ValueError(f"{path} mixes {f} values {sorted(map(str, vals))}; write each run to its own file")
        prov[f] = next(iter(vals)) if vals else None
    if expected_checkpoint_id is not None and rows and prov["checkpoint_id"] != expected_checkpoint_id:
        raise ValueError(f"{path} has checkpoint_id {prov['checkpoint_id']!r}, expected {expected_checkpoint_id!r}")
    return prov


def _gen_rows(path, prompt_id, temperature, sample_idx) -> list[dict]:
    return [
        r
        for r in read_jsonl(path)
        if r["prompt_id"] == prompt_id and float(r["temperature"]) == float(temperature) and int(r["sample_idx"]) == sample_idx
    ]


def load_generations(path, prompt_id: str = "caption", temperature: float = 0.0, sample_idx: int = 0, checkpoint_id: str | None = None) -> dict[int, str]:
    """``{image_id: caption}`` for one (prompt, temperature, sample) slice of a generations JSONL.

    Raises if the slice mixes checkpoints / prompt texts / models, or (with ``checkpoint_id``) if it
    belongs to another checkpoint, or if an image has two different captions.
    """
    rows = _gen_rows(path, prompt_id, temperature, sample_idx)
    _check_provenance(rows, path, checkpoint_id)
    out: dict[int, str] = {}
    for r in rows:
        iid = int(r["image_id"])
        if iid in out and out[iid] != r["caption"]:
            raise ValueError(f"conflicting captions for image {iid} in {path}")
        out[iid] = r["caption"]
    return out


def generations_provenance(path, prompt_id: str = "caption", temperature: float = 0.0, sample_idx: int = 0) -> dict:
    return _check_provenance(_gen_rows(path, prompt_id, temperature, sample_idx), path, None)


def _lp_rows(path, prompt_id) -> list[dict]:
    return [r for r in read_jsonl(path) if r.get("prompt_id", prompt_id) == prompt_id]


def load_logprobs(path, prompt_id: str = "caption", n_refs: int = 10, checkpoint_id: str | None = None) -> dict[int, memory.RefLogprobs]:
    """Teacher-forced log-probs JSONL rows ``{image_id, ref_idx, logp_sum, n_tokens, prompt_id, ...}``.

    Every image must have exactly ``n_refs`` rows (ref_idx 0..n_refs-1), each exactly once, and the
    file must come from a single checkpoint / prompt / model (``checkpoint_id`` checks which one).
    """
    raw = _lp_rows(path, prompt_id)
    _check_provenance(raw, path, checkpoint_id)
    rows: dict[int, dict[int, tuple[float, int]]] = {}
    for r in raw:
        iid, j = int(r["image_id"]), int(r["ref_idx"])
        d = rows.setdefault(iid, {})
        if j in d:
            raise ValueError(f"{path}: duplicate log-prob row for image {iid} ref {j}")
        d[j] = (float(r["logp_sum"]), int(r["n_tokens"]))
    out = {}
    for iid, d in rows.items():
        idx = sorted(d)
        if idx != list(range(n_refs)):
            raise ValueError(f"{path}: image {iid} has log-probs for refs {idx}, expected 0..{n_refs - 1}")
        out[iid] = memory.RefLogprobs(np.array([d[j][0] for j in idx]), np.array([d[j][1] for j in idx]))
    return out


def prepare_logprob_resume(path, prov: Mapping[str, object], n_refs: Mapping[int, int], prompt_id: str = "caption") -> set[int]:
    """Resume support for scripts/score_refs.py. Returns the image ids that are fully scored.

    An image is done only when all of its ``n_refs[i]`` references have a row. Rows of partially
    scored images (e.g. a crash mid-chunk) are dropped by atomically rewriting the file, so the
    re-scored rows never duplicate them. A file holding rows from a different checkpoint / prompt
    text / model / adapter is refused.
    """
    if not os.path.exists(path):
        return set()
    rows = list(read_jsonl(path))
    for r in rows:
        if r.get("prompt_id", prompt_id) != prompt_id:
            raise ValueError(f"{path} holds rows for prompt_id {r.get('prompt_id')!r}; use one file per prompt")
        for f, v in prov.items():
            if r.get(f) != v:
                raise ValueError(f"{path} holds rows with {f}={r.get(f)!r} (now {v!r}); write to a new file")
    by_img: dict[int, set[int]] = {}
    for r in rows:
        by_img.setdefault(int(r["image_id"]), set()).add(int(r["ref_idx"]))
    done = {i for i, js in by_img.items() if js == set(range(n_refs[i]))}
    if len(done) != len(by_img):
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            for r in rows:
                if int(r["image_id"]) in done:
                    f.write(json.dumps(r) + "\n")
        os.replace(tmp, path)
    return done


def logprobs_provenance(path, prompt_id: str = "caption") -> dict:
    return _check_provenance(_lp_rows(path, prompt_id), path, None)


# ----------------------------------------------------------------------------- B1 analysis exclusion


def load_analysis_exclusion(cfg: Mapping, include: bool = False) -> dict:
    """The B1 analysis exclusion from ``analysis.exclude_image_ids_file``: ``{file, sha256, image_ids,
    applied}``. ``include`` (robustness view) keeps the images: ``applied`` False."""
    path = cfg["analysis"]["exclude_image_ids_file"]
    return {"file": str(path), "sha256": file_sha256(resolve_path(path)), "image_ids": sorted(load_exclusion(path)), "applied": not include}


def analysis_ids(split: Mapping, group: str, available: Iterable[int], exclusion: Mapping | None = None) -> list[int]:
    """Ids of ``group`` (``all``: every available image) among ``available``, without the excluded
    images when ``exclusion`` is applied. Every group mean / comparison / margin / gate takes its
    images from here."""
    av = {int(i) for i in available}
    drop = set(exclusion["image_ids"]) if exclusion and exclusion["applied"] else set()
    ids = sorted(av) if group == "all" else group_ids(split, group)
    return [i for i in ids if i in av and i not in drop]


def exclusion_record(exclusion: Mapping | None, split: Mapping, available: Iterable[int]) -> dict | None:
    """``exclusion`` as recorded in an output, with ``n_excluded``: the listed images in each group
    (left out when ``applied``, kept in the robustness view)."""
    if exclusion is None:
        return None
    av = [int(i) for i in available]
    listed = set(exclusion["image_ids"])
    return {**exclusion, "n_excluded": {g: sum(i in listed for i in analysis_ids(split, g, av)) for g in GROUPS}}


def check_exclusion(named: Iterable[tuple[str, Mapping]], exclusion: Mapping | None, aggregates: bool = False) -> None:
    """Every ``(name, json)`` input must record the exclusion ``exclusion`` (same file sha256; None:
    none). Metrics from before the exclusion record none and are refused: re-run
    scripts/evaluate_checkpoint.py. ``aggregates`` (margins, gate reports) must also have applied it
    the same way (gate view vs. robustness view)."""
    want = exclusion["sha256"] if exclusion else None
    for name, x in named:
        rec = x.get("analysis_exclusion") or {}
        if rec.get("sha256") != want:
            raise ValueError(f"{name} records B1 analysis exclusion sha256 {rec.get('sha256')}, expected {want}"
                             f"{' (' + exclusion['file'] + ')' if exclusion else ''}; mixed exclusions are refused (re-evaluate inputs made before or under another exclusion)")
        if aggregates and exclusion and rec.get("applied") != exclusion["applied"]:
            raise ValueError(f"{name} was made {'without' if rec.get('applied') else 'with'} the B1 overlap images, this run is "
                             f"{'the robustness view (--include-b1-overlap)' if not exclusion['applied'] else 'a gate view (B1 overlap excluded)'}")


def mark_validity(out: dict, valid: bool, exclusion: Mapping | None) -> None:
    """Set ``gate_valid`` / ``robustness`` of an output. The robustness view (exclusion not applied)
    is never gate-valid; its ``robustness_valid`` says whether it is otherwise valid."""
    robust = bool(exclusion) and not exclusion["applied"]
    out["robustness"] = robust
    out["gate_valid"] = bool(valid) and not robust
    if robust:
        out["robustness_valid"] = bool(valid)
    else:
        out.pop("robustness_valid", None)


def is_valid(x: Mapping, exclusion: Mapping | None) -> bool:
    """``x`` (margins, a gate report) is valid for the view ``exclusion`` selects."""
    robust = bool(exclusion) and not exclusion["applied"]
    return x.get("robustness_valid" if robust else "gate_valid") is True


# ----------------------------------------------------------------------------- evaluation


def evaluate_checkpoint(
    gens: Mapping[int, str],
    annotations: Sequence[dict],
    split: dict,
    logprobs: Mapping[int, memory.RefLogprobs] | None = None,
    boilerplate_patterns: Iterable[str] | None = None,
    metrics: Iterable[str] = caption_metrics.METRICS,
    ref_configs: Iterable[str] = caption_metrics.REF_CONFIGS,
    tokenizer: str = "ptb",
    spice_cache=None,
    require_all: bool = True,
    checkpoint_id: str | None = None,
    with_pos: bool = False,
    exclusion: Mapping | None = None,
) -> dict:
    """Scores every image in ``gens`` (``per_image``); group summaries leave out ``exclusion``."""
    check_annotations(split, annotations)
    refs = {int(a["image_id"]): list(a["refs"]) for a in annotations}
    domain = {int(a["image_id"]): a["domain"] for a in annotations}
    tri = split["train_ref_idx"]
    ids = sorted(int(i) for i in gens)
    per_image: dict[int, dict[str, float]] = {i: {} for i in ids}

    q = caption_metrics.score_split(gens, refs, split, ref_configs, metrics, tokenizer, spice_cache, require_all)
    for cfg, by_metric in q.items():
        for m, scores in by_metric.items():
            for i, s in scores.items():
                per_image[i][f"{m}_{cfg}"] = s

    for i, feats in style.style_per_image({i: gens[i] for i in ids}, boilerplate_patterns, with_pos).items():
        per_image[i].update(feats)

    for i in ids:
        per_image[i].update(memory.text_memory_metrics(gens[i], refs[i], int(tri[str(i)])))
        if logprobs is not None:
            if i not in logprobs:
                raise ValueError(f"no teacher-forced log-probs for image {i}")
            per_image[i].update(memory.likelihood_memory_metrics(logprobs[i], int(tri[str(i)])))

    fields = sorted(per_image[ids[0]]) if ids else []
    summary = {}
    for g in GROUPS:
        gids = analysis_ids(split, g, ids, exclusion)
        summary[g] = {"n": len(gids), **{f: float(np.mean([per_image[i][f] for i in gids])) for f in fields}} if gids else {"n": 0}
    by_domain = {}
    for g in ("seen", "unseen"):
        for d in sorted({domain[i] for i in ids}):
            gids = [i for i in analysis_ids(split, g, ids, exclusion) if domain[i] == d]
            if gids:
                by_domain[f"{g}/{d}"] = {"n": len(gids), **{f: float(np.mean([per_image[i][f] for i in gids])) for f in fields if f.startswith(("cider", "spice", "word_count", "boilerplate"))}}
    return {
        "checkpoint_id": checkpoint_id,
        "split": {"name": split["name"], "content_hash": split["content_hash"]},
        "generations_hash": content_hash({str(i): gens[i] for i in ids}),
        "n_images": len(ids),
        "complete": set(ids) == set(split["seen"]) | set(split["unseen"]),
        "tokenizer": tokenizer,
        "has_logprobs": logprobs is not None,
        "analysis_exclusion": exclusion_record(exclusion, split, ids),
        "summary": summary,
        "by_domain": by_domain,
        "per_image": {str(i): per_image[i] for i in ids},
    }


# ----------------------------------------------------------------------------- comparisons


def per_image_field(metrics_by_seed: Sequence[dict], field: str) -> dict[int, float]:
    """Per-image value of ``field``, pooled (averaged) across the given seeds' metrics dicts."""
    hashes = {m["split"]["content_hash"] for m in metrics_by_seed}
    if len(hashes) != 1:
        raise ValueError("metrics come from different splits")
    return stats.pool_seeds([{int(k): v[field] for k, v in m["per_image"].items()} for m in metrics_by_seed])


S_FIELDS = ("word_count", "boilerplate", "sentence_count", "function_word_ratio")
V_TEXT_FIELDS = ("exact_match_train", "lsn_train", "lsn_heldout_max", "lsn_heldout_mean")
V_LIK_FIELDS = ("rmi_sum", "rmi_norm", "train_rank_sum", "train_rank_norm")


def check_comparable(metrics_list: Sequence[dict], split: dict, allow_partial: bool = False, exclusion: Mapping | None = None) -> None:
    """Raise unless every metrics dict was computed on ``split``, on the complete image set, on
    the same images as the others, and records the B1 exclusion ``exclusion``.

    Mixing splits silently changes the seen/unseen grouping and the training-reference assignment;
    partial image sets change the CIDEr IDF corpus. ``allow_partial`` (smoke tests only) relaxes
    the completeness requirement but never the split, equal-image-set or exclusion requirements.
    """
    check_exclusion(((f"metrics {m.get('checkpoint_id')!r}", m) for m in metrics_list), exclusion)
    for m in metrics_list:
        h = m["split"]["content_hash"]
        if h != split["content_hash"]:
            raise ValueError(f"metrics {m.get('checkpoint_id')!r} were computed on split {m['split']['name']} ({h[:12]}), not on the given split {split['name']} ({split['content_hash'][:12]})")
        if not m.get("complete", False) and not allow_partial:
            raise ValueError(f"metrics {m.get('checkpoint_id')!r} cover a partial image set (CIDEr IDF corpus differs); refusing to compare")
    if allow_partial and not all(m.get("complete", False) for m in metrics_list):
        warn_partial("compare on partial image sets")
    sets = {frozenset(m["per_image"]) for m in metrics_list}
    if len(sets) != 1:
        raise ValueError("metrics cover different image sets; refusing to compare")


def warn_partial(context: str) -> None:
    msg = f"WARNING: {context}: --allow-partial is set; results are NOT valid for gate decisions"
    warnings.warn(msg)
    print("!" * len(msg) + "\n" + msg + "\n" + "!" * len(msg), file=sys.stderr)


def compare(
    model_by_seed: Sequence[dict], base_by_seed: Sequence[dict], split: dict, n_boot: int = 2000, seed: int = 0, ci: float = 0.95, allow_partial: bool = False,
    exclusion: Mapping | None = None,
) -> dict:
    """Paired comparisons of a (seed-pooled) checkpoint against the (seed-pooled) d=0 baseline.

    * S: change (model − d0) of each S field, on unseen (primary) and seen images, with CIs.
    * A: Δ (eq. 3) for every Q field, with CIs; seen and unseen resampled separately.
    * V: change of text memory fields on seen and unseen images; ΔRMI and rank changes when both
      sides have teacher-forced log-probs.
    All values are raw (model − d0); gate logic applies expected directions. Inputs are validated
    with ``check_comparable``; the output's ``gate_valid`` is False when ``allow_partial`` was used or
    in the robustness view. Images in an applied ``exclusion`` are left out.
    """
    check_comparable(list(model_by_seed) + list(base_by_seed), split, allow_partial, exclusion)
    kw = dict(n_boot=n_boot, seed=seed, ci=ci)
    m0 = model_by_seed[0]["per_image"]
    fields = set(next(iter(m0.values())))
    for m in list(model_by_seed) + list(base_by_seed):
        fields &= set(next(iter(m["per_image"].values())))
    avail = [int(k) for k in m0]
    seen, unseen = analysis_ids(split, "seen", avail, exclusion), analysis_ids(split, "unseen", avail, exclusion)

    def arr(metrics_list, field, ids):
        return aligned(per_image_field(metrics_list, field), ids)

    out: dict = {"analysis_exclusion": exclusion_record(exclusion, split, avail), "n_model_seeds": len(model_by_seed), "n_base_seeds": len(base_by_seed), "n_seen": len(seen), "n_unseen": len(unseen), "S": {}, "A": {}, "V": {}}
    mark_validity(out, all(m.get("complete", False) for m in list(model_by_seed) + list(base_by_seed)), exclusion)
    for f in S_FIELDS:
        if f in fields:
            out["S"][f] = {g: stats.bootstrap_paired_diff(arr(model_by_seed, f, ids), arr(base_by_seed, f, ids), **kw).to_dict() for g, ids in (("unseen", unseen), ("seen", seen))}
    for f in sorted(x for x in fields if x.startswith(("cider_", "spice_"))):
        out["A"][f] = stats.bootstrap_delta(arr(model_by_seed, f, seen), arr(base_by_seed, f, seen), arr(model_by_seed, f, unseen), arr(base_by_seed, f, unseen), **kw).to_dict()
    for f in V_TEXT_FIELDS + V_LIK_FIELDS:
        if f in fields:
            out["V"][f] = {g: stats.bootstrap_paired_diff(arr(model_by_seed, f, ids), arr(base_by_seed, f, ids), **kw).to_dict() for g, ids in (("seen", seen), ("unseen", unseen))}
    return out
