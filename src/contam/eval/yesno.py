"""Phase 0: first-token yes/no scores on the NoCaps object questions, and the preregistered analysis.

Questions come from ``data/phase0/nocaps_yesno_<split>.share.json`` (``contam.data.oi_questions``).

Scoring (one checkpoint)
------------------------
Each question is asked as ``PROMPT_TEMPLATE`` (LLaVA-1.5's own short-answer instruction, the one its
POPE evaluation uses) under the model's chat template. The score is the log-odds of the first answer
token,

    score = log P(yes) - log P(no),   log P(yes) = logsumexp over the YES_WORDS token ids (same for no),

read from fp32 logits (``HFVLM.first_token_logprobs``). ``score > 0`` counts as answering yes. ``mass``
= P(yes) + P(no) is stored as a check that the model really answers with yes / no.

bf16 forward passes depend on batch composition, so batches are FIXED: questions sorted by
(image_id, qid) are cut into chunks of ``batch_size``; every checkpoint sees the same chunks. A chunk
with any missing row is rescored whole (its stale rows are dropped first). A file holding rows of
another checkpoint, model, adapter, prompt, question file, batch size or token ids is refused.

Analysis (prereg, Phase 0 amendment)
------------------------------------
For a checkpoint exposed to N_exp images the groups are the question file's ``group[str(N_exp)]``:
``seen`` and ``unseen``; ``unexposed`` is left out. Per analysis cell (``yes/direct``,
``yes/descendant``, ``yes/none``, ``no``) the effect is eq. 3 on the score,

    Δ = (mean_seen(run) - mean_seen(d0)) - (mean_unseen(run) - mean_unseen(d0)),

means over questions, d0 = each question's score averaged over the five d=0 seeds. CIs come from the
image-level (cluster) paired bootstrap: images are resampled within seen and unseen, each with all
of its questions, and every checkpoint is evaluated on the same draws. The margin of a cell is
δ = 2 × SD (ddof=1) across the five d=0 seeds of their seen-minus-unseen gap (the D5 convention).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from contam.config import append_jsonl, file_sha256, prompt_sha, read_json, read_jsonl
from contam.eval.stats import paired_bootstrap, sd_margin

# Preregistered Phase 0 settings (prereg.md, Amendments, 2026-11-12: locked).
PREREG_QUESTIONS = "data/phase0/nocaps_yesno_A.share.json"
PREREG_QUESTIONS_SHA256 = "635f8a002c2f56ed632fe0a9246c5bcbf374f165a04c75b6951475b4595389a0"
PREREG_PRIMARY = "G1_Paired_d4"   # the one run cell that decides P2 / familiarity (seeds s0, s1)
PREREG_BATCH_SIZE = 16
PREREG_N_BOOT = 10_000
PREREG_BOOT_SEED = 0
# Body-part "no" questions are left out of every analysis (scored, never analysed): the question drops
# "human" ("Is there a head?" is true on an animal photo) and many are marked absent on photos with people.
PREREG_EXCLUDE_BODY_PART_NO = True

PROMPT_TEMPLATE = "{question}\nAnswer the question using a single word or phrase."
YES_WORDS = ("Yes", "yes")
NO_WORDS = ("No", "no")
CELLS = ("yes/direct", "yes/descendant", "yes/none", "no")
N_EXP_COLUMNS = ("1000", "300", "100", "30")
ROW_FIELDS = ("checkpoint_id", "model", "adapter", "prompt_sha", "questions_sha256", "batch_size", "yes_ids", "no_ids",
              "qid", "image_id", "logp_yes", "logp_no", "score", "mass")
PROV_FIELDS = ("checkpoint_id", "model", "adapter", "prompt_sha", "questions_sha256", "batch_size", "yes_ids", "no_ids")


# ----------------------------------------------------------------------------- questions


def load_questions(path, expected_sha256: str | None = None) -> tuple[list[dict], str]:
    """The share file's questions sorted by (image_id, qid), and the file's sha256 (checked if given)."""
    sha = file_sha256(path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(f"{path}: sha256 {sha} is not the preregistered {expected_sha256}")
    qs = read_json(path)["questions"]
    seen_qid = set()
    for q in qs:
        if q["qid"] in seen_qid:
            raise ValueError(f"{path}: duplicate qid {q['qid']}")
        seen_qid.add(q["qid"])
        if q["answer"] not in ("yes", "no") or q["cell"] not in CELLS:
            raise ValueError(f"{path}: question {q['qid']} has answer {q['answer']!r} / cell {q['cell']!r}")
        if (q["answer"] == "no") != (q["cell"] == "no"):
            raise ValueError(f"{path}: question {q['qid']}: answer {q['answer']} does not fit cell {q['cell']}")
        if set(q["group"]) != set(N_EXP_COLUMNS) or not set(q["group"].values()) <= {"seen", "unseen", "unexposed"}:
            raise ValueError(f"{path}: question {q['qid']} has group {q['group']}")
    return sorted(qs, key=lambda q: (int(q["image_id"]), q["qid"])), sha


def analysis_questions(questions: Sequence[Mapping], exclude_body_part_no: bool = PREREG_EXCLUDE_BODY_PART_NO,
                       exclude_body_parts: bool = False) -> list[Mapping]:
    """The questions the analysis uses: without body-part "no" questions (prereg), and optionally
    without any body-part question (robustness view)."""
    out = []
    for q in questions:
        bp = bool(q.get("is_body_part"))
        if bp and (exclude_body_parts or (exclude_body_part_no and q["answer"] == "no")):
            continue
        out.append(q)
    return out


def question_prompt(q: Mapping) -> str:
    return PROMPT_TEMPLATE.format(question=q["question"])


# ----------------------------------------------------------------------------- scoring


def answer_token_ids(tokenizer, words: Sequence[str]) -> list[int]:
    """Vocabulary ids of ``words`` as a whole first token, with and without the SentencePiece word marker."""
    ids = set()
    for w in words:
        for tok in ("▁" + w, w):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != tokenizer.unk_token_id:
                ids.add(int(tid))
    if not ids:
        raise ValueError(f"none of {list(words)} is a single token of the tokenizer")
    return sorted(ids)


def _lse(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))[:, 0]


def chunks(questions: Sequence[Mapping], batch_size: int) -> list[list[Mapping]]:
    qs = sorted(questions, key=lambda q: (int(q["image_id"]), q["qid"]))
    return [qs[s : s + batch_size] for s in range(0, len(qs), batch_size)]


def _rewrite_without(path, qids: set[str]) -> None:
    rows = [r for r in read_jsonl(path) if r["qid"] not in qids]
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def score_questions(
    vlm,
    load_image: Callable[[int], object],
    questions: Sequence[Mapping],
    out_path,
    checkpoint_id: str,
    questions_sha256: str,
    batch_size: int = 16,
    progress: bool = True,
) -> int:
    """Score the missing questions of ``questions`` and append rows to ``out_path``; returns rows written."""
    tok = vlm.processor.tokenizer
    yes_ids, no_ids = answer_token_ids(tok, YES_WORDS), answer_token_ids(tok, NO_WORDS)
    if set(yes_ids) & set(no_ids):
        raise ValueError("yes and no token ids overlap")
    prov = dict(checkpoint_id=checkpoint_id, model=getattr(vlm, "model_id", None), adapter=getattr(vlm, "adapter", None),
                prompt_sha=prompt_sha(PROMPT_TEMPLATE), questions_sha256=questions_sha256, batch_size=int(batch_size),
                yes_ids=yes_ids, no_ids=no_ids)
    done: set[str] = set()
    if Path(out_path).exists():
        for r in read_jsonl(out_path):
            for f in PROV_FIELDS:
                if r.get(f) != prov[f]:
                    raise ValueError(f"{out_path} has rows with {f}={r.get(f)!r} (now {prov[f]!r}); write to a new file")
            done.add(r["qid"])
    todo = [c for c in chunks(questions, batch_size) if any(q["qid"] not in done for q in c)]
    stale = {q["qid"] for c in todo for q in c} & done
    if stale:
        _rewrite_without(out_path, stale)
    it = todo
    if progress:
        from tqdm import tqdm

        it = tqdm(todo, desc=f"yes/no {checkpoint_id}")
    ids = yes_ids + no_ids
    written = 0
    for c in it:
        cache: dict[int, object] = {}  # an image's questions are adjacent: load each image once per chunk
        images = []
        for q in c:
            iid = int(q["image_id"])
            if iid not in cache:
                cache[iid] = load_image(iid)
            images.append(cache[iid])
        lp = vlm.first_token_logprobs(images, [question_prompt(q) for q in c], ids)
        ly, ln = _lse(lp[:, : len(yes_ids)]), _lse(lp[:, len(yes_ids) :])
        rows = [dict(prov, qid=q["qid"], image_id=int(q["image_id"]), logp_yes=float(a), logp_no=float(b), score=float(a - b),
                     mass=float(np.exp(a) + np.exp(b))) for q, a, b in zip(c, ly, ln)]
        append_jsonl([{k: r[k] for k in ROW_FIELDS} for r in rows], out_path)
        written += len(rows)
    return written


def load_scores(path, questions: Sequence[Mapping], questions_sha256: str) -> tuple[dict[str, float], dict]:
    """``{qid: score}`` of one checkpoint's file and its provenance; it must cover every question once."""
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"{path} is empty")
    prov = {f: rows[0].get(f) for f in PROV_FIELDS}
    out: dict[str, float] = {}
    masses = [float(r["mass"]) for r in rows]
    for r in rows:
        if any(r.get(f) != prov[f] for f in PROV_FIELDS):
            raise ValueError(f"{path} mixes provenance")
        if r["qid"] in out:
            raise ValueError(f"{path}: duplicate qid {r['qid']}")
        out[r["qid"]] = float(r["score"])
    if prov["questions_sha256"] != questions_sha256:
        raise ValueError(f"{path} was scored on question file {prov['questions_sha256']}, not {questions_sha256}")
    want = {q["qid"] for q in questions}
    if set(out) != want:
        raise ValueError(f"{path}: {len(want - set(out))} questions missing, {len(set(out) - want)} unknown; finish scoring first")
    return out, prov | {"mean_mass": float(np.mean(masses)), "min_mass": float(np.min(masses))}


# ----------------------------------------------------------------------------- analysis


_N_EXP_RE = re.compile(r"_n(\d+)_s\d+$")
_SEED_RE = re.compile(r"^(?P<cell>.+)_s(?P<seed>\d+)$")


def n_exp_column(checkpoint_id: str) -> str:
    """The ``group`` column of a run: its ``_n<N>`` (K sweep) or the main 1,000."""
    m = _N_EXP_RE.search(checkpoint_id)
    col = m.group(1) if m else "1000"
    if col not in N_EXP_COLUMNS:
        raise ValueError(f"{checkpoint_id}: no question group column for N_exp={col}")
    return col


def run_cell(checkpoint_id: str) -> tuple[str, int]:
    m = _SEED_RE.match(checkpoint_id)
    if not m:
        raise ValueError(f"{checkpoint_id}: not a <cell>_s<seed> run id")
    return m["cell"], int(m["seed"])


def group_arrays(questions: Sequence[Mapping], column: str, cell: str, scores: Mapping[str, Mapping[str, float]]) -> dict:
    """Per-image arrays for the cluster bootstrap: ``{group: {"n": counts, name: score sums}}``.

    Every image of the group is included (count 0 if it has no question of ``cell``), so all cells of
    one column share the same image order and so the same bootstrap draws."""
    out = {}
    for g in ("seen", "unseen"):
        imgs = sorted({int(q["image_id"]) for q in questions if q["group"][column] == g})
        pos = {i: k for k, i in enumerate(imgs)}
        arr = {"n": np.zeros(len(imgs))} | {name: np.zeros(len(imgs)) for name in scores}
        for q in questions:
            if q["group"][column] != g or q["cell"] != cell:
                continue
            k = pos[int(q["image_id"])]
            arr["n"][k] += 1
            for name, s in scores.items():
                arr[name][k] += s[q["qid"]]
        out[g] = arr
    return out


def _mean(a: Mapping[str, np.ndarray], name: str) -> float:
    n = a["n"].sum()
    return float(a[name].sum() / n) if n > 0 else float("nan")


def delta_stat(run: str, base: str) -> Callable:
    def fn(d):
        s, u = d["seen"], d["unseen"]
        return (_mean(s, run) - _mean(s, base)) - (_mean(u, run) - _mean(u, base))

    return fn


def accuracy(scores: Mapping[str, float], questions: Sequence[Mapping]) -> float:
    ok = [(scores[q["qid"]] > 0) == (q["answer"] == "yes") for q in questions]
    return float(np.mean(ok)) if ok else float("nan")


def margins(questions: Sequence[Mapping], d0: Mapping[str, Mapping[str, float]], column: str, multiplier: float = 2.0) -> dict:
    """δ per cell: ``multiplier`` × SD (ddof=1) across the d=0 seeds of their seen-minus-unseen gap."""
    out = {}
    for cell in CELLS:
        a = group_arrays(questions, column, cell, d0)
        if a["seen"]["n"].sum() == 0 or a["unseen"]["n"].sum() == 0:
            out[cell] = {"delta": None, "gaps": None}
            continue
        gaps = [_mean(a["seen"], s) - _mean(a["unseen"], s) for s in d0]
        out[cell] = {"delta": sd_margin(gaps, multiplier), "gaps": dict(zip(d0, gaps))}
    return out


def cell_effects(questions, run_seeds: Mapping[str, Mapping[str, float]], d0_pooled: Mapping[str, float], column: str, cell: str,
                 n_boot: int, boot_seed: int, ci: float) -> dict | None:
    """Pooled (questions averaged over the run's seeds) Δ with bootstrap CI, and each seed's point Δ."""
    pooled = {qid: float(np.mean([s[qid] for s in run_seeds.values()])) for qid in d0_pooled}
    scores = {"pooled": pooled, "d0": d0_pooled} | {f"seed:{k}": v for k, v in run_seeds.items()}
    a = group_arrays(questions, column, cell, scores)
    n_seen, n_unseen = int(a["seen"]["n"].sum()), int(a["unseen"]["n"].sum())
    if n_seen == 0 or n_unseen == 0:
        return None
    boot = paired_bootstrap(delta_stat("pooled", "d0"), a, n_boot=n_boot, seed=boot_seed, ci=ci)
    per_seed = {k: delta_stat(f"seed:{k}", "d0")(a) for k in run_seeds}
    return {"n_questions": {"seen": n_seen, "unseen": n_unseen},
            "n_images": {"seen": int((a["seen"]["n"] > 0).sum()), "unseen": int((a["unseen"]["n"] > 0).sum())},
            "pooled": boot.to_dict(), "per_seed": per_seed}


def directional_pass(eff: Mapping | None, delta: float | None, direction: int) -> bool:
    """The G1(b)-style rule in ``direction`` (+1 / -1): the pooled CI is beyond ±δ and so is every seed."""
    if eff is None or delta is None:
        return False
    p = eff["pooled"]
    if direction > 0:
        return p["ci_lo"] > delta and all(v > delta for v in eff["per_seed"].values())
    return p["ci_hi"] < -delta and all(v < -delta for v in eff["per_seed"].values())


def primary_verdict(effects: Mapping[str, dict | None], margin: Mapping[str, Mapping]) -> dict:
    """The preregistered Phase 0 decision on the primary run's cell effects (see the prereg amendment).

    * P2 (caption closure): yes/none Δ beyond -δ; familiarity: beyond +δ; otherwise neither.
    * P2 is called mention-specific only if yes/direct moved the other way (pooled CI lower bound > 0);
      otherwise it is reported as a seen-image shift toward "no".
    * P1 (control): yes/direct Δ beyond +δ. If it fails, exposure left no detectable trace in the
      yes/no answers and a "neither" on yes/none is not read as evidence against P2.
    """
    none, direct = effects.get("yes/none"), effects.get("yes/direct")
    d_none, d_direct = margin["yes/none"]["delta"], margin["yes/direct"]["delta"]
    p1 = directional_pass(direct, d_direct, +1)
    if directional_pass(none, d_none, -1):
        specific = direct is not None and direct["pooled"]["ci_lo"] > 0
        v = "P2 (caption closure, mention-specific)" if specific else "P2 direction, not mention-specific (seen-image shift toward no)"
    elif directional_pass(none, d_none, +1):
        v = "familiarity"
    else:
        v = "neither"
    return {"verdict": v, "p1_control_pass": p1, "interpretable": bool(p1 or v != "neither"),
            "delta_yes_none": d_none, "delta_yes_direct": d_direct}
