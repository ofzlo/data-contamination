"""COCO image study (Phase 1) evaluation: VQAv2 accuracy, POPE, and the cross-half Leak metric.

Questions come from ``outputs/coco/coco_questions.json`` (``scripts/build_coco_questions.py``). Every
question carries its image's half (A / B) and a cell:
* VQAv2 cells: ``mentioned`` / ``unmentioned`` (``other`` answers), ``yes/no`` and ``number``;
* POPE cells: ``pos/mentioned``, ``pos/unmentioned``, ``neg/random``, ``neg/popular`` and
  ``neg/adversarial``.

Scoring (one checkpoint)
------------------------
* **VQAv2** (``score_vqa``): a greedy short answer to ``<question>`` + ``VQA_SUFFIX`` (LLaVA-1.5's
  VQAv2 prompt), ``max_new_tokens`` 10. Accuracy follows the official VQA evaluation (``vqa_accuracy``):
  * answers are normalized (punctuation, number words, articles, contractions);
  * each of the 10 annotators is left out in turn, and the score is the mean over those 10 of
    min(#matching other answers / 3, 1).
  * Deviation: the prediction is always normalized, as LLaVA's EvalAI answer processor does before
    submission.
  * Batches are fixed and sorted by prompt token length (``vqa_order``: batched generation with left
    padding garbled some answers). Runs resume, and provenance is checked, as in ``contam.eval.yesno``.
* **POPE**: the Phase 0 first-token yes/no scorer (``contam.eval.yesno.score_questions``), unchanged.
  A question counts as answered yes when log P(yes) − log P(no) > 0.

Leak (``leak``)
---------------
M_A is trained on half A and M_B on half B (one level, e.g. L3). Per cell,

    Leak = ½ [(acc(M_A, A) − acc(M_B, A)) + (acc(M_B, B) − acc(M_A, B))],

where acc is the mean over the cell's questions on that half's images. Both models are scored on the
same questions, so whatever the two models share cancels, and only "having seen the half" remains.
The CI comes from an image-level bootstrap within each half: halves A and B are resampled
independently, each image with all its questions, on the same draws for both models.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from contam.config import append_jsonl, prompt_sha, read_jsonl
from contam.eval.stats import paired_bootstrap

VQA_SUFFIX = "\nAnswer the question using a single word or phrase."
VQA_CELLS = ("mentioned", "unmentioned", "yes/no", "number")
POPE_CELLS = ("pos/mentioned", "pos/unmentioned", "neg/random", "neg/popular", "neg/adversarial")
NEG_TYPES = ("random", "popular", "adversarial")
VQA_ROW_FIELDS = ("checkpoint_id", "model", "adapter", "prompt_sha", "questions_sha256", "batch_size", "batch_order", "max_new_tokens", "qid", "image_id", "answer", "acc")
VQA_PROV_FIELDS = ("checkpoint_id", "model", "adapter", "prompt_sha", "questions_sha256", "batch_size", "batch_order", "max_new_tokens")

# --------------------------------------------------------------------------- official VQA normalization

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've",
    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
    "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've",
}
_NUMBERS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
            "eight": "8", "nine": "9", "ten": "10"}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCT = [";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]


def _punctuation(text: str) -> str:
    out = text
    for p in _PUNCT:
        if (p + " " in text or " " + p in text) or _COMMA_STRIP.search(text) is not None:
            out = out.replace(p, "")
        else:
            out = out.replace(p, " ")
    return _PERIOD_STRIP.sub("", out)


def _digit_article(text: str) -> str:
    words = [_NUMBERS.get(w, w) for w in text.lower().split()]
    words = [w for w in words if w not in _ARTICLES]
    return " ".join(_CONTRACTIONS.get(w, w) for w in words)


def normalize_answer(text: str) -> str:
    """The official VQA answer normalization (punctuation, number words, articles, contractions)."""
    return _digit_article(_punctuation(text.replace("\n", " ").replace("\t", " ").strip()))


def vqa_accuracy(pred: str, answers: Sequence[str]) -> float:
    """Official VQA accuracy of ``pred`` against the 10 annotator ``answers``: the mean over the 10
    leave-one-out subsets of min(#matching other answers / 3, 1). The prediction is always normalized,
    and the annotator answers are normalized when they disagree (as the official evaluation does)."""
    res = normalize_answer(pred)
    gts = [a.replace("\n", " ").replace("\t", " ").strip() for a in answers]
    if len(set(gts)) > 1:
        gts = [normalize_answer(a) for a in gts]
    accs = []
    for k in range(len(gts)):
        others = gts[:k] + gts[k + 1:]
        accs.append(min(1.0, sum(o == res for o in others) / 3.0))
    return float(np.mean(accs))


# --------------------------------------------------------------------------- scoring


def sort_questions(questions: Sequence[Mapping]) -> list[dict]:
    return sorted((dict(q) for q in questions), key=lambda q: (int(q["image_id"]), q["qid"]))


def vqa_order(questions: Sequence[Mapping], length_fn: Callable[[Mapping], int]) -> list[dict]:
    """VQAv2 batch order: by prompt token length, then image id, then question id.

    Batched greedy generation with left padding changed answers. For example, "Is it daytime?" gave
    "Yes Yes Yes ..." inside a mixed-length batch and "Yes" alone: 6 of 96 answers differed from
    single-question generation. With length-sorted batches the padding is (almost) gone, and 2 of 96
    differ, from bf16 batch numerics. The order is fixed, so every checkpoint sees the same batches."""
    return sorted((dict(q) for q in questions), key=lambda q: (int(length_fn(q)), int(q["image_id"]), q["qid"]))


def _chunks(questions, batch_size):
    return [questions[s : s + batch_size] for s in range(0, len(questions), batch_size)]


def _rewrite_without(path, qids: set[str]) -> None:
    import json
    import os

    rows = [r for r in read_jsonl(path) if r["qid"] not in qids]
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def score_vqa(vlm, load_image: Callable[[int], object], questions: Sequence[Mapping], out_path, checkpoint_id: str, questions_sha256: str,
              batch_size: int = 16, max_new_tokens: int = 10, progress: bool = True, batch_order: str = "length_sorted") -> int:
    """Answer and score the missing VQAv2 questions; append rows to ``out_path``. Returns rows written.
    ``questions`` must already be in the fixed batch order (``vqa_order``; ``batch_order`` names it in
    the provenance), so every checkpoint sees the same batches."""
    prov = dict(checkpoint_id=checkpoint_id, model=getattr(vlm, "model_id", None), adapter=getattr(vlm, "adapter", None),
                prompt_sha=prompt_sha("{question}" + VQA_SUFFIX), questions_sha256=questions_sha256, batch_size=int(batch_size),
                batch_order=batch_order, max_new_tokens=int(max_new_tokens))
    done: set[str] = set()
    if Path(out_path).exists():
        for r in read_jsonl(out_path):
            for f in VQA_PROV_FIELDS:
                if r.get(f) != prov[f]:
                    raise ValueError(f"{out_path} has rows with {f}={r.get(f)!r} (now {prov[f]!r}); write to a new file")
            done.add(r["qid"])
    todo = [c for c in _chunks(list(questions), batch_size) if any(q["qid"] not in done for q in c)]
    stale = {q["qid"] for c in todo for q in c} & done
    if stale:
        _rewrite_without(out_path, stale)
    it = todo
    if progress:
        from tqdm import tqdm

        it = tqdm(todo, desc=f"vqa {checkpoint_id}")
    written = 0
    for c in it:
        cache: dict[int, object] = {}
        images = []
        for q in c:
            iid = int(q["image_id"])
            if iid not in cache:
                cache[iid] = load_image(iid)
            images.append(cache[iid])
        preds = vlm.generate_answers(images, [q["question"] + VQA_SUFFIX for q in c], max_new_tokens)
        rows = [dict(prov, qid=q["qid"], image_id=int(q["image_id"]), answer=p, acc=vqa_accuracy(p, q["answers"])) for q, p in zip(c, preds)]
        append_jsonl([{k: r[k] for k in VQA_ROW_FIELDS} for r in rows], out_path)
        written += len(rows)
    return written


def load_vqa(path, questions: Sequence[Mapping], questions_sha256: str) -> tuple[dict[str, float], dict]:
    """``{qid: accuracy}`` of one checkpoint's VQAv2 file; it must cover every question once."""
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"{path} is empty")
    prov = {f: rows[0].get(f) for f in VQA_PROV_FIELDS}
    out = {}
    for r in rows:
        if any(r.get(f) != prov[f] for f in VQA_PROV_FIELDS):
            raise ValueError(f"{path} mixes provenance")
        if r["qid"] in out:
            raise ValueError(f"{path}: duplicate qid {r['qid']}")
        out[r["qid"]] = float(r["acc"])
    if prov["questions_sha256"] != questions_sha256:
        raise ValueError(f"{path} was scored on question file {prov['questions_sha256']}, not {questions_sha256}")
    want = {q["qid"] for q in questions}
    if set(out) != want:
        raise ValueError(f"{path}: {len(want - set(out))} questions missing, {len(set(out) - want)} unknown; finish scoring first")
    return out, prov


def pope_correct(scores: Mapping[str, float], questions: Sequence[Mapping]) -> dict[str, float]:
    """``{qid: 1.0 if the yes/no answer (score > 0) matches the label else 0.0}``."""
    return {q["qid"]: float((scores[q["qid"]] > 0) == (q["answer"] == "yes")) for q in questions}


def pope_f1(scores: Mapping[str, float], questions: Sequence[Mapping], half: str | None = None) -> dict:
    """POPE accuracy / precision / recall / F1 / yes-rate per negative type: all positives plus that
    type's negatives (optionally one half only)."""
    qs = [q for q in questions if half is None or q["half"] == half]
    pos = [q for q in qs if q["answer"] == "yes"]
    out = {}
    for t in NEG_TYPES:
        sel = pos + [q for q in qs if q.get("neg_type") == t]
        yhat = np.array([scores[q["qid"]] > 0 for q in sel])
        y = np.array([q["answer"] == "yes" for q in sel])
        tp, fp, fn = int(np.sum(yhat & y)), int(np.sum(yhat & ~y)), int(np.sum(~yhat & y))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[t] = {"n": len(sel), "accuracy": float(np.mean(yhat == y)) if len(sel) else float("nan"), "precision": prec, "recall": rec,
                  "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "yes_rate": float(np.mean(yhat)) if len(sel) else float("nan")}
    return out


# --------------------------------------------------------------------------- Leak


def _per_image(correct: Mapping[str, float], questions: Sequence[Mapping]) -> tuple[list[int], np.ndarray, np.ndarray]:
    sums, counts = defaultdict(float), defaultdict(int)
    for q in questions:
        sums[int(q["image_id"])] += correct[q["qid"]]
        counts[int(q["image_id"])] += 1
    ids = sorted(sums)
    return ids, np.array([sums[i] for i in ids]), np.array([counts[i] for i in ids], dtype=float)


def leak(correct_a: Mapping[str, float], correct_b: Mapping[str, float], questions: Sequence[Mapping], cell: str | None,
         n_boot: int = 10_000, boot_seed: int = 0, ci: float = 0.95) -> dict | None:
    """Leak of one level in one cell (``None``: all questions). ``correct_a`` / ``correct_b`` are the
    per-question accuracies of M_A (trained on half A) and M_B (trained on half B)."""
    qs = [q for q in questions if cell is None or q["cell"] == cell]
    data = {}
    for h in ("A", "B"):
        hq = [q for q in qs if q["half"] == h]
        if not hq:
            return None
        ids, sa, n = _per_image(correct_a, hq)
        _, sb, _ = _per_image(correct_b, hq)
        data[h] = {"a": sa, "b": sb, "n": n}

    def stat(d):
        acc = lambda h, m: float(d[h][m].sum() / d[h]["n"].sum())
        return 0.5 * ((acc("A", "a") - acc("A", "b")) + (acc("B", "b") - acc("B", "a")))

    boot = paired_bootstrap(stat, data, n_boot=n_boot, seed=boot_seed, ci=ci)
    acc = {f"{m}_on_{h}": float(data[h][m].sum() / data[h]["n"].sum()) for h in ("A", "B") for m in ("a", "b")}
    return {"estimate": boot.estimate, "ci_lo": boot.lo, "ci_hi": boot.hi, "ci": ci, "n_boot": n_boot,
            "n_questions": {h: int(data[h]["n"].sum()) for h in ("A", "B")}, "n_images": {h: len(data[h]["n"]) for h in ("A", "B")},
            "accuracy": {"M_A_on_A": acc["a_on_A"], "M_B_on_A": acc["b_on_A"], "M_B_on_B": acc["b_on_B"], "M_A_on_B": acc["a_on_B"]}}
