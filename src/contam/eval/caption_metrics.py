"""Q metrics: per-image CIDEr-D and SPICE via pycocoevalcap (eq. 2).

Single-corpus rule
------------------
CIDEr's IDF weights are computed from the *reference corpus of every image scored in the same
call*. Scoring seen and unseen images in separate calls would give the two groups different IDF
tables, and the seen-minus-unseen comparisons (eq. 3, eq. 4) would then mix model effects with
corpus effects. Therefore ``score_split`` ALWAYS scores seen + unseen images in one call per
reference configuration and only afterwards splits per-image scores by group.

``pycocoevalcap.cider.Cider`` is CIDEr-D (Gaussian length penalty, sigma=6, clipped n-gram
counts, x10), the version used by the COCO / NoCaps servers. We multiply CIDEr and SPICE by 100
(the conventional reporting scale), so a per-image CIDEr of 105.0 means 1.05 in pycocoevalcap.

Reference configurations (identical for seen and unseen images; see ``contam.data.splits``):

* ``heldout``: the 9 references other than ``train_ref_idx`` (primary for A and Q).
* ``train``:   the single training reference.
* ``all10``:   all 10 references (comparable to published NoCaps numbers).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REF_CONFIGS = ("heldout", "train", "all10")
METRICS = ("cider", "spice")
SCALE = 100.0


def refs_for_config(refs_by_id: Mapping[int, Sequence[str]], train_ref_idx: Mapping, config: str) -> dict[int, list[str]]:
    """References per image for a reference configuration. ``train_ref_idx`` keys may be str or int."""
    out = {}
    for iid, refs in refs_by_id.items():
        t = int(train_ref_idx[str(iid)] if str(iid) in train_ref_idx else train_ref_idx[iid])
        if config == "heldout":
            out[iid] = [r for j, r in enumerate(refs) if j != t]
        elif config == "train":
            out[iid] = [refs[t]]
        elif config == "all10":
            out[iid] = list(refs)
        else:
            raise ValueError(f"unknown reference config {config!r}; expected one of {REF_CONFIGS}")
    return out


# ----------------------------------------------------------------------------- tokenization

_SIMPLE_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def simple_tokenize(text: str) -> str:
    """Java-free tokenizer (lowercase, drop punctuation). For tests only; real runs use PTB."""
    return " ".join(_SIMPLE_TOKEN.findall(text.lower()))


def normalize_whitespace(text: str) -> str:
    """Collapse every Unicode whitespace run (incl. \\r, \\x0b, \\x0c, \\x85, U+2028, U+2029) to one space.

    The Java PTBTokenizer with ``-preserveLines`` treats several of these as line breaks; one stray
    separator would shift every later caption (hypotheses AND references) in the single call.
    """
    return " ".join(text.split())


def _ptb_lines(sentences: list[str]) -> list[str]:
    """Run the Stanford PTBTokenizer (as pycocoevalcap does) with a hard 1-line-in / 1-line-out check."""
    import subprocess
    import tempfile

    from pycocoevalcap.tokenizer import ptbtokenizer as ptb

    jar_dir = os.path.dirname(os.path.abspath(ptb.__file__))
    for s in sentences:
        if "\n" in s or not s:
            raise ValueError("PTB input lines must be non-empty single lines")
    with tempfile.NamedTemporaryFile("w", delete=False, dir=jar_dir, suffix=".ptb", encoding="utf-8") as f:
        f.write("\n".join(sentences))
        tmp = f.name
    try:
        cmd = ["java", "-cp", ptb.STANFORD_CORENLP_3_4_1_JAR, "edu.stanford.nlp.process.PTBTokenizer", "-preserveLines", "-lowerCase", os.path.basename(tmp)]
        out = subprocess.run(cmd, cwd=jar_dir, stdout=subprocess.PIPE, check=True).stdout.decode("utf-8")
    finally:
        os.remove(tmp)
    if out.endswith("\n"):
        out = out[:-1]
    lines = out.split("\n")
    if len(lines) != len(sentences):
        raise RuntimeError(f"PTB tokenizer returned {len(lines)} lines for {len(sentences)} input captions; captions would be misaligned")
    return [" ".join(w for w in line.rstrip().split(" ") if w not in ptb.PUNCTUATIONS) for line in lines]


def tokenize(captions: Mapping[str, Sequence[str]], tokenizer: str = "ptb") -> dict[str, list[str]]:
    """Tokenize ``{key: [caption, ...]}`` -> ``{key: [tokenized, ...]}`` in ONE tokenizer call.

    Every caption is whitespace-normalized first (``normalize_whitespace``).
    """
    if tokenizer == "simple":
        return {k: [simple_tokenize(normalize_whitespace(c)) for c in v] for k, v in captions.items()}
    if tokenizer != "ptb":
        raise ValueError(f"unknown tokenizer {tokenizer!r}")
    keys = [k for k, v in captions.items() for _ in v]
    # Empty captions would be empty lines; "." tokenizes to "" after punctuation removal.
    sents = [normalize_whitespace(c) or "." for v in captions.values() for c in v]
    out: dict[str, list[str]] = {k: [] for k in captions}
    for k, line in zip(keys, _ptb_lines(sents), strict=True):
        out[k].append(line)
    return out


# ----------------------------------------------------------------------------- SPICE cache


class SpiceCache:
    """On-disk cache of per-image SPICE F-scores keyed by (tokenized hypothesis, tokenized refs).

    SPICE is a per-image score (no corpus statistics), so caching by content is exact. ``save``
    takes an exclusive ``fcntl`` lock on ``<path>.lock``, merges with the current file contents
    (another process may have added entries), writes a per-process temp file and renames it
    atomically, so concurrent evaluations never lose or corrupt entries.
    """

    def __init__(self, path: str | os.PathLike | None):
        self.path = Path(path) if path else None
        self.data: dict[str, float] = {}
        if self.path and self.path.exists():
            with self._lock(fcntl_mode="shared"):
                self.data = self._read()

    def _read(self) -> dict[str, float]:
        if not self.path.exists():
            return {}
        with open(self.path) as f:
            return json.load(f)

    @contextlib.contextmanager
    def _lock(self, fcntl_mode: str = "exclusive"):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.path) + ".lock", "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX if fcntl_mode == "exclusive" else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    @staticmethod
    def key(hyp: str, refs: Sequence[str]) -> str:
        blob = hyp + "\x1f" + "\x1e".join(refs)
        return hashlib.sha1(blob.encode()).hexdigest()

    def get(self, hyp: str, refs: Sequence[str]) -> float | None:
        return self.data.get(self.key(hyp, refs))

    def put(self, hyp: str, refs: Sequence[str], value: float) -> None:
        self.data[self.key(hyp, refs)] = value

    def save(self) -> None:
        if not self.path:
            return
        import tempfile

        with self._lock():
            merged = self._read()
            merged.update(self.data)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(merged, f)
            os.replace(tmp, self.path)
            self.data = merged


def _spice_scores(gts_tok: dict[str, list[str]], res_tok: dict[str, list[str]], cache: SpiceCache | None) -> dict[str, float]:
    out: dict[str, float] = {}
    todo = []
    for k in gts_tok:
        hyp = res_tok[k][0]
        if not hyp.strip():
            out[k] = 0.0
            continue
        cached = cache.get(hyp, gts_tok[k]) if cache else None
        if cached is not None:
            out[k] = cached
        else:
            todo.append(k)
    if todo:
        from pycocoevalcap.spice.spice import Spice

        # pycocoevalcap's Spice returns per-image scores in sorted(key) order.
        keys = sorted(todo)
        _, scores = Spice().compute_score({k: gts_tok[k] for k in keys}, {k: res_tok[k] for k in keys})
        n_nan = 0
        for k, s in zip(keys, scores):
            f = float(s["All"]["f"])
            if math.isnan(f):
                f, n_nan = 0.0, n_nan + 1
            out[k] = f
            if cache:
                cache.put(res_tok[k][0], gts_tok[k], f)
        if n_nan:
            warnings.warn(f"SPICE returned NaN for {n_nan} images; set to 0")
        if cache:
            cache.save()
    return out


# ----------------------------------------------------------------------------- scoring


def score_corpus(
    hyps: Mapping[int, str],
    refs: Mapping[int, Sequence[str]],
    metrics: Iterable[str] = METRICS,
    tokenizer: str = "ptb",
    spice_cache: SpiceCache | str | None = None,
) -> dict[str, dict[int, float]]:
    """Score ``hyps`` against ``refs`` as ONE corpus. Returns ``{metric: {image_id: score x100}}``.

    Every image passed here contributes its references to the CIDEr IDF table.
    """
    ids = list(hyps)
    if set(ids) != set(refs):
        raise ValueError("hyps and refs must have the same image ids")
    if not ids:
        return {m: {} for m in metrics}
    # Zero-padded keys make sorted order == insertion order for every scorer.
    keys = {iid: f"{n:07d}" for n, iid in enumerate(ids)}
    to_tok = {"h" + keys[i]: [hyps[i]] for i in ids}
    to_tok.update({"r" + keys[i]: list(refs[i]) for i in ids})
    tok = tokenize(to_tok, tokenizer)
    res_tok = {keys[i]: tok["h" + keys[i]] for i in ids}
    gts_tok = {keys[i]: tok["r" + keys[i]] for i in ids}

    out: dict[str, dict[int, float]] = {}
    for m in metrics:
        if m == "cider":
            from pycocoevalcap.cider.cider import Cider

            _, scores = Cider().compute_score(gts_tok, res_tok)
            by_key = dict(zip(gts_tok.keys(), (float(s) for s in scores)))
        elif m == "spice":
            cache = spice_cache if isinstance(spice_cache, SpiceCache) or spice_cache is None else SpiceCache(spice_cache)
            by_key = _spice_scores(gts_tok, res_tok, cache)
        else:
            raise ValueError(f"unknown metric {m!r}")
        out[m] = {i: SCALE * by_key[keys[i]] for i in ids}
    return out


def score_split(
    hyps: Mapping[int, str],
    refs_by_id: Mapping[int, Sequence[str]],
    split: dict,
    ref_configs: Iterable[str] = REF_CONFIGS,
    metrics: Iterable[str] = METRICS,
    tokenizer: str = "ptb",
    spice_cache: SpiceCache | str | None = None,
    require_all: bool = True,
) -> dict[str, dict[str, dict[int, float]]]:
    """Per-image scores for every reference configuration, seen + unseen scored jointly.

    Returns ``{ref_config: {metric: {image_id: score}}}`` over all images in ``hyps``; use
    ``by_group`` to split by seen / unseen afterwards.

    Args:
        require_all: if True (default, required for gate runs), every image of the split must have a
            hypothesis, so the IDF corpus is always the full seen + unseen set.
    """
    hyps = {int(k): v for k, v in hyps.items()}
    split_ids = set(split["seen"]) | set(split["unseen"])
    missing = split_ids - set(hyps)
    if require_all and missing:
        raise ValueError(f"{len(missing)} split images have no hypothesis; CIDEr must be scored on the full corpus")
    extra = set(hyps) - split_ids
    if extra:
        raise ValueError(f"{len(extra)} hypotheses are not in the split")
    if isinstance(spice_cache, (str, os.PathLike)):
        spice_cache = SpiceCache(spice_cache)
    sub_refs = {i: refs_by_id[i] for i in hyps}
    out = {}
    for cfg in ref_configs:
        refs = refs_for_config(sub_refs, split["train_ref_idx"], cfg)
        out[cfg] = score_corpus(hyps, refs, metrics, tokenizer, spice_cache)
    return out


def by_group(scores: Mapping[int, float], split: dict, groups: Sequence[str] = ("seen", "unseen")) -> dict[str, dict[int, float]]:
    """Split a per-image score dict by split group (after joint scoring)."""
    out = {}
    for g in groups:
        ids = split[g]
        out[g] = {i: scores[i] for i in ids if i in scores}
    return out
