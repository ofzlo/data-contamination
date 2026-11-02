"""Caption diversity metrics (exploratory, docs/deviations.md; never gate inputs).

Per group of captions (e.g. a checkpoint's unseen images):

* ``distinct-n`` (n = 1, 2): corpus-level, number of distinct word n-grams / total word n-grams over
  all captions of the group (words: ``style.words``, lowercased; n-grams do not cross captions).
* duplicate rate: fraction of captions whose normalized text (``memory.normalize``, as for exact
  match) equals that of another caption in the group. Computed per image on the full group
  (``duplicate_flags``), so its group mean is the rate and it bootstraps like any per-image field.
* nouns per caption: per image, tokens tagged NOUN or PROPN (spaCy ``en_core_web_sm``; fallback
  NLTK's averaged perceptron tagger, tags NN*). ``noun_counts`` returns the backend used.

Distinct-n is not a mean over images. ``bootstrap_corpus`` resamples images (multiplicity weights,
one index vector per replicate shared by every corpus, so model − d0 stays paired) and recomputes it.
Resampling with replacement repeats captions and so biases distinct-n downward; the reported CI is
the percentile interval shifted by the estimated bias (replicate mean − point estimate).
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
from scipy import sparse

from contam.eval import memory, stats, style

NOUN_TAGS = ("NOUN", "PROPN")


def tokens(text: str) -> list[str]:
    return [w.lower() for w in style.words(text)]


def ngrams(toks: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(toks[k : k + n]) for k in range(len(toks) - n + 1)]


def ngram_matrix(texts: Sequence[str], n: int) -> sparse.csr_matrix:
    """(captions x distinct n-gram types) count matrix of one corpus."""
    vocab: dict[tuple[str, ...], int] = {}
    rows, cols, vals = [], [], []
    for r, t in enumerate(texts):
        for g, c in Counter(ngrams(tokens(t), n)).items():
            rows.append(r)
            cols.append(vocab.setdefault(g, len(vocab)))
            vals.append(c)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(vocab)), dtype=float)


def distinct_weighted(X: sparse.csr_matrix, w: np.ndarray | None = None) -> float:
    """Distinct-n of the corpus ``X`` with caption k counted ``w[k]`` times (None: once each)."""
    counts = X.sum(axis=0).A1 if w is None else X.T @ np.asarray(w, dtype=float)
    total = counts.sum()
    return float((counts > 0).sum() / total) if total else 0.0


def distinct_n(texts: Sequence[str], n: int) -> float:
    return distinct_weighted(ngram_matrix(texts, n))


def duplicate_flags(texts: Sequence[str]) -> np.ndarray:
    """1.0 where a caption's normalized text equals that of another caption in ``texts``."""
    norm = [memory.normalize(t) for t in texts]
    c = Counter(norm)
    return np.array([float(c[x] > 1) for x in norm])


def noun_counts(texts: Sequence[str], backend: str = "auto") -> tuple[np.ndarray, str]:
    """Nouns (NOUN + PROPN) per caption and the backend used (``spacy:<model>`` or ``nltk:averaged_perceptron``).

    ``auto`` tries spaCy ``en_core_web_sm`` and falls back to NLTK if it is not installed."""
    if backend in ("auto", "spacy"):
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
        except (ImportError, OSError):
            if backend == "spacy":
                raise
        else:
            out = [sum(t.pos_ in NOUN_TAGS for t in doc) for doc in nlp.pipe(texts, batch_size=256)]
            return np.array(out, dtype=float), f"spacy:en_core_web_sm-{nlp.meta.get('version')}"
    import nltk

    out = [sum(tag.startswith("NN") for _, tag in nltk.pos_tag(nltk.word_tokenize(t))) if t.strip() else 0 for t in texts]
    return np.array(out, dtype=float), "nltk:averaged_perceptron"


def bootstrap_corpus(model: Sequence[sparse.csr_matrix], base: Sequence[sparse.csr_matrix], n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict:
    """Distinct-n of a checkpoint (``model``: one corpus per training seed) vs. d=0 (``base``: one per seed).

    All corpora are aligned by image (row k = the same image). A level is the mean over seeds of each
    seed's corpus-level distinct-n; ``diff`` = model − base. Every replicate draws one image index
    vector and applies it to every corpus. CIs: bias-shifted percentile intervals (module docstring)."""
    n = {X.shape[0] for X in [*model, *base]}
    if len(n) != 1:
        raise ValueError(f"corpora are not aligned by image (rows {sorted(n)})")
    n = n.pop()
    if n == 0:
        raise ValueError("empty group")

    def stat(w):
        m = float(np.mean([distinct_weighted(X, w) for X in model]))
        b = float(np.mean([distinct_weighted(X, w) for X in base]))
        return np.array([m, b, m - b])

    point = stat(None)
    rng = np.random.default_rng(seed)
    reps = np.vstack([stat(np.bincount(rng.integers(0, n, n), minlength=n)) for _ in range(n_boot)])
    lo, hi = stats.percentile_ci(reps, ci)
    bias = reps.mean(axis=0) - point
    out = {}
    for k, name in enumerate(("model", "d0", "diff")):
        out[name] = {"estimate": float(point[k]), "ci_lo": float(lo[k] - bias[k]), "ci_hi": float(hi[k] - bias[k]), "boot_bias": float(bias[k]), "ci": ci, "n_boot": n_boot}
    return out
