"""Format-matched style and vocabulary-transfer metrics for short-prompt generations (exploratory).

EXPLORATORY (docs/deviations.md, 2026-11-02): not a gate input. Stands in for the B4 style
classifier, which does not exist yet (docs/task-allocation.md, W2–W4).

Two families, both on unseen images, both compared with the d=0 seeds by an image bootstrap:

* **Stylistic distribution distance.** Distributions of (i) the first two words of a caption,
  (ii) function-word types (``contam.eval.style.FUNCTION_WORDS``, among function-word tokens) and
  (iii) coarse POS bigrams (spaCy UPOS, with sentence-boundary markers) of the model outputs, vs. a
  fixed reference distribution (NoCaps references of other images; COCO captions). Distance:
  Jensen–Shannon divergence, base 2 (0 = identical, 1 = disjoint).
* **Vocabulary transfer.** Open Images class names (the 600 boxable classes NoCaps is built on)
  mentioned as nouns in an output, and whether the same class is named in that image's references.

Every corpus (one training seed of one checkpoint) is a set of per-image rows, all aligned on the
same image order. ``corpus_replicates`` evaluates a corpus statistic under one shared matrix of
bootstrap image weights (row 0 = the original sample), so every comparison built from the same
weights is paired (``compare_replicates``). Levels average the per-seed corpus statistics (plug-in
JS is biased upward in small samples; averaging per-seed values keeps the bias the same for a
1-seed checkpoint and the 5-seed d=0 reference). CIs are bias-shifted percentile intervals, as in
``contam.eval.diversity``.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse

from contam.eval import stats
from contam.eval.style import FUNCTION_WORDS, words

NOUN_TAGS = frozenset({"NOUN", "PROPN"})
DOMAINS = ("in", "near", "out")
SLICES = ("all",) + DOMAINS
CLASS_SETS = ("oi_all", "oi_noncoco", "oi_rare", "oi_noncoco_old", "oi_rare_old")  # *_old: the first (wider) non-COCO set, kept as a sensitivity
VOCAB_STATS = ("mentions", "matched", "precision", "recall")
FEATURES = ("first2", "function_words", "pos_bigrams")

# The 80 COCO detection categories (instances_*2017 names).
COCO_CATEGORIES = frozenset(c.strip() for c in """
    person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|parking meter|bench|bird|cat|dog|horse|sheep|cow|
    elephant|bear|zebra|giraffe|backpack|umbrella|handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|surfboard|
    tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|
    dining table|toilet|tv|laptop|mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|teddy bear|hair drier|toothbrush
    """.replace("\n", "").split("|"))
# Open Images class (normalized name) -> COCO category, where the names differ. Approximate: an
# Open Images class counts as COCO-overlapping if its name is a COCO category or listed here.
OI_COCO_ALIASES = {
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "cattle": "cow", "bull": "cow", "flying disc": "frisbee", "ski": "skis",
    "ball": "sports ball", "football": "sports ball", "tennis ball": "sports ball", "golf ball": "sports ball", "rugby ball": "sports ball",
    "volleyball": "sports ball", "cricket ball": "sports ball",
    "coffee cup": "cup", "mug": "cup", "houseplant": "potted plant", "table": "dining table", "kitchen & dining room table": "dining table",
    "television": "tv", "computer mouse": "mouse", "remote control": "remote", "computer keyboard": "keyboard", "mobile phone": "cell phone",
    "microwave oven": "microwave", "doughnut": "donut", "hair dryer": "hair drier", "sofa bed": "couch", "studio couch": "couch",
}
# Added 2026-11-02 (code review; docs/deviations.md): human body parts (Open Images ``Human <part>``, a person
# in COCO terms) and near-aliases / kinds named by a COCO head noun. Reviewed against all 601 boxable classes.
# Deliberately NOT added (new object vocabulary, not a COCO name): subtypes named by a non-COCO noun (duck,
# owl, goose -> bird; mule -> horse; hamburger is added because COCO labels it sandwich), dog bed, billiard
# table, table tennis racket, telephone / corded phone, tablet computer, jet ski, golf cart; also the broad
# parent classes (luggage and bags, vehicle, land vehicle, animal), which name no specific COCO object.
OI_COCO_NEAR_ALIASES = {
    **{part: "person" for part in ("arm", "beard", "body", "ear", "eye", "face", "foot", "hair", "hand", "head", "leg", "mouth", "nose")},
    "kitchen knife": "knife", "coffee table": "dining table", "computer monitor": "tv", "loveseat": "couch",
    "wall clock": "clock", "alarm clock": "clock", "digital clock": "clock", "infant bed": "bed", "brown bear": "bear", "polar bear": "bear",
    "mixing bowl": "bowl", "submarine sandwich": "sandwich", "hamburger": "sandwich", "flowerpot": "potted plant", "racket": "tennis racket",
    "taxi": "car", "limousine": "car", "van": "car", "ambulance": "truck", "stationary bicycle": "bicycle", "bicycle wheel": "bicycle",
    "aircraft": "airplane", "watercraft": "boat", "canoe": "boat", "gondola": "boat", "barge": "boat",
}
IRREGULAR_PLURALS = {"man": "men", "woman": "women", "person": "people", "mouse": "mice", "goose": "geese", "foot": "feet", "tooth": "teeth", "knife": "knives",
                     "shelf": "shelves", "wolf": "wolves", "leaf": "leaves", "loaf": "loaves", "ox": "oxen"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ----------------------------------------------------------------------------- distributions


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen–Shannon divergence (base 2) between the rows of ``p`` and ``q`` (counts or
    probabilities; normalized here). Broadcasts over leading dimensions; 0·log 0 = 0."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)

    def kl(a):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * np.log2(a / m), 0.0)
        return t.sum(axis=-1)

    return 0.5 * kl(p) + 0.5 * kl(q)


def first2(text: str) -> Counter:
    """The first two words, lowercased (one category per caption; empty caption: none)."""
    w = [x.lower() for x in words(text)[:2]]
    return Counter({" ".join(w): 1}) if w else Counter()


def function_word_counts(text: str) -> Counter:
    return Counter(x for x in (t.lower() for t in words(text)) if x in FUNCTION_WORDS)


def pos_bigrams(tags: Sequence[str]) -> Counter:
    """Coarse POS bigrams with ``<s>`` / ``</s>`` markers (SPACE tokens dropped)."""
    seq = ["<s>", *(t for t in tags if t != "SPACE"), "</s>"]
    return Counter(f"{a}_{b}" for a, b in zip(seq, seq[1:]))


def tag_texts(texts: Iterable[str], nlp=None, batch_size: int = 256) -> dict[str, list[tuple[str, str]]]:
    """``{text: [(lowercased token, UPOS), ...]}`` for the distinct texts (spaCy en_core_web_sm)."""
    if nlp is None:
        import spacy

        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    uniq = list(dict.fromkeys(texts))
    return {t: [(tok.lower_, tok.pos_) for tok in doc] for t, doc in zip(uniq, nlp.pipe(uniq, batch_size=batch_size))}


def feature_counters(text: str, tagged: Sequence[tuple[str, str]]) -> dict[str, Counter]:
    return {"first2": first2(text), "function_words": function_word_counts(text), "pos_bigrams": pos_bigrams([p for _, p in tagged])}


class Vocab:
    """Feature-name -> column index, shared by every corpus and reference of one feature."""

    def __init__(self):
        self.index: dict[str, int] = {}

    def add(self, counters: Iterable[Counter]) -> None:
        for c in counters:
            for k in c:
                self.index.setdefault(k, len(self.index))

    def matrix(self, counters: Sequence[Counter]) -> sparse.csr_matrix:
        rows, cols, vals = [], [], []
        for i, c in enumerate(counters):
            for k, v in c.items():
                rows.append(i)
                cols.append(self.index[k])
                vals.append(v)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(counters), len(self.index)), dtype=float)

    def vector(self, counters: Iterable[Counter]) -> np.ndarray:
        v = np.zeros(len(self.index))
        for c in counters:
            for k, x in c.items():
                v[self.index[k]] += x
        return v


# ----------------------------------------------------------------------------- Open Images vocabulary


def normalize_class_name(name: str) -> str:
    """Lowercase, drop a parenthetical qualifier (``Bat (Animal)`` -> ``bat``) and a leading
    ``Human`` (``Human face`` -> ``face``)."""
    n = re.sub(r"\s*\(.*?\)\s*", " ", name).strip().lower()
    return re.sub(r"^human\s+", "", n)


def load_oi_classes(path) -> list[str]:
    """Display names from an Open Images ``class-descriptions-boxable.csv`` (``MID,DisplayName``;
    untrusted download: plain CSV parse, nothing else)."""
    with open(path, newline="", encoding="utf-8") as f:
        names = [row[1].strip() for row in csv.reader(f) if len(row) >= 2 and row[1].strip()]
    if not names:
        raise ValueError(f"no class names in {path}")
    return names


def coco_overlap(name: str, near_aliases: bool = True) -> bool:
    """COCO-overlapping Open Images class: name is a COCO category or an alias; ``near_aliases`` (default,
    the corrected set) also counts body parts and near-aliases (``OI_COCO_NEAR_ALIASES``); False = the
    first, wider non-COCO set (sensitivity)."""
    n = normalize_class_name(name)
    return n in COCO_CATEGORIES or n in OI_COCO_ALIASES or (near_aliases and n in OI_COCO_NEAR_ALIASES)


def _plural_forms(w: str) -> set[str]:
    out = {w, w + "s", w + "es"}
    if w in IRREGULAR_PLURALS:
        out.add(IRREGULAR_PLURALS[w])
    if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
        out.add(w[:-1] + "ies")
    if w.endswith("fe"):
        out.add(w[:-2] + "ves")
    elif w.endswith("f"):
        out.add(w[:-1] + "ves")
    return out


class ClassMatcher:
    """Finds class names in tagged text: longest match first, non-overlapping, on alphanumeric
    tokens; the last token must be tagged NOUN/PROPN (``orange`` the colour is not ``Orange``);
    the last word may be plural. Returns the set of matched display names."""

    def __init__(self, names: Sequence[str]):
        self.variants: dict[tuple[str, ...], str] = {}
        for name in names:
            toks = _TOKEN_RE.findall(normalize_class_name(name))
            if not toks:
                continue
            for last in _plural_forms(toks[-1]):
                self.variants.setdefault((*toks[:-1], last), name)
        self.max_len = max(len(k) for k in self.variants)

    def __call__(self, tagged: Sequence[tuple[str, str]]) -> set[str]:
        toks = [(t, p) for t, p in tagged if _TOKEN_RE.fullmatch(t)]
        found, i = set(), 0
        while i < len(toks):
            for n in range(min(self.max_len, len(toks) - i), 0, -1):
                key = tuple(t for t, _ in toks[i:i + n])
                if key in self.variants and toks[i + n - 1][1] in NOUN_TAGS:
                    found.add(self.variants[key])
                    i += n
                    break
            else:
                i += 1
        return found


def vocab_rows(mentioned: Sequence[set], ref_named: Sequence[set], class_sets: Mapping[str, set]) -> dict[str, np.ndarray]:
    """Per-image arrays ``<set>/mentioned``, ``<set>/matched`` (also named in the image's
    references) and ``<set>/ref`` (named in the references) for every class set."""
    out = {}
    for s, members in class_sets.items():
        out[f"{s}/mentioned"] = np.array([len(m & members) for m in mentioned], dtype=float)
        out[f"{s}/matched"] = np.array([len(m & r & members) for m, r in zip(mentioned, ref_named)], dtype=float)
        out[f"{s}/ref"] = np.array([len(r & members) for r in ref_named], dtype=float)
    return out


# ----------------------------------------------------------------------------- bootstrap


def boot_weights(n: int, n_boot: int, seed: int) -> np.ndarray:
    """``(n_boot + 1, n)`` image multiplicities; row 0 is the original sample (all ones)."""
    rng = np.random.default_rng(seed)
    w = np.empty((n_boot + 1, n), dtype=np.float32)
    w[0] = 1.0
    for b in range(n_boot):
        w[b + 1] = np.bincount(rng.integers(0, n, n), minlength=n)
    return w


def js_replicates(X: sparse.csr_matrix, refs: Mapping[str, np.ndarray], W: np.ndarray) -> dict[str, np.ndarray]:
    """JS(model distribution under each weight row, reference) for every reference: ``{ref: (B+1,)}``."""
    C = np.asarray((X.T @ W.T).T)
    return {r: js_divergence(C, q[None, :]) for r, q in refs.items()}


def vocab_replicates(rows: Mapping[str, np.ndarray], domain: np.ndarray, W: np.ndarray, class_sets: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    """``{<slice>/<set>/<stat>: (B+1,)}``: mentions and matched per caption, precision =
    Σ matched / Σ mentioned, recall = Σ matched / Σ named in the references. ``class_sets``: default
    every set in ``rows``."""
    if class_sets is None:
        class_sets = list(dict.fromkeys(k.rsplit("/", 1)[0] for k in rows))
    out = {}
    for sl in SLICES:
        mask = np.ones(len(domain), bool) if sl == "all" else (domain == sl)
        Wm = W[:, mask].astype(float)
        n = Wm.sum(axis=1)
        for s in class_sets:
            ment, match, ref = (Wm @ rows[f"{s}/{k}"][mask] for k in ("mentioned", "matched", "ref"))
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"{sl}/{s}/mentions"] = ment / n
                out[f"{sl}/{s}/matched"] = match / n
                out[f"{sl}/{s}/precision"] = np.where(ment > 0, match / ment, np.nan)
                out[f"{sl}/{s}/recall"] = np.where(ref > 0, match / ref, np.nan)
    return out


def _summ(rep: np.ndarray, ci: float) -> dict:
    point, reps = float(rep[0]), rep[1:]
    reps = reps[np.isfinite(reps)]
    if not np.isfinite(point) or reps.size == 0:
        return {"estimate": None, "ci_lo": None, "ci_hi": None, "boot_bias": None, "ci": ci, "n_boot": int(rep.size - 1), "n_boot_used": int(reps.size)}
    lo, hi = stats.percentile_ci(reps, ci)
    bias = float(reps.mean() - point)
    return {"estimate": point, "ci_lo": float(lo - bias), "ci_hi": float(hi - bias), "boot_bias": bias, "ci": ci, "n_boot": int(rep.size - 1),
            "n_boot_used": int(reps.size)}  # n_boot_used: finite (non-NaN) replicates the CI is computed from


def level(per_seed: Sequence[Mapping[str, np.ndarray]], ci: float = 0.95) -> dict:
    """Level of each statistic: mean over seeds of the per-seed corpus statistic."""
    keys = per_seed[0].keys()
    return {k: _summ(np.mean([s[k] for s in per_seed], axis=0), ci) for k in keys}


def compare_replicates(model: Sequence[Mapping[str, np.ndarray]], base: Sequence[Mapping[str, np.ndarray]], ci: float = 0.95) -> dict:
    """``{stat: {model, base, diff}}``: seed-mean levels and model − base, paired through the shared
    bootstrap weights (every replicate dict must come from the same ``W``)."""
    out = {}
    for k in model[0]:
        m = np.mean([s[k] for s in model], axis=0)
        b = np.mean([s[k] for s in base], axis=0)
        out[k] = {"model": _summ(m, ci), "base": _summ(b, ci), "diff": _summ(m - b, ci)}
    return out
