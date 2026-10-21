"""S (style) metrics per caption, and group aggregates.

Per caption: word count, sentence count, boilerplate flag, function-word ratio and (optional,
needs spaCy) a coarse POS mix. Gate decisions use the two primary S metrics (plan, "Shared metrics"):

* mean word count on unseen images, as a change vs. the (seed-pooled) d=0 model;
* boilerplate rate on unseen images, as a change vs. the d=0 model.

Seen images are logged too. The paired change and its CI are computed in ``contam.eval.stats``;
this module only produces per-image values and plain group means.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

import numpy as np

# Default boilerplate regexes (case-insensitive, matched anywhere in the caption). The list used
# for gate runs lives in configs/default.yaml (style.boilerplate_patterns) and overrides this one.
DEFAULT_BOILERPLATE_PATTERNS: tuple[str, ...] = (
    r"\bthe (image|picture|photo|photograph|scene) (shows|showcases|depicts|features|displays|captures|presents|portrays|contains|is of)\b",
    r"\bin (this|the) (image|picture|photo|photograph|scene)\b",
    r"\bin the (center|centre|middle|foreground|background) of the (image|picture|photo|photograph|scene|frame)\b",
    r"\bthis (image|picture|photo|photograph) (shows|showcases|depicts|features|displays|captures|contains|is)\b",
    r"\bthis is (an?|the) (image|picture|photo|photograph|close-up|view)\b",
    r"^\W*(an?|the) (image|picture|photo|photograph) of\b",  # opener only
    r"^\W*the (image|picture|photo|photograph) is an?\b",  # opener only
    r"\b(we|you) can see\b",
    r"\bit appears (to be|that)\b",
    r"\boverall,",
)

# Closed-class English function words (articles, pronouns, prepositions, conjunctions,
# auxiliaries, determiners, common adverbial particles).
FUNCTION_WORDS = frozenset(
    """
    a an the this that these those some any each every no other another such what which whose
    i me my mine we us our ours you your yours he him his she her hers it its they them their theirs
    myself yourself himself herself itself ourselves themselves one
    of in on at by for with about against between into through during before after above below
    to from up down out off over under onto upon within without along across behind beyond near
    around among toward towards beside besides via per than like
    and or but nor so yet if because although though while whereas since unless until whether as
    be am is are was were been being have has had having do does did doing
    can could may might must shall should will would
    not there here then too very also just only both either neither all most more less few
    who whom whose when where why how
    """.split()
)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
SENT_SPLIT_RE = re.compile(r"[.!?]+(?=\s|$)")

PER_CAPTION_FIELDS = ("word_count", "sentence_count", "boilerplate", "function_word_ratio")


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def sentence_count(text: str) -> int:
    """Number of non-empty segments after splitting on sentence-final punctuation."""
    return sum(1 for s in SENT_SPLIT_RE.split(text) if words(s))


def compile_patterns(patterns: Iterable[str] | None = None) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in (DEFAULT_BOILERPLATE_PATTERNS if patterns is None else patterns)]


def is_boilerplate(text: str, compiled: Sequence[re.Pattern]) -> bool:
    return any(p.search(text) for p in compiled)


def caption_style(text: str, compiled: Sequence[re.Pattern] | None = None) -> dict[str, float]:
    """Style features of one caption."""
    compiled = compile_patterns() if compiled is None else compiled
    w = words(text)
    n = len(w)
    return {
        "word_count": float(n),
        "sentence_count": float(sentence_count(text)),
        "boilerplate": float(is_boilerplate(text, compiled)),
        "function_word_ratio": (sum(x.lower() in FUNCTION_WORDS for x in w) / n) if n else 0.0,
    }


POS_TAGS = ("NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "ADP", "DET", "PRON", "CCONJ", "NUM")


def pos_mix(texts: Sequence[str], model: str = "en_core_web_sm") -> list[dict[str, float]]:
    """Fraction of tokens per coarse POS tag. Optional: requires spaCy and the model to be installed."""
    try:
        import spacy
    except ImportError as e:  # optional dependency
        raise ImportError("pos_mix needs spaCy: pip install spacy && python -m spacy download en_core_web_sm") from e
    nlp = spacy.load(model, disable=["ner", "parser", "lemmatizer"])
    out = []
    for doc in nlp.pipe(texts):
        toks = [t for t in doc if not t.is_punct and not t.is_space]
        n = len(toks) or 1
        out.append({f"pos_{tag.lower()}": sum(t.pos_ == tag for t in toks) / n for tag in POS_TAGS})
    return out


def style_per_image(captions: Mapping[int, str], patterns: Iterable[str] | None = None, with_pos: bool = False) -> dict[int, dict[str, float]]:
    """``{image_id: {feature: value}}`` for every caption."""
    compiled = compile_patterns(patterns)
    out = {iid: caption_style(c, compiled) for iid, c in captions.items()}
    if with_pos:
        ids = list(captions)
        for iid, mix in zip(ids, pos_mix([captions[i] for i in ids])):
            out[iid].update(mix)
    return out


def group_means(per_image: Mapping[int, Mapping[str, float]], ids: Sequence[int]) -> dict[str, float]:
    """Mean of each feature over the given image ids (boilerplate mean = boilerplate rate)."""
    ids = [i for i in ids if i in per_image]
    if not ids:
        return {}
    keys = per_image[ids[0]].keys()
    return {k: float(np.mean([per_image[i][k] for i in ids])) for k in keys}
