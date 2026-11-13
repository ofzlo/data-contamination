"""Phase 0 yes/no object questions on NoCaps val, from Open Images human-verified image labels.

NoCaps val images are Open Images validation images (``annotations.json`` carries ``open_images_id``),
so each image has human-verified labels: Confidence 1 = verified present, 0 = verified absent. We turn
them into ``Is there a {object} in the image?`` questions:

* **yes** candidates: verified-present labels whose class is one of the 600 boxable (object) classes;
  the full 20k vocabulary is mostly scene / attribute labels (``Nature``, ``Fauna``) that make poor
  object questions.
* **no** candidates: verified-absent labels of the same 600 boxable classes (the full vocabulary gave
  fine-grained negatives such as ``Diatonic button accordion`` on an accordion photo). These are hard
  negatives: the verification task checked labels proposed by a classifier, so a negative was plausible
  for the image (POPE-adversarial-like). Dropped when (a) the same class is also verified present, (b)
  the class is an ancestor of a verified-present class in the boxable hierarchy (``Mammal`` absent while
  ``Dog`` present: contradictory labels), or (c) any of the image's 10 references mentions it (label noise).

**Ambiguous candidates** are written with ``excluded`` set and never sampled (manual review, 2026-11-12):

* ``depiction``: a yes label with any box marked ``IsDepiction`` (statue, toy, drawing: "a lion" for a
  lion-head door knocker);
* ``person_subtype``: ``Man`` / ``Woman`` / ``Boy`` / ``Girl`` either way (an age / gender call, e.g.
  ``Girl`` on a woman); whether a person is there is still asked through ``Person``;
* ``human_present``: a no for ``Mammal`` / ``Animal`` on an image with a person (the hierarchy does not
  put ``Person`` under ``Mammal``, and annotators split on it).

Each image keeps exactly ``k_pos`` yes questions and between ``min_neg`` and ``k_neg`` no questions
(all of its no candidates up to ``k_neg``), sampled with a per-image RNG (``seed``, ``image_id``); images
with fewer than ``k_pos`` yes or ``min_neg`` no candidates are left out. Every candidate is written,
with ``in_sample`` marking the sampled ones, so a different sample or the full candidate set can be
scored without rebuilding.

**Main analysis** (``main_analysis``): sampled questions, without B1-overlap images and without parent
classes of the boxable hierarchy (``Animal``, ``Mammal``, ``Clothing``, ...) except ``Person``. Body-part
classes (``Human eye``, ...) are kept and flagged ``is_body_part``.

**Seen / unseen is NOT defined here.** ``split_half`` only records the split's half (2,250 / 2,250);
the seen group of a checkpoint is its actual exposure set (``contam.eval.evaluate.load_seen_definition``
with the run's N_exp: 1,000, or 30 / 100 / 300 for the K sweep), and ``unseen`` is the split's unseen half.

**Mention labels** use the split's training reference ``refs[train_ref_idx[image_id]]``, for seen and
unseen images alike (every image has an assigned training reference; only seen ones are exposed):

* ``direct``: the class name (last word possibly plural) or a listed synonym appears as a noun
  (``contam.eval.style_transfer.ClassMatcher`` over spaCy UPOS tags);
* ``descendant``: a descendant class in the boxable hierarchy is mentioned (``bird`` -> ``Animal``);
* ``none``.

``n_refs_mentioning`` counts the 10 references with a direct or descendant mention; it is a salience
covariate (mentioned objects are usually large and obvious), e.g. to compare seen vs unseen images
within the same salience band.

The synonym list (``SYNONYMS``) and the phrase rules are drafts, to be checked by hand (proposal:
100 items per type).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from contam.eval.style_transfer import ClassMatcher, _plural_forms, _TOKEN_RE, normalize_class_name

FORMAT_VERSION = 2
QUESTION_TEMPLATE = "Is there {np} in the image?"
PLURAL_TEMPLATE = "Are there {np} in the image?"

# Extra surface forms counted as a direct mention of a class (display name -> forms). Draft: to be
# reviewed by hand. Hierarchy handles parents (``man`` already mentions ``Person`` via ``Man``).
SYNONYMS: dict[str, tuple[str, ...]] = {
    "Person": ("people", "child", "children", "kid", "toddler", "baby", "player", "rider", "crowd", "skier", "surfer"),
    "Man": ("guy", "gentleman"),
    "Woman": ("lady",),
    "Car": ("automobile",),
    "Bicycle": ("bike",),
    "Motorcycle": ("motorbike",),
    "Airplane": ("plane", "jet", "airliner"),
    "Mobile phone": ("phone", "cellphone", "cell phone", "smartphone"),
    "Television": ("tv",),
    "Footwear": ("shoe", "sneaker"),
    "Sunglasses": ("shades",),
    "Bus": ("coach",),
    "Building": ("house", "skyscraper", "tower"),
    "Tree": ("palm",),
}

# Phrase forms (normalized names, i.e. lowercased, without "Human " and parentheses). Draft: every class
# in these lists is also put on the review list. Mass nouns -> "Is there any X ...", plurals -> "Are there any X ...".
MASS_NOUNS = frozenset({
    "clothing", "footwear", "food", "furniture", "swimwear", "kitchenware", "tableware", "cosmetics", "bread", "pasta",
    "fast food", "sushi", "coffee", "juice", "milk", "tea", "wine", "beer", "popcorn", "salad", "dessert", "cheese", "hair",
    "cream", "guacamole", "seafood", "shellfish", "cabinetry", "office supplies", "personal care", "medical equipment",
    "hiking equipment", "bowling equipment", "sports equipment", "toilet paper", "adhesive tape", "face powder",
    "hair spray", "cooking spray", "candy", "perfume", "lavender", "asparagus", "broccoli", "luggage and bags",
})
PLURAL_NOUNS = frozenset({
    "stairs", "shorts", "glasses", "sunglasses", "goggles", "scissors", "trousers", "headphones", "binoculars", "jeans",
    "chopsticks", "earrings", "roller skates", "high heels", "french fries", "salt and pepper shakers",
    "power plugs and sockets", "eyeglasses", "maracas", "moths and butterflies", "rays and skates", "marine invertebrates", "baked goods",
})
# Display name -> phrase head, where the plain normalization reads badly.
PHRASE_OVERRIDES = {
    "Human body": "human body", "Christmas tree": "Christmas tree", "Ipod": "iPod", "Segway": "Segway",
    "Garden Asparagus": "asparagus", "Dairy Product": "dairy product", "Common fig": "fig", "Common sunflower": "sunflower",
    "Kitchen & dining room table": "dining table", "Glasses": "eyeglasses",
}
PERSON_SUBTYPES = frozenset({"Man", "Woman", "Boy", "Girl"})
HUMAN_NAMES = PERSON_SUBTYPES | {"Person"}
NOT_ABSENT_WITH_PERSON = frozenset({"Mammal", "Animal"})


def is_body_part(display_name: str) -> bool:
    return display_name.startswith("Human ")


# ----------------------------------------------------------------------------- loading (untrusted CSV / JSON)


def load_class_names(path) -> dict[str, str]:
    """``{MID: DisplayName}`` from an Open Images class-descriptions CSV (with or without header)."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0].startswith("/m/") and row[1].strip():
                out[row[0]] = row[1].strip()
    if not out:
        raise ValueError(f"no classes in {path}")
    return out


def load_image_labels(path, image_ids: Iterable[str]) -> dict[str, dict[str, dict]]:
    """``{open_images_id: {MID: {"confidence": 0|1, "source": str}}}`` for the given images.

    A MID labelled both ways (it happens across sources) is kept as ``"conflict"``.
    """
    want = set(image_ids)
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            iid = r["ImageID"]
            if iid not in want:
                continue
            c = 1 if float(r["Confidence"]) >= 0.5 else 0
            prev = out[iid].get(r["LabelName"])
            if prev is not None and prev["confidence"] != c:
                prev["confidence"] = "conflict"
            elif prev is None:
                out[iid][r["LabelName"]] = {"confidence": c, "source": r["Source"]}
    return dict(out)


def load_depictions(path, image_ids: Iterable[str]) -> dict[str, dict[str, tuple[int, int]]]:
    """``{open_images_id: {MID: (n_boxes, n_depiction_boxes)}}`` from an Open Images box CSV (``IsDepiction``)."""
    want = set(image_ids)
    out: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["ImageID"] in want:
                c = out[r["ImageID"]][r["LabelName"]]
                c[0] += 1
                c[1] += r.get("IsDepiction") == "1"
    return {i: {m: (n, d) for m, (n, d) in v.items()} for i, v in out.items()}


def load_hierarchy(path) -> dict[str, set[str]]:
    """``{MID: set of ancestor MIDs}`` from ``bbox_labels_600_hierarchy.json`` (``Subcategory`` edges only;
    ``Part`` edges are part-of, not is-a). The synthetic root (``/m/0bl9f``) is not an ancestor."""
    with open(path, encoding="utf-8") as f:
        root = json.load(f)
    anc: dict[str, set[str]] = defaultdict(set)

    def walk(node: Mapping, path: tuple[str, ...]) -> None:
        mid = node["LabelName"]
        if path:
            anc[mid].update(path)
        for child in node.get("Subcategory", []) or []:
            walk(child, path + (mid,) if path or mid != root["LabelName"] else ())

    walk(root, ())
    return dict(anc)


# ----------------------------------------------------------------------------- phrasing


def phrase(display_name: str) -> tuple[str, str, list[str]]:
    """``(noun phrase, question, review reasons)`` for a class display name. Reasons (empty = no flag):
    parenthesis, override, mass_noun, plural_noun, ends_in_s (not listed but may be plural), and_phrase,
    capitalized (a proper noun lowercased), body_part, digits."""
    n = PHRASE_OVERRIDES.get(display_name) or normalize_class_name(display_name)
    reasons = []
    if "(" in display_name:
        reasons.append("parenthesis")
    if display_name in PHRASE_OVERRIDES:
        reasons.append("override")
    if any(w[:1].isupper() for w in display_name.split()[1:]) and display_name not in PHRASE_OVERRIDES:
        reasons.append("capitalized")
    if is_body_part(display_name):
        reasons.append("body_part")
    if re.search(r"\d", display_name):
        reasons.append("digits")
    if " and " in n or " & " in n:
        reasons.append("and_phrase")
    if n in PLURAL_NOUNS:
        reasons.append("plural_noun")
        return f"any {n}", PLURAL_TEMPLATE.format(np=f"any {n}"), reasons
    if n in MASS_NOUNS:
        reasons.append("mass_noun")
        np_ = f"any {n}"
    else:
        if n.endswith("s") and not n.endswith("ss"):
            reasons.append("ends_in_s")
        np_ = ("an " if n[:1].lower() in "aeiou" else "a ") + n
    return np_, QUESTION_TEMPLATE.format(np=np_), reasons


# ----------------------------------------------------------------------------- mentions


class MentionMatcher:
    """Direct / descendant class mentions in spaCy-tagged text (``[(lower token, UPOS), ...]``)."""

    def __init__(self, names: Mapping[str, str], ancestors: Mapping[str, set[str]], synonyms: Mapping[str, Sequence[str]] = SYNONYMS):
        self.mid_of = {v: k for k, v in names.items()}
        self.ancestors = ancestors
        self.matcher = ClassMatcher(sorted(set(names.values())))
        for name, forms in synonyms.items():
            if name not in self.mid_of:
                continue
            for form in forms:
                toks = _TOKEN_RE.findall(form.lower())
                for last in _plural_forms(toks[-1]):
                    self.matcher.variants.setdefault((*toks[:-1], last), name)
                self.matcher.max_len = max(self.matcher.max_len, len(toks))

    def mentioned_mids(self, tagged) -> set[str]:
        return {self.mid_of[n] for n in self.matcher(tagged)}

    def mention_type(self, mid: str, mentioned: set[str]) -> str:
        if mid in mentioned:
            return "direct"
        if any(mid in self.ancestors.get(m, ()) for m in mentioned):
            return "descendant"
        return "none"


# ----------------------------------------------------------------------------- building


def _qid(image_id: int, mid: str) -> str:
    return f"{image_id}:{mid.replace('/m/', '')}"


def build_questions(
    annotations: Sequence[dict],
    split: Mapping,
    labels: Mapping[str, Mapping[str, dict]],
    names: Mapping[str, str],
    boxable: set[str],
    ancestors: Mapping[str, set[str]],
    tagged: Mapping[str, list],
    *,
    k_pos: int = 3,
    k_neg: int = 3,
    min_neg: int = 1,
    seed: int = 20261111,
    b1_overlap: Iterable[int] = (),
    synonyms: Mapping[str, Sequence[str]] = SYNONYMS,
    depictions: Mapping[str, Mapping[str, tuple[int, int]]] | None = None,
) -> tuple[list[dict], dict]:
    """All candidate questions (``in_sample`` marks the sampled ones) and a stats dict.

    ``tagged`` maps every reference string to its spaCy tags (``contam.eval.style_transfer.tag_texts``).
    ``depictions`` is ``load_depictions`` output (``None``: no depiction filter); its Person boxes also
    count as a person being present.
    """
    seen = {int(i) for i in split["seen"]}
    train_idx = {int(k): int(v) for k, v in split["train_ref_idx"].items()}
    b1 = {int(i) for i in b1_overlap}
    mm = MentionMatcher(names, ancestors, synonyms)
    depictions = depictions or {}
    human_mids = {m for m, n in names.items() if n in HUMAN_NAMES or is_body_part(n)}
    person_mid = next((m for m, n in names.items() if n == "Person"), None)
    parents = set().union(*ancestors.values()) if ancestors else set()
    drops: Counter = Counter()
    rows: list[dict] = []
    n_images = Counter()
    for a in sorted(annotations, key=lambda a: int(a["image_id"])):
        iid, oi = int(a["image_id"]), a["open_images_id"]
        lab = labels.get(oi, {})
        ref_mentions = [mm.mentioned_mids(tagged[r]) for r in a["refs"]]
        any_ref = set().union(*ref_mentions)
        train = ref_mentions[train_idx[iid]]
        present = {m for m, v in lab.items() if v["confidence"] == 1}
        drops["conflict"] += sum(v["confidence"] == "conflict" for v in lab.values())
        boxes = depictions.get(oi, {})
        has_person = bool(present & human_mids) or (person_mid in boxes)
        pos, neg, excluded = [], [], {}
        for mid, v in sorted(lab.items()):
            if mid not in names:
                drops["no_display_name"] += 1
            elif v["confidence"] == 1:
                if mid in boxable:
                    if boxes.get(mid, (0, 0))[1]:
                        excluded[mid] = "depiction"
                    elif names[mid] in PERSON_SUBTYPES:
                        excluded[mid] = "person_subtype"
                    pos.append(mid)
                else:
                    drops["pos_not_boxable"] += 1
            elif v["confidence"] == 0:
                if mid not in boxable:
                    drops["neg_not_boxable"] += 1
                elif any(mid in ancestors.get(p, ()) for p in present):
                    drops["neg_ancestor_of_positive"] += 1
                elif any(mm.mention_type(mid, s) != "none" for s in ref_mentions):
                    drops["neg_mentioned_in_refs"] += 1
                else:
                    if names[mid] in PERSON_SUBTYPES:
                        excluded[mid] = "person_subtype"
                    elif names[mid] in NOT_ABSENT_WITH_PERSON and has_person:
                        excluded[mid] = "human_present"
                    neg.append(mid)
        for mid, why in excluded.items():
            drops[f"{'pos' if mid in present else 'neg'}_{why}"] += 1
        pos_ok = [m for m in pos if m not in excluded]
        neg_ok = [m for m in neg if m not in excluded]
        eligible = len(pos_ok) >= k_pos and len(neg_ok) >= min_neg
        n_images["eligible" if eligible else "too_few_labels"] += 1
        sampled: set[str] = set()
        if eligible:
            rng = np.random.default_rng([seed, iid])
            sampled = {pos_ok[i] for i in rng.choice(len(pos_ok), k_pos, replace=False)} | {neg_ok[i] for i in rng.choice(len(neg_ok), min(k_neg, len(neg_ok)), replace=False)}
        for mid in pos + neg:
            np_, question, reasons = phrase(names[mid])
            parent = mid in parents
            main = mid in sampled and iid not in b1 and (not parent or names[mid] == "Person")
            rows.append({
                "qid": _qid(iid, mid),
                "image_id": iid,
                "open_images_id": oi,
                "file_name": a["file_name"],
                "domain": a["domain"],
                "split": split["name"],
                "split_half": "seen_half" if iid in seen else "unseen_half",
                "b1_overlap": iid in b1,
                "label_mid": mid,
                "label_name": names[mid],
                "phrase": np_,
                "question": question,
                "answer": "yes" if mid in present else "no",
                "label_source": lab[mid]["source"],
                "boxable": mid in boxable,
                "is_parent_class": parent,
                "is_body_part": is_body_part(names[mid]),
                "train_ref_idx": train_idx[iid],
                "mention_train": mm.mention_type(mid, train),
                "n_refs_mentioning": sum(mm.mention_type(mid, s) != "none" for s in ref_mentions),
                "phrase_review": reasons,
                "excluded": excluded.get(mid),
                "eligible_image": eligible,
                "in_sample": mid in sampled,
                "main_analysis": main,
            })
    stats = {"images": dict(n_images), "drops": dict(drops), "counts_sampled": count_table(rows, "in_sample"), "counts_main": count_table(rows, "main_analysis")}
    return rows, stats


def cell(r: Mapping) -> str:
    """Analysis cell: ``yes/direct``, ``yes/descendant``, ``yes/none`` or ``no``."""
    return f"yes/{r['mention_train']}" if r["answer"] == "yes" else "no"


def count_table(rows: Iterable[dict], flag: str = "in_sample", groups: Mapping[str, set[int]] | None = None) -> dict:
    """Question counts by cell among rows with ``flag``; with ``groups`` (``{name: image ids}``, e.g. the
    seen / unseen groups of a checkpoint), by ``group/cell`` and with image counts."""
    rows = [r for r in rows if r[flag]]
    if groups is None:
        c = Counter(cell(r) for r in rows)
        return {"images": len({r["image_id"] for r in rows}), "questions": len(rows), **dict(sorted(c.items()))}
    out = {}
    for g, ids in groups.items():
        sub = [r for r in rows if r["image_id"] in ids]
        out[g] = {"images": len({r["image_id"] for r in sub}), "questions": len(sub), **dict(sorted(Counter(cell(r) for r in sub).items()))}
    return out


# ----------------------------------------------------------------------------- review files


def phrase_review_rows(rows: Iterable[dict]) -> list[dict]:
    """One row per class with a review reason: phrase, question, reasons and how often it is asked."""
    by: dict[str, dict] = {}
    for r in rows:
        if not r["phrase_review"]:
            continue
        d = by.setdefault(r["label_mid"], {"label_mid": r["label_mid"], "label_name": r["label_name"], "phrase": r["phrase"], "question": r["question"],
                                           "reasons": ";".join(r["phrase_review"]), "n_candidates": 0, "n_sampled": 0, "n_main": 0,
                                           "ok": "", "suggested_phrase": "", "note": ""})
        d["n_candidates"] += 1
        d["n_sampled"] += r["in_sample"]
        d["n_main"] += r["main_analysis"]
    return sorted(by.values(), key=lambda d: (-d["n_main"], d["label_name"]))


def mention_check_rows(rows: Iterable[dict], annotations: Sequence[dict], n_per_cell: int = 100, seed: int = 20261111) -> list[dict]:
    """``n_per_cell`` random main-analysis questions per cell, with the training caption and blank
    columns for the manual check (label correct, mention label correct, phrase ok). No seen / unseen
    column, so the check is blind to exposure."""
    refs = {int(a["image_id"]): a["refs"] for a in annotations}
    rng = np.random.default_rng(seed)
    out = []
    for c in ("yes/direct", "yes/descendant", "yes/none", "no"):
        pool = sorted((r for r in rows if r["main_analysis"] and cell(r) == c), key=lambda r: r["qid"])
        for i in sorted(rng.choice(len(pool), min(n_per_cell, len(pool)), replace=False)):
            r = pool[i]
            out.append({"cell": c, "qid": r["qid"], "file_name": r["file_name"], "question": r["question"], "label_name": r["label_name"],
                        "answer": r["answer"], "mention_train": r["mention_train"], "train_caption": refs[r["image_id"]][r["train_ref_idx"]],
                        "label_correct": "", "mention_correct": "", "phrase_ok": "", "note": ""})
    return out


SHARE_FIELDS = ("qid", "image_id", "file_name", "open_images_id", "domain", "label_name", "question", "answer",
                "mention_train", "n_refs_mentioning", "is_body_part")


def share_table(rows: Iterable[dict], seen_by_n_exp: Mapping[int, set[int]], unseen: set[int]) -> list[dict]:
    """Main-analysis questions for the team file: the fields above, ``cell``, and per N_exp the image's group
    for checkpoints exposed to that many images: ``seen`` (in the exposure set), ``unseen`` (the split's
    unseen half) or ``unexposed`` (seen half but not exposed: leave out of seen / unseen comparisons)."""
    out = []
    for r in rows:
        if not r["main_analysis"]:
            continue
        iid = r["image_id"]
        d = {k: r[k] for k in SHARE_FIELDS}
        d["cell"] = cell(r)
        d["group"] = {str(n): "seen" if iid in s else "unseen" if iid in unseen else "unexposed" for n, s in sorted(seen_by_n_exp.items(), reverse=True)}
        out.append(d)
    return out
