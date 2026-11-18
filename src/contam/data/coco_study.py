"""COCO image study (Phase 1): clean image pool, A/B halves, exposure rows and evaluation questions.

Pool (``build_pool``)
---------------------
COCO val2017 images that have at least one non-crowd object annotation and VQAv2 val questions, minus
every image LLaVA-1.5 saw (``seen_ids``):

* 665K ``coco/`` examples: same COCO id, or same Flickr photo id;
* 665K ``vg/`` and ``gqa/`` examples (GQA images are Visual Genome images): the VG image's ``coco_id``,
  or the same Flickr photo id;
* LCS-558K pretraining images: the same Flickr photo id.

A Visual Genome image that no 665K example uses was never shown to LLaVA, so it stays in the pool.
The pool records how many such images there are (``n_in_vg_unused``), for a conservative robustness
view.

Halves
------
Each image's stratum is the supercategory of its largest-area non-crowd object. As in
``contam.data.splits``:
* every stratum gets one permutation, seeded by ``(split_seed, crc32(stratum))``;
* half A is the union of per-stratum prefixes, and half B is the rest;
* stratum counts come from largest-remainder rounding to exactly round(N / 2), with ties broken in
  stratum-name order.

Every image gets a training caption: index ``train_caption_idx`` into its COCO captions, drawn with
``ref_seed`` over sorted ids.

Exposure levels (``build_exposure``; image paths ``coco_val2017/<file>`` under ``paths.llava_images``)
----------------------------------------------------------------------------------------------------
* ``L3``: image plus every VQAv2 val question of the image as one multi-turn conversation, in the
  665K VQAv2 format (``<question>\\nAnswer the question using a single word or phrase.`` ->
  ``multiple_choice_answer``). This is the positive control: the benchmark answers themselves.
* ``L2``: image + p_cap -> the training caption (informative text about the image).
* ``L1``: image + p_cap -> ``UNINFORMATIVE`` (the image without information-bearing text).
* ``Cap``: no image; ``caption_text`` -> the training caption (text only).

Every exposed row is repeated ``d`` times, and every level exposes all images of one half.

Questions (``vqav2_questions``, ``pope_questions``)
---------------------------------------------------
* VQAv2: every val question of every pool image. Mention labels are computed for ``other``
  answers only: ``mentioned`` if the normalized answer occurs as a token sequence in the training
  caption (words compared in singular form), else ``unmentioned``. ``yes/no`` and ``number``
  answers get those types as their cell.
* POPE-style: up to 3 positives per image (present categories, largest total area first) and 3
  negatives per negative type:
  * ``random``: drawn with a per-image RNG;
  * ``popular``: most frequent categories in the pool;
  * ``adversarial``: categories that co-occur most with the image's present ones.
  Negatives are absent categories that none of the image's 5 captions mentions. Positive mention
  labels come from the COCO-category matcher (spaCy nouns, plural forms, ``SYNONYMS``) run on the
  training caption.

Design choices for team review: ``UNINFORMATIVE``, p_cap for L1 / L2, the stratum rule, the 3 / 3
POPE counts, and ``SYNONYMS``.
"""

from __future__ import annotations

import hashlib
import math
import re
import zlib
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from contam.config import content_hash, file_sha256, read_json

POOL_FORMAT_VERSION = 1
QUESTIONS_FORMAT_VERSION = 1
LEVELS = ("L3", "L2", "L1", "Cap")
HALVES = ("A", "B")
IMAGE_PREFIX = "coco_val2017"
VQA_SUFFIX = "\nAnswer the question using a single word or phrase."   # the 665K VQAv2 format
UNINFORMATIVE = "This is a photo."
POPE_TEMPLATE = "Is there {np} in the image?"
NEG_TYPES = ("random", "popular", "adversarial")

FLICKR_IMG = re.compile(r"staticflickr\.com/(?:\d+/)*\d+/(\d+)_[0-9a-f]+")
FLICKR_PAGE = re.compile(r"flickr\.com/photos/[^/]+/(\d+)")
_NUM = re.compile(r"(\d+)\.jpg$")
_TOK = re.compile(r"[a-z0-9]+")

# Extra surface forms per COCO category (category name -> forms); the name itself always counts.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "person": ("people", "man", "men", "woman", "women", "boy", "girl", "child", "children", "kid", "player", "guy", "lady", "baby", "skier", "surfer", "skateboarder", "rider", "person"),
    "bicycle": ("bike",),
    "motorcycle": ("motorbike", "motor bike", "scooter"),
    "airplane": ("plane", "jet", "aircraft", "airliner"),
    "car": ("cars", "taxi", "suv"),
    "bus": ("buses",),
    "train": ("locomotive",),
    "truck": ("pickup",),
    "boat": ("ship", "kayak", "canoe", "sailboat"),
    "traffic light": ("stoplight", "traffic signal"),
    "fire hydrant": ("hydrant",),
    "stop sign": (),
    "parking meter": ("meter",),
    "bench": (),
    "bird": ("pigeon", "seagull", "duck", "parrot"),
    "cat": ("kitten", "kitty"),
    "dog": ("puppy", "pup"),
    "horse": ("pony",),
    "sheep": ("lamb",),
    "cow": ("cattle", "calf", "bull"),
    "elephant": (), "bear": (), "zebra": (), "giraffe": (),
    "backpack": ("back pack",),
    "umbrella": ("parasol",),
    "handbag": ("purse", "hand bag"),
    "tie": ("necktie",),
    "suitcase": ("luggage", "suit case"),
    "frisbee": (),
    "skis": ("ski",),
    "snowboard": (),
    "sports ball": ("ball", "soccer ball", "football", "baseball", "basketball", "tennis ball"),
    "kite": (),
    "baseball bat": ("bat",),
    "baseball glove": ("glove", "mitt"),
    "skateboard": ("skate board",),
    "surfboard": ("surf board",),
    "tennis racket": ("racket", "racquet"),
    "bottle": (),
    "wine glass": ("glass of wine",),
    "cup": ("mug",),
    "fork": (), "knife": (), "spoon": (),
    "bowl": (),
    "banana": (), "apple": (), "sandwich": (), "orange": (), "broccoli": (), "carrot": (),
    "hot dog": ("hotdog",),
    "pizza": (),
    "donut": ("doughnut",),
    "cake": ("cupcake",),
    "chair": (),
    "couch": ("sofa",),
    "potted plant": ("plant", "houseplant"),
    "bed": (),
    "dining table": ("table",),
    "toilet": (),
    "tv": ("television", "tv"),
    "laptop": ("computer", "notebook"),
    "mouse": (),
    "remote": ("controller", "remote control"),
    "keyboard": (),
    "cell phone": ("phone", "cellphone", "smartphone"),
    "microwave": (), "oven": ("stove",), "toaster": (), "sink": (), "refrigerator": ("fridge",),
    "book": (), "clock": (), "vase": (), "scissors": (),
    "teddy bear": ("teddy",),
    "hair drier": ("hair dryer", "hairdryer"),
    "toothbrush": ("tooth brush",),
}


def flickr_id(url: str | None) -> str | None:
    if not url:
        return None
    m = FLICKR_IMG.search(url) or FLICKR_PAGE.search(url)
    return m.group(1) if m else None


def norm_tokens(text: str) -> list[str]:
    return _TOK.findall(text.lower())


# ----------------------------------------------------------------------------- pool


def seen_ids(coco_images: Sequence[Mapping], mix: Sequence[Mapping], vg_images: Sequence[Mapping],
             lcs_meta: Sequence[Mapping] = (), coco_train_images: Sequence[Mapping] = ()) -> dict[int, list[str]]:
    """``{coco_id: [reasons]}`` for the given COCO images that LLaVA-1.5 saw (module docstring)."""
    by_flickr = {}
    ids = set()
    for im in coco_images:
        ids.add(int(im["id"]))
        f = flickr_id(im.get("flickr_url"))
        if f:
            by_flickr[f] = int(im["id"])
    vg = {int(v["image_id"]): v for v in vg_images}
    train_flickr = {int(im["id"]): flickr_id(im.get("flickr_url")) for im in coco_train_images}
    out: dict[int, set[str]] = defaultdict(set)

    def hit(cid, reason):
        if cid in ids:
            out[cid].add(reason)

    used_vg, used_coco = set(), set()
    for ex in mix:
        im = ex.get("image")
        if not im:
            continue
        m = _NUM.search(im)
        if not m:
            continue
        if im.startswith(("vg/", "gqa/")):
            used_vg.add(int(m.group(1)))
        elif im.startswith("coco/"):
            used_coco.add(int(m.group(1)))
    for v in sorted(used_vg):
        r = vg.get(v)
        if not r:
            continue
        if r.get("coco_id"):
            hit(int(r["coco_id"]), "665k_vg_gqa:coco_id")
        f = str(r["flickr_id"]) if r.get("flickr_id") else None
        if f in by_flickr:
            hit(by_flickr[f], "665k_vg_gqa:flickr_id")
    for c in sorted(used_coco):
        hit(c, "665k_coco:coco_id")
        f = train_flickr.get(c)
        if f in by_flickr:
            hit(by_flickr[f], "665k_coco:flickr_id")
    for r in lcs_meta:
        f = flickr_id(r.get("url"))
        if f in by_flickr:
            hit(by_flickr[f], "lcs558k:flickr_id")
    return {k: sorted(v) for k, v in sorted(out.items())}


def _largest_remainder(sizes: Mapping[str, int], fraction: float) -> dict[str, int]:
    total = int(math.floor(fraction * sum(sizes.values()) + 0.5))
    keys = sorted(sizes)
    base = {k: int(math.floor(fraction * sizes[k])) for k in keys}
    rem = sorted(keys, key=lambda k: (-(fraction * sizes[k] - base[k]), k))
    for k in rem[: max(0, total - sum(base.values()))]:
        base[k] += 1
    return base


def _stratum_rng(seed: int, stratum: str) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(stratum.encode())])


def build_pool(instances: Mapping, captions: Mapping, vqa_questions: Sequence[Mapping], seen: Mapping[int, Sequence[str]],
               split_seed: int, ref_seed: int, n_in_vg: Iterable[int] = (), inputs_sha256: Mapping[str, str] | None = None) -> dict:
    """The frozen pool: clean images with half (A / B), stratum and training-caption index."""
    cats = {c["id"]: c for c in instances["categories"]}
    area = defaultdict(lambda: defaultdict(float))
    for a in instances["annotations"]:
        if not a.get("iscrowd"):
            area[a["image_id"]][a["category_id"]] += float(a["area"])
    has_q = {int(q["image_id"]) for q in vqa_questions}
    caps = defaultdict(list)
    for c in sorted(captions["annotations"], key=lambda c: c["id"]):
        caps[int(c["image_id"])].append(c["caption"].strip())
    images = {int(im["id"]): im for im in instances["images"]}
    drop = Counter()
    keep = []
    for iid in sorted(images):
        if iid in seen:
            drop["seen_by_llava"] += 1
        elif not area.get(iid):
            drop["no_object"] += 1
        elif iid not in has_q:
            drop["no_vqav2_val_question"] += 1
        elif len(caps[iid]) < 2:
            drop["fewer_than_2_captions"] += 1
        else:
            keep.append(iid)

    def stratum(iid):
        cid = max(area[iid].items(), key=lambda kv: (kv[1], -kv[0]))[0]
        return cats[cid]["supercategory"]

    strata = defaultdict(list)
    for iid in keep:
        strata[stratum(iid)].append(iid)
    order = {s: [ids[i] for i in _stratum_rng(split_seed, s).permutation(len(ids))] for s, ids in sorted(strata.items())}
    alloc = _largest_remainder({s: len(v) for s, v in strata.items()}, 0.5)
    half_a = {i for s in order for i in order[s][: alloc[s]]}
    rng = np.random.default_rng(int(ref_seed))
    rows = []
    for iid in keep:
        rows.append({"image_id": iid, "file_name": images[iid]["file_name"], "stratum": stratum(iid), "half": "A" if iid in half_a else "B",
                     "n_captions": len(caps[iid]), "train_caption_idx": int(rng.integers(0, len(caps[iid])))})
    vg_set = set(int(i) for i in n_in_vg)
    body = {
        "format_version": POOL_FORMAT_VERSION,
        "name": "coco_pool",
        "split_seed": int(split_seed),
        "ref_seed": int(ref_seed),
        "rule": "COCO val2017 with >=1 non-crowd object and VQAv2 val questions, minus images LLaVA-1.5 saw (665K coco / vg / gqa, LCS-558K)",
        "inputs_sha256": dict(inputs_sha256 or {}),
        "counts": {"val2017": len(images), "dropped": dict(drop), "pool": len(rows),
                   "half": dict(Counter(r["half"] for r in rows)), "stratum_by_half": {h: dict(Counter(r["stratum"] for r in rows if r["half"] == h)) for h in HALVES},
                   "n_in_vg_unused": len(vg_set & set(keep))},
        "seen": {str(k): list(v) for k, v in seen.items()},
        "images": rows,
    }
    body["content_hash"] = content_hash(body)
    return body


def validate_pool(pool: Mapping) -> None:
    body = {k: v for k, v in pool.items() if k != "content_hash"}
    if content_hash(body) != pool.get("content_hash"):
        raise ValueError("coco pool content hash mismatch")
    ids = [r["image_id"] for r in pool["images"]]
    if len(ids) != len(set(ids)) or set(map(int, pool["seen"])) & set(ids):
        raise ValueError("coco pool has duplicate or seen images")


def captions_by_image(captions: Mapping) -> dict[int, list[str]]:
    out = defaultdict(list)
    for c in sorted(captions["annotations"], key=lambda c: c["id"]):
        out[int(c["image_id"])].append(c["caption"].strip())
    return dict(out)


def train_caption(pool_row: Mapping, caps: Mapping[int, Sequence[str]]) -> str:
    cs = caps[int(pool_row["image_id"])]
    if len(cs) != pool_row["n_captions"]:
        raise ValueError(f"image {pool_row['image_id']}: {len(cs)} captions, the pool was built with {pool_row['n_captions']}")
    return cs[pool_row["train_caption_idx"]]


def load_inputs(cfg: dict, pool: dict) -> tuple[dict, dict, dict]:
    """``(captions by image, VQAv2 questions by image, VQAv2 answers by question id)``, checked against
    the input hashes recorded in the pool."""
    c = cfg["coco_study"]
    for key in ("captions", "vqa_questions", "vqa_annotations"):
        if file_sha256(c[key]) != pool["inputs_sha256"][key]:
            raise ValueError(f"{c[key]} is not the {key} file the pool was built from ({pool['inputs_sha256'][key][:12]})")
    caps = captions_by_image(read_json(c["captions"]))
    qs = defaultdict(list)
    for q in read_json(c["vqa_questions"])["questions"]:
        qs[int(q["image_id"])].append(q)
    ans = {a["question_id"]: a for a in read_json(c["vqa_annotations"])["annotations"]}
    for q in (q for v in qs.values() for q in v):
        q["multiple_choice_answer"] = ans[q["question_id"]]["multiple_choice_answer"]
    return caps, dict(qs), ans


# ----------------------------------------------------------------------------- exposure


def _turns(pairs: Sequence[tuple[str, str]], image: bool) -> list[dict]:
    conv = []
    for k, (q, a) in enumerate(pairs):
        conv.append({"from": "human", "value": ("<image>\n" + q) if image and k == 0 else q})
        conv.append({"from": "gpt", "value": a})
    return conv


def run_id(level: str, half: str, d: int, slot: int) -> str:
    return f"C_{level}_{half}_d{d}_s{slot}"


def build_exposure(level: str, half: str, pool: Mapping, caps: Mapping[int, Sequence[str]], vqa_by_image: Mapping[int, Sequence[Mapping]],
                   d: int, prompts: Mapping[str, str]) -> dict:
    """Unique exposed rows of one level and half (each with ``repeat = d``) plus metadata."""
    if level not in LEVELS or half not in HALVES:
        raise ValueError(f"level {level!r} / half {half!r}")
    if d < 1:
        raise ValueError("d must be >= 1")
    rows = []
    for r in sorted((r for r in pool["images"] if r["half"] == half), key=lambda r: r["image_id"]):
        iid, img = r["image_id"], f"{IMAGE_PREFIX}/{r['file_name']}"
        if level == "L3":
            qs = sorted(vqa_by_image[iid], key=lambda q: q["question_id"])
            conv = _turns([(q["question"] + VQA_SUFFIX, q["multiple_choice_answer"]) for q in qs], True)
        elif level == "L2":
            conv = _turns([(prompts["caption"], train_caption(r, caps))], True)
        elif level == "L1":
            conv = _turns([(prompts["caption"], UNINFORMATIVE)], True)
        else:
            conv = _turns([(prompts["caption_text"], train_caption(r, caps))], False)
        row = {"id": f"coco_{level}:{iid}", "coco_image_id": iid, "conversations": conv, "repeat": int(d)}
        if level != "Cap":
            row["image"] = img
        rows.append(row)
    meta = {"format_version": 1, "level": level, "half": half, "pool_content_hash": pool["content_hash"], "n_exp": len(rows), "d": int(d),
            "K": len(rows) * int(d), "uninformative": UNINFORMATIVE if level == "L1" else None, "vqa_suffix": VQA_SUFFIX if level == "L3" else None,
            "prompts": {k: prompts[k] for k in (("caption_text",) if level == "Cap" else ("caption",) if level in ("L1", "L2") else ())}}
    body = {"meta": meta, "rows": rows}
    body["content_hash"] = content_hash(body)
    return body


def expand(exposure: Mapping) -> list[dict]:
    keep = ("id", "image", "conversations")
    return [{k: r[k] for k in keep if k in r} for r in exposure["rows"] for _ in range(r["repeat"])]


# ----------------------------------------------------------------------------- questions


def _plural(w: str) -> set[str]:
    out = {w, w + "s", w + "es"}
    if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
        out.add(w[:-1] + "ies")
    return out


def _forms(w: str) -> frozenset[str]:
    """``w`` and its possible singular forms (``buses`` -> buses, buse, bus; ``glass`` -> glass)."""
    out = {w}
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        out.add(w[:-1])
        if w.endswith("es"):
            out.add(w[:-2])
        if w.endswith("ies") and len(w) > 3:
            out.add(w[:-3] + "y")
    return frozenset(out)


def answer_mentioned(answer: str, caption: str) -> bool:
    """The normalized answer occurs as a token sequence in the caption; two words match when their
    singular forms overlap (``dogs`` matches ``dog``, ``bus`` matches ``buses``)."""
    a = [_forms(t) for t in norm_tokens(answer)]
    c = [_forms(t) for t in norm_tokens(caption)]
    n = len(a)
    return bool(a) and any(all(a[k] & c[i + k] for k in range(n)) for i in range(len(c) - n + 1))


def vqav2_questions(pool: Mapping, caps: Mapping[int, Sequence[str]], vqa_by_image: Mapping[int, Sequence[Mapping]],
                    answers: Mapping[int, Mapping]) -> list[dict]:
    out = []
    for r in sorted(pool["images"], key=lambda r: r["image_id"]):
        cap = train_caption(r, caps)
        for q in sorted(vqa_by_image[r["image_id"]], key=lambda q: q["question_id"]):
            a = answers[q["question_id"]]
            t = a["answer_type"]
            cell = t if t in ("yes/no", "number") else ("mentioned" if answer_mentioned(a["multiple_choice_answer"], cap) else "unmentioned")
            out.append({"qid": f"vqa:{q['question_id']}", "question_id": q["question_id"], "image_id": r["image_id"], "file_name": r["file_name"],
                        "half": r["half"], "question": q["question"], "answer_type": t, "multiple_choice_answer": a["multiple_choice_answer"],
                        "answers": [x["answer"] for x in a["answers"]], "cell": cell})
    return out


class CocoMatcher:
    """COCO category mentions in spaCy-tagged text (``[(lower token, UPOS), ...]``): longest match
    first, last token a noun, last word possibly plural; names plus ``SYNONYMS``."""

    def __init__(self, names: Sequence[str], synonyms: Mapping[str, Sequence[str]] = SYNONYMS):
        self.variants: dict[tuple[str, ...], str] = {}
        for name in names:
            for form in (name, *synonyms.get(name, ())):
                toks = norm_tokens(form)
                for last in _plural(toks[-1]):
                    self.variants.setdefault((*toks[:-1], last), name)
        self.max_len = max(len(k) for k in self.variants)

    def __call__(self, tagged: Sequence[tuple[str, str]]) -> set[str]:
        toks = [(t, p) for t, p in tagged if _TOK.fullmatch(t)]
        found, i = set(), 0
        while i < len(toks):
            for n in range(min(self.max_len, len(toks) - i), 0, -1):
                key = tuple(t for t, _ in toks[i:i + n])
                if key in self.variants and toks[i + n - 1][1] in ("NOUN", "PROPN"):
                    found.add(self.variants[key])
                    i += n
                    break
            else:
                i += 1
        return found


def _np(name: str) -> str:
    return ("an " if name[0] in "aeiou" else "a ") + name


def pope_questions(pool: Mapping, instances: Mapping, caps: Mapping[int, Sequence[str]], matcher: CocoMatcher, tagged: Mapping[str, list],
                   seed: int, k_pos: int = 3, k_neg: int = 3) -> list[dict]:
    """POPE-style yes/no questions (module docstring). ``tagged`` maps every caption to its spaCy tags."""
    names = {c["id"]: c["name"] for c in instances["categories"]}
    area = defaultdict(lambda: defaultdict(float))
    present = defaultdict(set)
    for a in instances["annotations"]:
        present[a["image_id"]].add(names[a["category_id"]])          # crowd regions count as present (never a negative)
        if not a.get("iscrowd"):
            area[a["image_id"]][names[a["category_id"]]] += float(a["area"])
    pool_ids = [r["image_id"] for r in pool["images"]]
    freq = Counter(n for i in pool_ids for n in present[i])
    popular = [n for n, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]
    co = defaultdict(Counter)
    for i in pool_ids:
        for a in present[i]:
            for b in present[i]:
                if a != b:
                    co[a][b] += 1
    all_names = sorted(names.values())
    out = []
    for r in sorted(pool["images"], key=lambda r: r["image_id"]):
        iid = r["image_id"]
        train = matcher(tagged[train_caption(r, caps)])
        any_cap = set().union(*(matcher(tagged[c]) for c in caps[iid]))
        pos = [n for n, _ in sorted(area[iid].items(), key=lambda kv: (-kv[1], kv[0]))][:k_pos]
        absent = [n for n in all_names if n not in present[iid] and n not in any_cap]
        rng = np.random.default_rng([int(seed), iid])
        cand = {"random": [absent[j] for j in rng.permutation(len(absent))],
                "popular": [n for n in popular if n in absent],
                "adversarial": [n for n, _ in sorted(((n, sum(co[p][n] for p in present[iid])) for n in absent), key=lambda kv: (-kv[1], kv[0]))]}
        base = {"image_id": iid, "file_name": r["file_name"], "half": r["half"]}
        for n in pos:
            out.append({**base, "qid": f"pope:{iid}:pos:{n}", "object": n, "question": POPE_TEMPLATE.format(np=_np(n)), "answer": "yes",
                        "neg_type": None, "cell": "pos/mentioned" if n in train else "pos/unmentioned"})
        for t in NEG_TYPES:
            for n in cand[t][:k_neg]:
                out.append({**base, "qid": f"pope:{iid}:{t}:{n}", "object": n, "question": POPE_TEMPLATE.format(np=_np(n)), "answer": "no",
                            "neg_type": t, "cell": f"neg/{t}"})
    return out


def review_rows(questions: Sequence[Mapping], caps: Mapping[int, Sequence[str]], pool: Mapping, cells: Sequence[str], n_per_cell: int = 100,
                seed: int = 20261118) -> list[dict]:
    """``n_per_cell`` questions per mention cell for the hand check (Injoh), with the training caption."""
    by_id = {r["image_id"]: r for r in pool["images"]}
    out = []
    for cell in cells:
        sel = sorted((q for q in questions if q["cell"] == cell), key=lambda q: hashlib.sha256(f"{seed}|{q['qid']}".encode()).hexdigest())[:n_per_cell]
        for q in sel:
            out.append({"cell": cell, "qid": q["qid"], "image_id": q["image_id"], "file_name": q["file_name"], "question": q["question"],
                        "answer": q.get("multiple_choice_answer", q.get("answer")), "object": q.get("object", ""),
                        "training_caption": train_caption(by_id[q["image_id"]], caps), "label_correct": "", "note": ""})
    return out
