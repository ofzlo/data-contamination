"""Exposure-condition training examples (task 1.3) for one split, condition, N_exp and d.

Every condition uses the same exposed images: the first ``n_exp`` of the split's exposure order
(ground rule 3). For N_exp = 2,250 / 563 / 141 that is a stored exposure set; any other N (e.g.
1,000 if D2 picks it) is rebuilt from the split's per-stratum permutation, so it is nested between
the stored sets. Each exposed item is repeated d times, so K = N_exp × d.

Conditions (``CONDITIONS``), all in LLaVA conversation format with image paths relative to one
image root (``nocaps/<file>``, ``coco/train2017/<file>``, as in LLaVA-665K):

* ``paired``   NoCaps image + p_cap -> its training reference.
* ``caption``  text only: ``prompts.caption_text`` -> the training reference (D3 template).
* ``image``    NoCaps image + p_cap -> the base model's own T=0 caption of that image (D3), read
               from a ``contam.generate`` file of the untrained base model.
* ``shuffled`` NoCaps image + p_cap -> the training reference of ANOTHER exposed image. The
               permutation is a derangement (images sorted by sha256 of (shuffle seed, id), each
               mapped to the next), so no image keeps its own caption, and every exposed caption
               and image is still seen exactly d times.
* ``samedist`` COCO image + p_cap -> the length-matched COCO caption of the SameDist row paired
               with each exposed image (``samedist_{split}.json``, D4).

Rows are unique examples with ``repeat = d``; the trainer expands them. Targets are the normalized
references from annotations.json, the same strings used for scoring.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from contam.config import content_hash, read_jsonl
from contam.data.splits import DOMAINS, _fkey, _stratum_rng, allocate_stratified, check_annotations, normalize_domain, validate_split

CONDITIONS = ("paired", "caption", "image", "shuffled", "samedist")
EXPOSURE_FORMAT_VERSION = 1


def exposure_ids(split: dict, annotations: Sequence[dict], n_exp: int) -> list[int]:
    """The first ``n_exp`` exposed images of ``split`` (sorted ids), nested with the stored sets."""
    validate_split(split)
    check_annotations(split, annotations)
    stored = {len(v): sorted(int(i) for i in v) for v in split["exposure_sets"].values()}
    if n_exp in stored:
        return stored[n_exp]
    if split.get("kind") == "class_disjoint":
        raise ValueError(f"class-disjoint split {split['name']}: only its stored exposure set ({sorted(stored)}) exists")
    n_seen = len(split["seen"])
    if not 0 < n_exp <= n_seen:
        raise ValueError(f"n_exp={n_exp} must be in 1..{n_seen} (the seen group)")
    strata: dict[str, list[int]] = {d: [] for d in DOMAINS}
    for a in sorted(annotations, key=lambda a: int(a["image_id"])):
        strata[normalize_domain(a["domain"])].append(int(a["image_id"]))
    order = {d: [ids[i] for i in _stratum_rng(split["split_seed"], d).permutation(len(ids))] for d, ids in strata.items()}
    sizes = {d: len(v) for d, v in strata.items()}
    total = sum(sizes.values())

    def prefix(n: int) -> set[int]:
        alloc = allocate_stratified(sizes, n / total)
        return {i for d in DOMAINS for i in order[d][: alloc[d]]}

    for n, ids in stored.items():  # the rebuilt order must reproduce every stored set
        if prefix(n) != set(ids):
            raise AssertionError(f"rebuilt exposure order does not reproduce the stored {n}-image set")
    out = prefix(n_exp)
    if len(out) != n_exp:
        raise AssertionError(f"rebuilt exposure set has {len(out)} images, expected {n_exp}")
    for n, ids in stored.items():
        if n < n_exp and not set(ids) <= out or n > n_exp and not out <= set(ids):
            raise AssertionError(f"N_exp={n_exp} set is not nested with the stored {n}-image set")
    return sorted(out)


def derangement(ids: Sequence[int], seed: int) -> dict[int, int]:
    """``{image_id: caption_source_id}`` with no fixed point (needs >= 2 ids)."""
    if len(ids) < 2:
        raise ValueError("shuffled needs at least 2 exposed images")
    ring = sorted(ids, key=lambda i: (hashlib.sha256(f"shuffle|{int(seed)}|{int(i)}".encode()).digest(), i))
    return {ring[k]: ring[(k + 1) % len(ring)] for k in range(len(ring))}


def base_captions(path, prompt_sha: str) -> dict[int, str]:
    """T=0 captions of the base model from a ``contam.generate`` file, for the given prompt."""
    out: dict[int, str] = {}
    for r in read_jsonl(path):
        if float(r["temperature"]) == 0.0 and r.get("prompt_sha") == prompt_sha and int(r["sample_idx"]) == 0:
            if r.get("adapter"):
                raise ValueError(f"{path}: Image targets must come from the untrained base model (row has adapter {r['adapter']!r})")
            out[int(r["image_id"])] = r["caption"].strip()
    return out


def _conv(prompt: str, target: str, image: bool) -> list[dict]:
    return [{"from": "human", "value": ("<image>\n" + prompt) if image else prompt}, {"from": "gpt", "value": target}]


def build_exposure(
    condition: str,
    split: dict,
    annotations: Sequence[dict],
    n_exp: int,
    d: int,
    prompts: dict[str, str],
    shuffle_seed: int | None = None,
    samedist: dict | None = None,
    base_caps: dict[int, str] | None = None,
) -> dict:
    """Unique exposed rows (each with ``repeat = d``) plus metadata; see the module docstring."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; one of {CONDITIONS}")
    if d < 1:
        raise ValueError("d must be >= 1 (d=0 has no exposed examples)")
    ids = exposure_ids(split, annotations, n_exp)
    by_id = {int(a["image_id"]): a for a in annotations}
    ref = lambda i: by_id[i]["refs"][split["train_ref_idx"][str(i)]]
    p_cap = prompts["caption"]
    rows = []
    if condition == "paired":
        rows = [{"id": f"paired:{i}", "nocaps_image_id": i, "image": f"nocaps/{by_id[i]['file_name']}", "conversations": _conv(p_cap, ref(i), True)} for i in ids]
    elif condition == "caption":
        rows = [{"id": f"caption:{i}", "nocaps_image_id": i, "conversations": _conv(prompts["caption_text"], ref(i), False)} for i in ids]
    elif condition == "image":
        if base_caps is None:
            raise ValueError("image condition needs the base model's captions (base_caps)")
        missing = [i for i in ids if i not in base_caps]
        if missing:
            raise ValueError(f"no base caption for {len(missing)} exposed images, e.g. {missing[:3]}")
        rows = [{"id": f"image:{i}", "nocaps_image_id": i, "image": f"nocaps/{by_id[i]['file_name']}", "conversations": _conv(p_cap, base_caps[i], True)} for i in ids]
    elif condition == "shuffled":
        if shuffle_seed is None:
            raise ValueError("shuffled needs shuffle_seed")
        perm = derangement(ids, shuffle_seed)
        rows = [{"id": f"shuffled:{i}", "nocaps_image_id": i, "caption_from": perm[i], "image": f"nocaps/{by_id[i]['file_name']}", "conversations": _conv(p_cap, ref(perm[i]), True)} for i in ids]
    elif condition == "samedist":
        if samedist is None or samedist["split"]["content_hash"] != split["content_hash"]:
            raise ValueError("samedist needs the SameDist file built from this split")
        sd = {r["nocaps_image_id"]: r for r in samedist["rows"]}
        rows = [{"id": f"samedist:{i}", "nocaps_image_id": i, "coco_image_id": sd[i]["coco_image_id"], "image": f"coco/train2017/{sd[i]['coco_file_name']}", "conversations": _conv(p_cap, sd[i]["caption"], True)} for i in ids]
    for r in rows:
        r["repeat"] = int(d)
    meta = {
        "format_version": EXPOSURE_FORMAT_VERSION,
        "condition": condition,
        "split": {"name": split["name"], "content_hash": split["content_hash"]},
        "n_exp": len(ids),
        "d": int(d),
        "K": len(ids) * int(d),
        "prompts": {k: prompts[k] for k in (("caption_text",) if condition == "caption" else ("caption",))},
        "shuffle_seed": shuffle_seed if condition == "shuffled" else None,
        "samedist_hash": samedist["content_hash"] if condition == "samedist" else None,
        "base_captions_hash": content_hash(sorted(base_caps.items())) if condition == "image" else None,
    }
    body = {"meta": meta, "rows": rows}
    body["content_hash"] = content_hash(body)
    return body


def expand(exposure: dict) -> list[dict]:
    """All K training examples (each unique row ``repeat`` times), without the bookkeeping fields."""
    keep = ("id", "image", "conversations")
    return [{k: r[k] for k in keep if k in r} for r in exposure["rows"] for _ in range(r["repeat"])]
