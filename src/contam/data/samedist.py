"""Same-distribution control (D4): COCO train captions matched to a split's exposed NoCaps items.

SameDist replaces the NoCaps items of an exposure run with COCO train2017 captions, so it carries
the same caption register without exposing any NoCaps image or reference. COCO and NoCaps share
the annotation protocol but not images (NoCaps images come from Open Images), so there is no
eval-image overlap.

Matching (D4: "matched to NoCaps in count, length and total K")
---------------------------------------------------------------
* **Count.** One SameDist row per seen image of the split, keyed by ``nocaps_image_id``. A run at
  exposure fraction f uses the rows of the images in ``exposure_sets[f]``, so SameDist sets are
  nested exactly like the NoCaps exposure sets (3.125% ⊂ 12.5% ⊂ 50%).
* **Length.** Each row's caption has the same word count as that image's training reference
  (``refs[train_ref_idx]``), counted with ``contam.eval.style.words`` (the S metric's word count).
  If a length bucket runs out, the nearest length is used (shorter first on ties); the build
  reports how many rows are exact.
* **Total K.** The run builder repeats each row d times, as it does the NoCaps item, so
  K = N_exp × d is identical by construction.

Each COCO image contributes at most one caption (like the one training reference per NoCaps image).
Within a length bucket captions are ordered by sha256 of (``seed``, COCO annotation id), so the
result does not depend on input order. Rows are assigned smallest exposure set first (images
sorted by id inside each set), so a smaller set's rows do not depend on the larger sets.

COCO captions are whitespace-normalized (runs collapsed to one space, ends stripped); nothing else
is changed. Rows keep the COCO image, so D3 can turn them into Paired-format (image + caption) or
Caption-format (text only) examples.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import re
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from contam.config import content_hash, load_config, read_json, write_json
from contam.data.splits import _fkey, check_annotations, group_ids, validate_split
from contam.eval.style import words

SAMEDIST_FORMAT_VERSION = 1
_WS = re.compile(r"\s+")


def normalize_caption(text: str) -> str:
    return _WS.sub(" ", text).strip()


def n_words(text: str) -> int:
    return len(words(text))


def load_coco_captions(path, exclude_image_ids: Iterable[int] = ()) -> list[dict]:
    """``captions_train2017.json`` -> ``[{image_id, ann_id, file_name, caption}]`` (excluded images dropped)."""
    obj = read_json(path)
    files = {int(im["id"]): im["file_name"] for im in obj["images"]}
    excl = {int(i) for i in exclude_image_ids}
    out = []
    for a in obj["annotations"]:
        iid = int(a["image_id"])
        if iid in excl:
            continue
        cap = normalize_caption(a["caption"])
        if cap:
            out.append({"image_id": iid, "ann_id": int(a["id"]), "file_name": files[iid], "caption": cap})
    return out


def _tiebreak(seed: int, ann_id: int) -> bytes:
    return hashlib.sha256(f"samedist|{int(seed)}|{int(ann_id)}".encode()).digest()


class _LengthPool:
    """COCO captions bucketed by word count; take() pops the best unused caption of the nearest length."""

    def __init__(self, captions: Sequence[dict], seed: int):
        buckets: dict[int, list] = defaultdict(list)
        for c in captions:
            buckets[n_words(c["caption"])].append((_tiebreak(seed, c["ann_id"]), c))
        # reversed so list.pop() returns the smallest tie-break hash
        self.buckets = {n: [c for _, c in sorted(b, key=lambda t: t[0], reverse=True)] for n, b in buckets.items() if n > 0}
        self.lengths = sorted(self.buckets)
        self.used_images: set[int] = set()

    def _pop(self, n: int) -> dict | None:
        b = self.buckets[n]
        while b:
            c = b.pop()
            if c["image_id"] not in self.used_images:
                self.used_images.add(c["image_id"])
                if not b:
                    self._drop(n)
                return c
        self._drop(n)
        return None

    def _drop(self, n: int) -> None:
        del self.buckets[n]
        self.lengths.remove(n)

    def take(self, target: int) -> dict:
        while self.lengths:
            i = bisect.bisect_left(self.lengths, target)
            n = min(self.lengths[max(i - 1, 0) : i + 1], key=lambda m: (abs(m - target), m))
            c = self._pop(n)
            if c is not None:
                return c
        raise RuntimeError("COCO caption pool exhausted")


def make_samedist(split: dict, annotations: Sequence[dict], coco_captions: Sequence[dict], seed: int, source: dict | None = None) -> dict:
    """Build the SameDist control for ``split`` (see module docstring). ``source`` is recorded as-is."""
    validate_split(split)
    check_annotations(split, annotations)
    refs = {int(a["image_id"]): list(a["refs"]) for a in annotations}
    pool = _LengthPool(coco_captions, seed)

    sets = split["exposure_sets"]
    fractions = sorted((float(k) for k in sets), key=float)  # smallest first
    rows: dict[int, dict] = {}
    for f in fractions:
        for iid in sorted(int(i) for i in sets[_fkey(f)]):
            if iid in rows:
                continue
            target = n_words(refs[iid][split["train_ref_idx"][str(iid)]])
            c = pool.take(target)
            rows[iid] = {
                "nocaps_image_id": iid,
                "level": f,  # smallest exposure fraction containing this image
                "target_words": target,
                "coco_image_id": c["image_id"],
                "coco_ann_id": c["ann_id"],
                "coco_file_name": c["file_name"],
                "caption": c["caption"],
                "caption_words": n_words(c["caption"]),
            }
    if set(rows) != {int(i) for i in split["seen"]}:
        raise AssertionError("SameDist rows must cover exactly the seen images")

    ordered = [rows[i] for i in sorted(rows)]
    body = {
        "format_version": SAMEDIST_FORMAT_VERSION,
        "name": split["name"],
        "split": {"name": split["name"], "content_hash": split["content_hash"]},
        "annotations_hash": split["annotations_hash"],
        "seed": int(seed),
        "source": source or {},
        "word_count": "contam.eval.style.words",
        "stats": _stats(ordered, split),
        "rows": ordered,
    }
    body["content_hash"] = content_hash(body)
    return body


def _stats(rows: Sequence[dict], split: dict) -> dict:
    by_id = {r["nocaps_image_id"]: r for r in rows}
    out = {}
    for k, ids in split["exposure_sets"].items():
        rs = [by_id[int(i)] for i in ids]
        t = np.array([r["target_words"] for r in rs], dtype=float)
        c = np.array([r["caption_words"] for r in rs], dtype=float)
        out[k] = {
            "n": len(rs),
            "exact_length": int(np.sum(t == c)),
            "max_abs_word_diff": int(np.max(np.abs(t - c))) if len(rs) else 0,
            "words_nocaps_train_refs": int(t.sum()),
            "words_samedist": int(c.sum()),
            "mean_words_nocaps": round(float(t.mean()), 4) if len(rs) else 0.0,
            "mean_words_samedist": round(float(c.mean()), 4) if len(rs) else 0.0,
        }
    return out


def validate_samedist(sd: dict, split: dict) -> None:
    """Raise AssertionError if ``sd`` does not match ``split`` or breaks a SameDist invariant."""
    assert sd["split"]["content_hash"] == split["content_hash"], "SameDist built from a different split"
    ids = [r["nocaps_image_id"] for r in sd["rows"]]
    assert sorted(ids) == sorted(int(i) for i in split["seen"]), "rows must cover exactly the seen images"
    coco = [r["coco_image_id"] for r in sd["rows"]]
    assert len(set(coco)) == len(coco), "a COCO image is used twice"
    for r in sd["rows"]:
        assert r["caption_words"] == n_words(r["caption"]), f"word count mismatch for COCO ann {r['coco_ann_id']}"
    body = {k: v for k, v in sd.items() if k != "content_hash"}
    assert content_hash(body) == sd["content_hash"], "content hash mismatch"


def samedist_rows(sd: dict, split: dict, group: str) -> list[dict]:
    """Rows for an exposure fraction (e.g. "0.125"), ``seen`` or ``shared_seen``, in image-id order."""
    want = set(group_ids(split, str(group)))
    return [r for r in sd["rows"] if r["nocaps_image_id"] in want]


def main(argv: Sequence[str] | None = None) -> None:
    from contam.config import file_sha256
    from contam.data.nocaps import load_annotations

    ap = argparse.ArgumentParser(description="Build the SameDist control (D4) for splits A and B.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--annotations", default=None, help="annotations.json (default: config paths.annotations)")
    ap.add_argument("--coco", default=None, help="captions_train2017.json (default: config paths.coco_captions)")
    ap.add_argument("--split", action="append", default=None, help="split name(s) (default: A and B)")
    ap.add_argument("--splits-dir", default=None, help="where split_{name}.json live and samedist_{name}.json go")
    ap.add_argument("--exclude-coco-ids", default=None, help="text file, one COCO image id per line (e.g. from the B1 audit)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = cfg["paths"]
    ann_path = args.annotations or paths["annotations"]
    coco_path = args.coco or paths["coco_captions"]
    splits_dir = args.splits_dir or paths["splits_dir"]
    exclude = []
    if args.exclude_coco_ids:
        with open(args.exclude_coco_ids) as f:
            exclude = [int(x) for x in f.read().split()]
    ann = load_annotations(ann_path)
    coco = load_coco_captions(coco_path, exclude)
    source = {
        "coco_file": str(coco_path).replace("\\", "/").split("/")[-1],
        "coco_sha256": file_sha256(coco_path),
        "n_coco_captions": len(coco),
        "n_coco_images": len({c["image_id"] for c in coco}),
        "n_excluded_coco_images": len(set(exclude)),
    }
    seed = int(cfg["samedist"]["seed"])
    for name in args.split or ["A", "B"]:
        split = read_json(f"{splits_dir}/split_{name}.json")
        sd = make_samedist(split, ann, coco, seed, source)
        validate_samedist(sd, split)
        path = f"{splits_dir}/samedist_{name}.json"
        write_json(sd, path, indent=None)
        print(f"[samedist {name}] split={split['content_hash'][:12]} seed={seed} hash={sd['content_hash'][:12]} -> {path}")
        for k, s in sorted(sd["stats"].items(), key=lambda kv: float(kv[0]), reverse=True):
            print(f"  exposure {float(k):.5f}: n={s['n']} exact_len={s['exact_length']} max|diff|={s['max_abs_word_diff']} words nocaps/samedist={s['words_nocaps_train_refs']}/{s['words_samedist']}")


if __name__ == "__main__":
    main()
