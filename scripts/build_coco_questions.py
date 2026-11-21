#!/usr/bin/env python
"""COCO image-study evaluation questions and the mention-label review file (``contam.data.coco_study``).

From the frozen pool (lock-checked), this writes:
* ``coco_questions.json``:
  * every VQAv2 val question of every pool image, with its 10 answers and its cell (``yes/no``,
    ``number``, or for ``other`` answers ``mentioned`` / ``unmentioned`` in the image's training
    caption);
  * POPE-style yes/no questions with cells ``pos/mentioned``, ``pos/unmentioned``, ``neg/random``,
    ``neg/popular`` and ``neg/adversarial``.
  Every question carries its half (A / B), so the Leak metric pairs models and halves.
* ``coco_mention_review.csv``: 100 questions per mention cell, with the training caption, for the
  hand check (Injoh). Fill ``label_correct`` with y / n.

Upload both to ``$HF_REPO`` under ``outputs/coco/`` (overall_guideline M-1).

Example:
    python scripts/build_coco_questions.py --out outputs/coco
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from contam.config import content_hash, file_sha256, load_config, read_json, write_json
from contam.data import coco_study as cs
from contam.data.freeze import DEFAULT_LOCK, verify

POOL_FILE = "coco_pool.json"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="default: <outputs_dir>/coco")
    ap.add_argument("--n-review", type=int, default=100)
    ap.add_argument("--lock", default=str(DEFAULT_LOCK))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    c = cfg["coco_study"]
    try:
        lock = verify(cfg, args.lock, allow_unlocked=True, required=[POOL_FILE])
    except RuntimeError as e:
        raise SystemExit(str(e))
    pool = read_json(Path(cfg["paths"]["splits_dir"]) / POOL_FILE)
    cs.validate_pool(pool)
    if lock["files"][POOL_FILE].get("content_hash") != pool["content_hash"]:
        raise SystemExit(f"{POOL_FILE} content hash differs from the lock")
    for key in ("instances", "captions", "vqa_questions", "vqa_annotations"):
        if file_sha256(c[key]) != pool["inputs_sha256"][key]:
            raise SystemExit(f"{c[key]} is not the {key} file the pool was built from")

    from contam.eval.style_transfer import tag_texts

    try:
        caps, vqa_by_image, answers = cs.load_inputs(cfg, pool)
    except ValueError as e:
        raise SystemExit(str(e))
    inst = read_json(c["instances"])
    pool_ids = {r["image_id"] for r in pool["images"]}
    print("tagging captions with spaCy ...")
    tagged = tag_texts(t for i in pool_ids for t in caps[i])
    matcher = cs.CocoMatcher([cat["name"] for cat in inst["categories"]])
    vqa = cs.vqav2_questions(pool, caps, vqa_by_image, answers)
    pope = cs.pope_questions(pool, inst, caps, matcher, tagged, c["pope_seed"])
    body = {"format_version": cs.QUESTIONS_FORMAT_VERSION, "pool_content_hash": pool["content_hash"], "pope_seed": c["pope_seed"],
            "synonyms_hash": content_hash(cs.SYNONYMS), "pos_tagger": "spacy en_core_web_sm (UPOS)",
            "counts": {"vqav2": {h: dict(Counter(q["cell"] for q in vqa if q["half"] == h)) for h in cs.HALVES},
                       "pope": {h: dict(Counter(q["cell"] for q in pope if q["half"] == h)) for h in cs.HALVES}},
            "vqav2": vqa, "pope": pope}
    body["content_hash"] = content_hash(body)
    out = Path(args.out) if args.out else Path(cfg["paths"]["outputs_dir"]) / "coco"
    out.mkdir(parents=True, exist_ok=True)
    write_json(body, out / "coco_questions.json")
    rows = cs.review_rows(vqa, caps, pool, ["mentioned", "unmentioned"], args.n_review) + \
        cs.review_rows(pope, caps, pool, ["pos/mentioned", "pos/unmentioned"], args.n_review)
    with open(out / "coco_mention_review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(body["counts"], indent=1))
    print(f"-> {out / 'coco_questions.json'} (content_hash {body['content_hash'][:12]}), {out / 'coco_mention_review.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
