#!/usr/bin/env python
"""Phase 0: score every yes/no question of the question file on one checkpoint (``contam.eval.yesno``).

Writes one JSONL row per question (first-token log P(yes), log P(no), score = their difference, mass).
Resumable; batches are fixed, so use the same ``--batch-size`` for every checkpoint (the prereg value
is the default). Checks before scoring: the question file's sha256 is the preregistered one, its
build record (``*.meta.json``) names the locked split and the annotations in use, and every question's
image file is the annotations' file for that image id.

Example (one checkpoint, ~10 minutes on one GPU):
    python scripts/phase0_score.py --checkpoint-id G1_Paired_d4_s0 \
        --adapter $CONTAM_CKPT_DIR/G1_Paired_d4_s0/adapter --out outputs/phase0/G1_Paired_d4_s0.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from contam.config import file_sha256, load_config, read_json, resolve_path
from contam.data.freeze import add_check_args, cli_check
from contam.eval import yesno


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Phase 0 yes/no scoring of one checkpoint.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint-id", required=True)
    ap.add_argument("--adapter", default=None, help="PEFT LoRA adapter dir (omit for the untrained base model)")
    ap.add_argument("--model", default=None, help="HF model id (default: config models.main)")
    ap.add_argument("--questions", default=yesno.PREREG_QUESTIONS)
    ap.add_argument("--questions-sha256", default=yesno.PREREG_QUESTIONS_SHA256, help="'none' skips the check (smoke tests)")
    ap.add_argument("--split", default="data/splits/split_A.json", help="the split the question file was built from")
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=yesno.PREREG_BATCH_SIZE)
    ap.add_argument("--max-questions", type=int, default=None, help="first N questions only (smoke tests)")
    add_check_args(ap)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    qpath = resolve_path(args.questions)
    want = None if args.questions_sha256 == "none" else args.questions_sha256
    questions, qsha = yesno.load_questions(qpath, want)
    split_path = resolve_path(args.split)
    cli_check(args, cfg, split_path, "phase0_score")
    ann_path = Path(args.annotations or cfg["paths"]["annotations"])
    meta = read_json(qpath.with_name(qpath.name.replace(".share.json", ".meta.json")))["inputs_sha256"]
    for name, p in (("split", split_path), ("annotations", ann_path)):
        if file_sha256(p) != meta[name]:
            raise SystemExit(f"phase0_score: {p} is not the {name} file the questions were built from ({meta[name]})")

    from contam.data.nocaps import ImageLoader, load_annotations
    from contam.models import HFVLM

    ann = load_annotations(ann_path)
    files = {int(a["image_id"]): a["file_name"] for a in ann}
    bad = [q["qid"] for q in questions if files.get(int(q["image_id"])) != q["file_name"]]
    if bad:
        raise SystemExit(f"phase0_score: {len(bad)} questions name another image file than the annotations, e.g. {bad[:3]}")
    if args.max_questions:
        questions = questions[: args.max_questions]
    vlm = HFVLM(args.model or cfg["models"]["main"], adapter=args.adapter, dtype=cfg["models"]["dtype"], end_token=cfg["models"]["end_token"])
    n = yesno.score_questions(vlm, ImageLoader(ann, args.images_dir or cfg["paths"]["images_dir"]), questions, args.out,
                              args.checkpoint_id, qsha, batch_size=args.batch_size)
    print(f"wrote {n} new rows -> {args.out}")


if __name__ == "__main__":
    main()
