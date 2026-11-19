#!/usr/bin/env python
"""Score one checkpoint on the COCO image-study questions: VQAv2 answers and/or POPE yes/no (``contam.eval.coco_eval``).

Checks before scoring:
* the question file's content hash is intact;
* its pool is the locked ``coco_pool.json`` (manifest lock).

Images are read from ``coco_study.images_dir``. Writes ``<out-dir>/<checkpoint-id>.vqa.jsonl`` and
``<out-dir>/<checkpoint-id>.pope.jsonl``. Both are resumable, with fixed batches of ``--batch-size``.
Score every checkpoint of a comparison on the same GPU type.

Example:
    python scripts/coco_eval_score.py --checkpoint-id C_L3_A_d16_s0 --adapter $SHARED/ckpt/C_L3_A_d16_s0
    python scripts/coco_eval_score.py --checkpoint-id base            # untrained base model (Leak sanity: M_A = M_B)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from contam.config import content_hash, load_config, read_json
from contam.data import coco_study as cs
from contam.data.freeze import DEFAULT_LOCK, verify
from contam.eval import coco_eval as ce
from contam.eval import yesno


def load_question_file(path, cfg, lock_path=DEFAULT_LOCK) -> dict:
    q = read_json(path)
    body = {k: v for k, v in q.items() if k != "content_hash"}
    if content_hash(body) != q.get("content_hash"):
        raise SystemExit(f"{path}: content hash mismatch (file changed after it was built)")
    try:
        lock = verify(cfg, lock_path, allow_unlocked=True, required=["coco_pool.json"])
    except RuntimeError as e:
        raise SystemExit(str(e))
    if lock["files"]["coco_pool.json"].get("content_hash") != q["pool_content_hash"]:
        raise SystemExit(f"{path} was built from pool {q['pool_content_hash'][:12]}, not the locked coco_pool.json")
    return q


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint-id", required=True)
    ap.add_argument("--adapter", default=None, help="PEFT adapter dir (omit for the untrained base model)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--task", choices=["vqa", "pope", "both"], default="both")
    ap.add_argument("--questions", default="outputs/coco/coco_questions.json")
    ap.add_argument("--out-dir", default="outputs/coco_eval")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=10)
    ap.add_argument("--max-questions", type=int, default=None, help="first N questions per task (smoke tests)")
    ap.add_argument("--lock", default=str(DEFAULT_LOCK))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    qf = load_question_file(args.questions, cfg, args.lock)
    sha = qf["content_hash"]

    from PIL import Image

    from contam.models import HFVLM

    pool = {r["image_id"]: r["file_name"] for r in read_json(Path(cfg["paths"]["splits_dir"]) / "coco_pool.json")["images"]}
    img_dir = Path(cfg["coco_study"]["images_dir"])

    def load_image(iid: int):
        with Image.open(img_dir / pool[int(iid)]) as im:
            return im.convert("RGB")

    vlm = HFVLM(args.model or cfg["models"]["main"], adapter=args.adapter, dtype=cfg["models"]["dtype"], end_token=cfg["models"]["end_token"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.task in ("vqa", "both"):
        tok = vlm.processor.tokenizer
        qs = ce.vqa_order(qf["vqav2"], lambda q: len(tok(vlm.prefix(q["question"] + ce.VQA_SUFFIX))["input_ids"]))[: args.max_questions]
        n = ce.score_vqa(vlm, load_image, qs, out / f"{args.checkpoint_id}.vqa.jsonl", args.checkpoint_id, sha, args.batch_size, args.max_new_tokens)
        print(f"vqa: wrote {n} rows")
    if args.task in ("pope", "both"):
        qs = ce.sort_questions(qf["pope"])[: args.max_questions]
        n = yesno.score_questions(vlm, load_image, qs, out / f"{args.checkpoint_id}.pope.jsonl", args.checkpoint_id, sha, batch_size=args.batch_size)
        print(f"pope: wrote {n} rows")


if __name__ == "__main__":
    main()
