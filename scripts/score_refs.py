#!/usr/bin/env python
"""Teacher-forced log p of all 10 references per image for one checkpoint (input to V / RMI).

Writes JSONL rows ``{checkpoint_id, model, adapter, image_id, ref_idx, prompt_id, prompt_sha, logp_sum,
n_tokens}``; resumable per image (an image is done only when all its refs are present; see
contam.eval.evaluate.prepare_logprob_resume). Uses the same prompt template as generation
(contam.models.HFVLM).

Example:
    python scripts/score_refs.py --checkpoint-id d0_s0 --adapter /scratch/.../d0_s0 \
        --split data/splits/split_A.json --out outputs/logprobs/d0_s0.jsonl
"""

from __future__ import annotations

import argparse

from contam.config import append_jsonl, load_config, load_prompts, prompt_sha, read_json
from contam.data.freeze import add_check_args, cli_check
from contam.data.nocaps import ImageLoader, load_annotations, refs_by_id
from contam.data.splits import check_annotations
from contam.eval.evaluate import prepare_logprob_resume
from contam.eval.memory import score_all_refs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--checkpoint-id", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--prompt-id", default="caption")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--chunk-images", type=int, default=64, help="images per write (resumption granularity)")
    add_check_args(ap)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cli_check(args, cfg, args.split, "score_refs")

    from contam.models import HFVLM

    prompt = load_prompts(args.prompts)[args.prompt_id]
    ann = load_annotations(args.annotations or cfg["paths"]["annotations"])
    refs = refs_by_id(ann)
    sp = read_json(args.split)
    check_annotations(sp, ann)
    ids = sorted(set(sp["seen"]) | set(sp["unseen"]))[: args.max_images]
    model_id = args.model or cfg["models"]["main"]
    prov = dict(checkpoint_id=args.checkpoint_id, model=model_id, adapter=str(args.adapter) if args.adapter else None, prompt_sha=prompt_sha(prompt))
    done = prepare_logprob_resume(args.out, prov, {i: len(r) for i, r in refs.items()}, args.prompt_id)
    ids = [i for i in ids if i not in done]
    vlm = HFVLM(model_id, adapter=args.adapter, dtype=cfg["models"]["dtype"], end_token=cfg["models"]["end_token"])
    loader = ImageLoader(ann, args.images_dir or cfg["paths"]["images_dir"])
    from tqdm import tqdm

    for start in tqdm(range(0, len(ids), args.chunk_images), desc=f"score {args.checkpoint_id}"):
        chunk = ids[start : start + args.chunk_images]
        res = score_all_refs(vlm, loader, chunk, refs, prompt, batch_size=cfg["scoring"]["batch_size"])
        rows = [
            dict(**prov, image_id=i, ref_idx=j, prompt_id=args.prompt_id, logp_sum=float(res[i].logp_sum[j]), n_tokens=int(res[i].n_tokens[j]))
            for i in chunk
            for j in range(len(refs[i]))
        ]
        append_jsonl(rows, args.out)
    print(f"scored {len(ids)} images -> {args.out}")


if __name__ == "__main__":
    main()
