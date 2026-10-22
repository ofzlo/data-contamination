"""Caption generation for one checkpoint (base model + optional LoRA adapter).

Writes JSONL rows ``{checkpoint_id, model, adapter, image_id, prompt_id, prompt_sha, temperature,
sample_idx, seed, batch_size, max_new_tokens, caption}`` (``prompt_sha`` = sha256 of the exact
prompt text).

Resumable: the done-key is (image_id, prompt_id, prompt_sha, model, adapter, temperature,
sample_idx). A file that already holds rows for the same prompt_id / temperature from a different
checkpoint, prompt text, model, adapter, batch_size or max_new_tokens is refused (write a new
file), so one file never mixes runs.

* T=0: greedy, one row per image (``sample_idx`` 0, ``seed`` null). Batched greedy decoding can
  depend on batch composition (padding / kernel numerics), so batches are FIXED: the sorted image
  ids are cut into consecutive chunks of ``batch_size``. A chunk with any missing row (e.g. the
  partial last chunk of an earlier ``--max-images`` run) is regenerated whole: its stale rows are
  first dropped by atomically rewriting the file, so every chunk in the final file was generated
  in one batch.
* T>0: ``n`` samples per image generated in one call with ``num_return_sequences=n`` and seed
  ``t1_base_seed + image_id``, so each image's samples are reproducible regardless of resumption.

Example:
    python -m contam.generate --checkpoint-id d0_s0 --adapter /scratch/.../d0_s0 \
        --split data/splits/split_A.json --out outputs/gens/d0_s0.jsonl --temperature 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from contam.config import append_jsonl, load_config, load_prompts, prompt_sha, read_json, read_jsonl
from contam.data.freeze import add_check_args, cli_check

ROW_FIELDS = ("checkpoint_id", "model", "adapter", "image_id", "prompt_id", "prompt_sha", "temperature", "sample_idx", "seed", "batch_size", "max_new_tokens", "caption")


def _key(r: dict) -> tuple:
    return (int(r["image_id"]), r["prompt_id"], r.get("prompt_sha"), r.get("model"), r.get("adapter"), float(r["temperature"]), int(r["sample_idx"]))


def done_keys(path: str | Path) -> set[tuple]:
    if not Path(path).exists():
        return set()
    return {_key(r) for r in read_jsonl(path)}


def _check_existing(path: str | Path, prov: dict, prompt_id: str, temperature: float) -> None:
    if not Path(path).exists():
        return
    for r in read_jsonl(path):
        if r["prompt_id"] != prompt_id or float(r["temperature"]) != temperature:
            continue
        for f, v in prov.items():
            if r.get(f) != v:
                raise ValueError(f"{path} already has {prompt_id} T={temperature} rows with {f}={r.get(f)!r} (now {v!r}); write to a new file")


def _drop_rows(path: str | Path, keys: set[tuple]) -> None:
    rows = [r for r in read_jsonl(path) if _key(r) not in keys]
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def run_generation(
    vlm,
    load_image,
    image_ids: Sequence[int],
    prompt_id: str,
    prompt: str,
    out_path: str | Path,
    checkpoint_id: str,
    temperature: float = 0.0,
    num_samples: int = 1,
    base_seed: int = 1234,
    batch_size: int = 16,
    max_new_tokens: int = 300,
    progress: bool = True,
) -> int:
    """Generate missing rows and append them to ``out_path``. Returns the number of new rows."""
    temperature = float(temperature)
    n = 1 if temperature == 0 else int(num_samples)
    prov = dict(
        checkpoint_id=checkpoint_id,
        model=getattr(vlm, "model_id", None),
        adapter=getattr(vlm, "adapter", None),
        prompt_sha=prompt_sha(prompt),
        batch_size=batch_size if temperature == 0 else 1,
        max_new_tokens=max_new_tokens,
    )
    _check_existing(out_path, prov, prompt_id, temperature)
    done = done_keys(out_path)

    def key(iid, k):
        return (int(iid), prompt_id, prov["prompt_sha"], prov["model"], prov["adapter"], temperature, k)

    ids = sorted(int(i) for i in image_ids)
    if temperature == 0:
        chunks = [ids[s : s + batch_size] for s in range(0, len(ids), batch_size)]
    else:
        chunks = [[i] for i in ids]
    chunks = [c for c in chunks if any(key(i, k) not in done for i in c for k in range(n))]
    if temperature == 0:
        stale = {key(i, 0) for c in chunks for i in c} & done
        if stale:
            _drop_rows(out_path, stale)
            done -= stale
    it = chunks
    if progress:
        from tqdm import tqdm

        it = tqdm(chunks, desc=f"gen {checkpoint_id} {prompt_id} T={temperature}")
    written = 0
    for chunk in it:
        if temperature == 0:
            caps = vlm.generate([load_image(i) for i in chunk], prompt, 0.0, max_new_tokens)
            rows = [dict(image_id=i, sample_idx=0, seed=None, caption=c) for i, c in zip(chunk, caps) if key(i, 0) not in done]
        else:
            iid = chunk[0]
            seed = int(base_seed) + iid
            caps = vlm.generate([load_image(iid)], prompt, temperature, max_new_tokens, num_return_sequences=n, seed=seed)
            rows = [dict(image_id=iid, sample_idx=k, seed=seed, caption=c) for k, c in enumerate(caps) if key(iid, k) not in done]
        for r in rows:
            r.update(prov, prompt_id=prompt_id, temperature=temperature)
        append_jsonl([{k: r[k] for k in ROW_FIELDS} for r in rows], out_path)
        written += len(rows)
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate captions for one checkpoint.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--model", default=None, help="HF model id (default: config models.main)")
    ap.add_argument("--adapter", default=None, help="optional PEFT LoRA adapter dir")
    ap.add_argument("--checkpoint-id", required=True)
    ap.add_argument("--split", default=None, help="split JSON; generate for seen + unseen images (default: all annotated images)")
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--prompt-id", default="caption")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num-samples", type=int, default=None, help="samples per image at T>0 (default: config)")
    ap.add_argument("--max-images", type=int, default=None, help="first N image ids only (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=None)
    add_check_args(ap)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.split:
        cli_check(args, cfg, args.split, "generate")

    from contam.data.nocaps import ImageLoader, load_annotations
    from contam.data.splits import check_annotations
    from contam.models import HFVLM

    prompts = load_prompts(args.prompts)
    ann = load_annotations(args.annotations or cfg["paths"]["annotations"])
    if args.split:
        sp = read_json(args.split)
        check_annotations(sp, ann)
        ids = sorted(set(sp["seen"]) | set(sp["unseen"]))
    else:
        ids = sorted(int(a["image_id"]) for a in ann)
    if args.max_images:
        ids = ids[: args.max_images]
    g = cfg["generation"]
    vlm = HFVLM(args.model or cfg["models"]["main"], adapter=args.adapter, dtype=cfg["models"]["dtype"], end_token=cfg["models"]["end_token"])
    n = run_generation(
        vlm,
        ImageLoader(ann, args.images_dir or cfg["paths"]["images_dir"]),
        ids,
        args.prompt_id,
        prompts[args.prompt_id],
        args.out,
        args.checkpoint_id,
        temperature=args.temperature,
        num_samples=args.num_samples or g["t1_num_samples"],
        base_seed=g["t1_base_seed"],
        batch_size=args.batch_size or g["batch_size"],
        max_new_tokens=g["max_new_tokens"],
    )
    print(f"wrote {n} new rows -> {args.out}")


if __name__ == "__main__":
    main()
