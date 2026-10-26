#!/usr/bin/env python
"""Word-level perturbations of the training references for the B6 DCQ quiz (contam.eval.detectors).

For every image in the DCQ subset of a split (``detectors.dcq``: seen and unseen, B1 exclusion
applied), asks an instruction-tuned text LM (``detectors.dcq.perturb_model``) for four rewrites of
the image's training reference that swap words for contextual synonyms and keep the sentence
structure, as in Golchin & Surdeanu (2023). A rewrite set is kept only if it passes
``check_perturbations`` (four, distinct, none equal to the original, word count within
``perturb_max_len_diff``); otherwise it is asked again with the next seed, up to ``--tries`` times.
The output is a frozen input of every DCQ run (its sha256 is recorded there), so make it once per
split and keep it.

Writes JSONL rows ``{image_id, ref_idx, original, perturbations, model, seed, tries}``; resumable.

Example:
    python scripts/dcq_perturb.py --split data/splits/split_A.json --out outputs/detectors/dcq_perturbations_A.jsonl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from contam.config import append_jsonl, load_config, read_json, read_jsonl
from contam.data.freeze import add_check_args, cli_check
from contam.eval import detectors as det

INSTRUCTION = (
    "Your task is to write four different versions of the text below. In each version, replace only some of the words "
    "with contextually relevant synonyms. The meaning and the sentence structure of every version must mirror every detail "
    "of the text, and no version may be identical to the text or to another version.\n"
    "Text: {text}\n"
    "Answer with exactly four lines, starting with 'A) ', 'B) ', 'C) ' and 'D) ', and nothing else."
)
_LINE = re.compile(r"^\s*([A-D])[).:]\s*(.+?)\s*$")


def parse_versions(text: str) -> list[str]:
    """The four lettered lines of the LM's answer, in A-D order (fewer if it did not comply)."""
    got = {}
    for line in text.splitlines():
        m = _LINE.match(line)
        if m and m.group(1) not in got:
            got[m.group(1)] = m.group(2).strip().strip('"')
    return [got[L] for L in "ABCD" if L in got]


class ChatLM:
    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device, self.model_id = torch, device, model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()

    def __call__(self, text: str, seed: int, max_new_tokens: int = 256) -> str:
        torch = self.torch
        msgs = [{"role": "user", "content": INSTRUCTION.format(text=text)}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        torch.manual_seed(seed)
        with torch.inference_mode():
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0, top_p=1.0, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)


def perturb(lm, original: str, seed: int, tries: int, max_len_diff: int) -> tuple[list[str] | None, int]:
    for t in range(tries):
        versions = parse_versions(lm(original, seed + t))
        if not det.check_perturbations(original, versions, max_len_diff):
            return versions, t + 1
    return None, tries


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default=None, help="HF chat LM (default: detectors.dcq.perturb_model)")
    ap.add_argument("--tries", type=int, default=5)
    ap.add_argument("--keep-excluded", action="store_true")
    ap.add_argument("--max-images", type=int, default=None, help="smoke tests: first N images per group")
    ap.add_argument("--out", required=True)
    add_check_args(ap)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    cli_check(args, cfg, args.split, "dcq_perturb")

    from contam.data.nocaps import load_annotations, refs_by_id
    from contam.data.splits import check_annotations

    dcfg = cfg["detectors"]["dcq"]
    ann = load_annotations(cfg["paths"]["annotations"])
    split = read_json(args.split)
    check_annotations(split, ann)
    refs = refs_by_id(ann)
    groups, _ = det.detector_groups(cfg, split, "dcq", args.keep_excluded)
    ids = sorted(set(groups["seen"][: args.max_images]) | set(groups["unseen"][: args.max_images]))
    done = {int(r["image_id"]) for r in read_jsonl(args.out)} if Path(args.out).exists() else set()
    todo = [i for i in ids if i not in done]
    model_id = args.model or dcfg["perturb_model"]
    lm = ChatLM(model_id)
    from tqdm import tqdm

    failed = []
    for i in tqdm(todo, desc="DCQ perturbations"):
        j = int(split["train_ref_idx"][str(i)])
        seed = int(dcfg["perturb_seed"]) + 1000 * i
        versions, n = perturb(lm, refs[i][j], seed, args.tries, dcfg["perturb_max_len_diff"])
        if versions is None:
            failed.append(i)
            continue
        append_jsonl([dict(image_id=i, ref_idx=j, original=refs[i][j], perturbations=versions, model=model_id, seed=seed, tries=n)], args.out)
    print(f"{len(todo) - len(failed)} new items -> {args.out}")
    if failed:
        raise SystemExit(f"{len(failed)} images got no valid perturbations after {args.tries} tries (rerun with more --tries): {failed[:10]}")


if __name__ == "__main__":
    main()
