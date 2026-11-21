#!/usr/bin/env python
"""Build the frozen COCO image-study pool, ``<splits_dir>/coco_pool.json`` (``contam.data.coco_study``).

The pool is the COCO val2017 images LLaVA-1.5 never saw, with halves A / B and a training caption
per image. Inputs come from ``coco_study`` in the config:
* COCO val2017 instances and captions, and train2017 instances (for the Flickr ids of the 665K
  ``coco/`` images);
* VQAv2 val questions;
* Visual Genome ``image_data.json``;
* LCS-558K metadata;
* the LLaVA-665K json.

Their sha256s are recorded in the pool.

The output is deterministic. An identical existing file is left alone; a different one is refused.
Freeze it afterwards with ``python -m contam.data.freeze --add``. Training and evaluation then refuse
a changed pool. A copy also goes to the repo's ``data/splits/``, which git tracks.

Example:
    python scripts/build_coco_pool.py && python -m contam.data.freeze --add
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from contam.config import REPO_ROOT, file_sha256, load_config, read_json, write_json
from contam.data import coco_study as cs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="default: <splits_dir>/coco_pool.json")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    c = cfg["coco_study"]
    inputs = {"instances": c["instances"], "captions": c["captions"], "instances_train": c["instances_train"], "vqa_questions": c["vqa_questions"], "vqa_annotations": c["vqa_annotations"],
              "vg_image_data": c["vg_image_data"], "lcs558k_meta": c["lcs558k_meta"], "llava_mix": cfg["paths"]["llava_mix"]}
    for k, p in inputs.items():
        if not Path(p).exists():
            raise SystemExit(f"{k}: {p} is missing")
    print("hashing inputs ...")
    shas = {k: file_sha256(p) for k, p in inputs.items()}
    inst, caps = read_json(inputs["instances"]), read_json(inputs["captions"])
    vqa = read_json(inputs["vqa_questions"])["questions"]
    vg = read_json(inputs["vg_image_data"])
    print("matching against LLaVA-1.5 training data ...")
    seen = cs.seen_ids(inst["images"], read_json(inputs["llava_mix"]), vg, read_json(inputs["lcs558k_meta"]), read_json(inputs["instances_train"])["images"])
    val = {int(im["id"]) for im in inst["images"]}
    in_vg = {int(v["coco_id"]) for v in vg if v.get("coco_id")} & val
    pool = cs.build_pool(inst, caps, vqa, seen, c["split_seed"], c["ref_seed"], n_in_vg=in_vg, inputs_sha256=shas)
    cs.validate_pool(pool)

    out = Path(args.out) if args.out else Path(cfg["paths"]["splits_dir"]) / "coco_pool.json"
    tmp = out.with_name(out.name + ".new")
    write_json(pool, tmp)
    if out.exists() and file_sha256(out) != file_sha256(tmp):
        tmp.unlink()
        raise SystemExit(f"{out} exists with different content; the pool is frozen once built (move it away only with team sign-off)")
    tmp.replace(out)
    repo_copy = REPO_ROOT / "data" / "splits" / out.name
    if repo_copy.resolve() != out.resolve():
        shutil.copyfile(out, repo_copy)
    print(json.dumps(pool["counts"], indent=1))
    print(f"-> {out} (sha256 {file_sha256(out)[:12]}, content_hash {pool['content_hash'][:12]}); copy in {repo_copy}")


if __name__ == "__main__":
    main()
