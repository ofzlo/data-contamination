#!/usr/bin/env python
"""Phase 0: build yes/no object questions on NoCaps val from Open Images human-verified labels.

No training. The questions are scored on the existing caption checkpoints; see
``contam.data.oi_questions`` for the selection rules (yes = verified-present boxable class, no =
verified-absent boxable class with contradictions removed; ambiguous candidates (depiction, Man / Woman /
Boy / Girl, Mammal / Animal absent with a person) kept but never sampled; k_pos yes and min_neg..k_neg no per image),
the mention labels (training reference of the split: direct / descendant / none) and the main-analysis
filter (no B1 images, no parent classes except Person).

Seen / unseen is never the split's 2,250 / 2,250 halves: for each checkpoint type the seen group is
its exposure set (``load_seen_definition`` with N_exp), unseen the split's unseen half. The meta file
reports the main-analysis counts for N_exp in --n-exp (1,000 runs and the 30 / 100 / 300 K sweep).

Inputs under ``<data_root>/openimages`` (provenance and sha256 in its SOURCE.txt):
    oidv7-val-annotations-human-imagelabels.csv   human-verified image labels (Confidence 1 / 0)
    oidv7-class-descriptions.csv                  MID -> display name (all classes)
    class-descriptions-boxable.csv                the 600 boxable (object) classes
    bbox_labels_600_hierarchy.json                is-a hierarchy of the boxable classes
    validation-annotations-bbox.csv               boxes (IsDepiction: statue / toy / drawing; Person boxes)

Writes, under --out (default <data_root>/phase0):
    nocaps_yesno_{split}.jsonl                every candidate; ``in_sample`` / ``main_analysis`` mark the sample / main set
    nocaps_yesno_{split}.meta.json            parameters, input sha256, drops, counts (all, main, per N_exp seen / unseen)
    nocaps_yesno_{split}.phrase_review.csv    classes whose phrasing needs a manual check (reasons, frequency)
    nocaps_yesno_{split}.mention_check.csv    100 main-analysis questions per cell for the manual label / mention check
    nocaps_yesno_{split}.share.json           team file: main-analysis questions with file_name, cell and the
                                              seen / unseen / unexposed group per N_exp

Example (needs spaCy en_core_web_sm for the mention labels):
    python scripts/build_phase0_questions.py --split A
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from contam.config import content_hash, file_sha256, load_config, read_json, write_json
from contam.data import oi_questions as oq
from contam.data.nocaps import load_annotations
from contam.data.splits import check_annotations
from contam.eval.evaluate import analysis_ids, load_seen_definition
from contam.eval.style_transfer import tag_texts

B1_OVERLAP = Path(__file__).resolve().parents[1] / "resources" / "b1" / "b1_overlap_nocaps_image_ids.txt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="A", help="split name: <splits_dir>/split_{name}.json (default A, the split the checkpoints used)")
    ap.add_argument("--oi-dir", default=None, help="Open Images label dir (default <data_root>/openimages)")
    ap.add_argument("--out", default=None, help="output dir (default <data_root>/phase0)")
    ap.add_argument("--k-pos", type=int, default=3)
    ap.add_argument("--k-neg", type=int, default=3, help="at most this many no questions per image")
    ap.add_argument("--min-neg", type=int, default=1, help="images need at least this many no candidates")
    ap.add_argument("--n-exp", type=int, nargs="+", default=[1000, 300, 100, 30], help="N_exp values to report seen / unseen counts for")
    ap.add_argument("--n-check", type=int, default=100, help="questions per cell in the mention-check file")
    ap.add_argument("--seed", type=int, default=20261111)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    oi_dir = Path(args.oi_dir or Path(paths["data_root"]) / "openimages")
    out_dir = Path(args.out or Path(paths["data_root"]) / "phase0")
    inputs = {
        "annotations": Path(paths["annotations"]),
        "split": Path(paths["splits_dir"]) / f"split_{args.split}.json",
        "image_labels": oi_dir / "oidv7-val-annotations-human-imagelabels.csv",
        "class_names": oi_dir / "oidv7-class-descriptions.csv",
        "boxable": oi_dir / "class-descriptions-boxable.csv",
        "hierarchy": oi_dir / "bbox_labels_600_hierarchy.json",
        "boxes": oi_dir / "validation-annotations-bbox.csv",
        "b1_overlap": B1_OVERLAP,
    }
    for k, p in inputs.items():
        if not p.exists():
            raise FileNotFoundError(f"{k}: {p}")

    annotations = load_annotations(inputs["annotations"])
    split = read_json(inputs["split"])
    check_annotations(split, annotations)
    names = oq.load_class_names(inputs["class_names"])
    boxable = set(oq.load_class_names(inputs["boxable"]))
    ancestors = oq.load_hierarchy(inputs["hierarchy"])
    labels = oq.load_image_labels(inputs["image_labels"], (a["open_images_id"] for a in annotations))
    depictions = oq.load_depictions(inputs["boxes"], (a["open_images_id"] for a in annotations))
    b1 = [int(x) for x in inputs["b1_overlap"].read_text().split()]
    print(f"{len(annotations)} images, {len(labels)} with labels; {len(names)} classes ({len(boxable)} boxable)")

    refs = sorted({r for a in annotations for r in a["refs"]})
    print(f"tagging {len(refs)} references with spaCy ...")
    tagged = tag_texts(refs)

    rows, stats = oq.build_questions(annotations, split, labels, names, boxable, ancestors, tagged, k_pos=args.k_pos, k_neg=args.k_neg, min_neg=args.min_neg, seed=args.seed, b1_overlap=b1, depictions=depictions)

    all_ids = [int(a["image_id"]) for a in annotations]
    per_nexp, seen_sets = {}, {}
    for n in args.n_exp:
        seen = load_seen_definition(cfg, split, annotations, n_exp=n)
        groups = {g: set(analysis_ids(split, g, all_ids, seen=seen)) for g in ("seen", "unseen")}
        seen_sets[n], unseen = groups["seen"], groups["unseen"]
        per_nexp[str(n)] = {"ids_sha256": seen["ids_sha256"], **oq.count_table(rows, "main_analysis", groups)}

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"nocaps_yesno_{args.split}.jsonl"
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    tmp.replace(out)

    import spacy

    meta = {
        "format_version": oq.FORMAT_VERSION,
        "split": args.split,
        "params": {"k_pos": args.k_pos, "k_neg": args.k_neg, "min_neg": args.min_neg, "seed": args.seed, "question_template": oq.QUESTION_TEMPLATE, "plural_template": oq.PLURAL_TEMPLATE},
        "synonyms_hash": content_hash(oq.SYNONYMS),
        "phrase_lists_hash": content_hash([sorted(oq.MASS_NOUNS), sorted(oq.PLURAL_NOUNS), oq.PHRASE_OVERRIDES]),
        "pos_tagger": f"spacy {spacy.__version__} en_core_web_sm (UPOS)",
        "inputs_sha256": {k: file_sha256(p) for k, p in inputs.items()},
        "output_sha256": file_sha256(out),
        "n_candidates": len(rows),
        "n_sampled": sum(r["in_sample"] for r in rows),
        **stats,
        "main_by_n_exp": per_nexp,
    }
    review = oq.phrase_review_rows(rows)
    check = oq.mention_check_rows(rows, annotations, n_per_cell=args.n_check, seed=args.seed)
    for name, table in (("phrase_review", review), ("mention_check", check)):
        with open(out_dir / f"nocaps_yesno_{args.split}.{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    share = {
        "description": "Phase 0 yes/no object questions on NoCaps val (split " + args.split + "), main analysis only. "
        "Images: NoCaps val file_name (download NoCaps yourself). group[N_exp] = the image's group for checkpoints "
        "exposed to N_exp images (1000: main runs; 300 / 100 / 30: K sweep): seen, unseen, or unexposed (exclude).",
        "format_version": oq.FORMAT_VERSION,
        "source_sha256": meta["output_sha256"],
        "seen_ids_sha256": {n: v["ids_sha256"] for n, v in per_nexp.items()},
        "counts": {n: {g: per_nexp[n][g] for g in ("seen", "unseen")} for n in per_nexp},
        "questions": oq.share_table(rows, seen_sets, unseen),
    }
    write_json(share, out_dir / f"nocaps_yesno_{args.split}.share.json")
    meta["n_share"] = len(share["questions"])
    meta["n_phrase_review_classes"] = len(review)
    meta["n_mention_check"] = len(check)
    write_json(meta, out_dir / f"nocaps_yesno_{args.split}.meta.json")
    print(f"wrote {len(rows)} candidates ({meta['n_sampled']} sampled) -> {out}")
    print(f"images: {stats['images']}")
    print(f"drops: {stats['drops']}")
    print(f"sampled: {stats['counts_sampled']}")
    print(f"main:    {stats['counts_main']}")
    for n, v in per_nexp.items():
        print(f"N_exp={n}: seen {v['seen']} | unseen {v['unseen']}")
    print(f"phrase review: {len(review)} classes; mention check: {len(check)} rows")


if __name__ == "__main__":
    main()
