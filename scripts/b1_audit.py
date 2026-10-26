#!/usr/bin/env python
"""B1 training-data audit of LLaVA-1.5 against NoCaps val (images and reference captions).

Checks, for the 4,500 NoCaps val images and their 45,000 normalized references:

1. Same image, by id. Open Images ids against the 665K ``image`` paths (TextVQA images are Open
   Images train, named by id), and Flickr photo ids against every Flickr-sourced training set we
   can resolve: COCO train2017 (``flickr_url``; in 665K and the SameDist source), Visual Genome
   (``flickr_id``; 665K vg/ and gqa/) and the LCS-558K pretraining set (LAION / CC / SBU ``url``s
   on staticflickr). NoCaps val images are Open Images validation images, which come from Flickr.
2. Same caption text. A reference counts as present when its normalized form (lowercase, ASCII
   letters and digits, single spaces) equals a normalized sentence of a 665K conversation turn, an
   LCS-558K BLIP caption, or a COCO train2017 caption. Short generic captions can match by chance,
   so matches are reported for references of >= 8 words separately.

Not covered: near-duplicate images under other ids or URLs (needs pixel hashes of ~100 GB of
images), CLIP's and Vicuna's own training data (not public), and web text outside these files.

Example:
    python scripts/b1_audit.py --out docs/results/b1_audit.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from contam.config import load_config, read_json, write_json
from contam.data.nocaps import load_annotations

FLICKR_IMG = re.compile(r"staticflickr\.com/(?:\d+/)*\d+/(\d+)_[0-9a-f]+")
FLICKR_PAGE = re.compile(r"flickr\.com/photos/[^/]+/(\d+)")
_NORM = re.compile(r"[^a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def norm(text: str) -> str:
    return _NORM.sub(" ", text.lower()).strip()


def flickr_id(url: str | None) -> str | None:
    if not url:
        return None
    m = FLICKR_IMG.search(url) or FLICKR_PAGE.search(url)
    return m.group(1) if m else None


def sentences(text: str) -> list[str]:
    return [s for s in (norm(x) for x in _SENT.split(text.replace("<image>", " "))) if s]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--oi-images", default="data/audit/oi_validation_images.csv", help="Open Images validation-images-with-rotation.csv")
    ap.add_argument("--vg-image-data", default="data/audit/image_data.json")
    ap.add_argument("--lcs558k", default="data/audit/llava_pretrain/blip_laion_cc_sbu_558k_meta.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ann = load_annotations(cfg["paths"]["annotations"])
    oi_ids = {a["open_images_id"] for a in ann}

    # NoCaps val -> Flickr photo ids
    nc_flickr = {}
    with open(args.oi_images, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ImageID"] in oi_ids:
                fid = flickr_id(row["OriginalURL"]) or flickr_id(row["OriginalLandingURL"])
                if fid:
                    nc_flickr[fid] = row["ImageID"]
    rep: dict = {"nocaps_images": len(oi_ids), "nocaps_with_flickr_id": len(nc_flickr)}

    mix = read_json(cfg["paths"]["llava_mix"])
    stem = lambda p: Path(p).stem
    rep["llava665k_open_images_id_overlap"] = sorted({ex["image"] for ex in mix if ex.get("image") and stem(ex["image"]) in oi_ids})

    coco = read_json(cfg["paths"]["coco_captions"])
    coco_f = {flickr_id(im.get("flickr_url")): im["id"] for im in coco["images"]}
    vg = read_json(args.vg_image_data)
    vg_f = {str(im["flickr_id"]): im["image_id"] for im in vg if im.get("flickr_id")}
    lcs = read_json(args.lcs558k)
    lcs_f = {}
    for r in lcs:
        fid = flickr_id(r.get("url"))
        if fid:
            lcs_f[fid] = r["id"]
    for name, ids in (("coco_train2017", coco_f), ("visual_genome", vg_f), ("lcs558k", lcs_f)):
        hit = sorted(set(nc_flickr) & set(ids))
        rep[f"flickr_overlap_{name}"] = {"training_flickr_ids": len(ids), "nocaps_hits": len(hit), "examples": [{"flickr_id": h, "open_images_id": nc_flickr[h], "train_id": ids[h]} for h in hit[:20]]}

    # NoCaps val images that are the same Flickr photo as a COCO / VG training image, and the 665K
    # examples and COCO ids to exclude (general pool: --exclude-images; SameDist: --exclude-coco-ids)
    coco_hit = {coco_f[h]: nc_flickr[h] for h in set(nc_flickr) & set(coco_f)}
    vg_hit = {str(vg_f[h]): nc_flickr[h] for h in set(nc_flickr) & set(vg_f)}
    num = re.compile(r"(\d+)\.jpg$")
    excl = sorted({ex["image"] for ex in mix if ex.get("image") and (
        (ex["image"].startswith("coco/") and int(num.search(ex["image"]).group(1)) in coco_hit)
        or (ex["image"].startswith(("vg/", "gqa/")) and num.search(ex["image"]) and num.search(ex["image"]).group(1) in vg_hit))})
    oi2iid = {a["open_images_id"]: a["image_id"] for a in ann}
    rep["overlap_nocaps_image_ids"] = sorted({oi2iid[o] for o in list(coco_hit.values()) + list(vg_hit.values())})
    rep["exclude_coco_ids"] = sorted(coco_hit)
    rep["exclude_665k_images"] = excl
    rep["exclude_665k_examples"] = sum(1 for ex in mix if ex.get("image") in set(excl))

    # text
    refs = [(a["image_id"], r) for a in ann for r in a["refs"]]
    pools = {
        "llava665k_turns": {s for ex in mix for t in ex["conversations"] for s in sentences(t["value"])},
        "lcs558k_blip": {norm(r["blip_caption"]) for r in lcs},
        "coco_train2017": {norm(a["caption"]) for a in coco["annotations"]},
    }
    for name, pool in pools.items():
        hits = [(i, r) for i, r in refs if norm(r) in pool]
        long_hits = [(i, r) for i, r in hits if len(norm(r).split()) >= 8]
        rep[f"text_{name}"] = {"pool_size": len(pool), "refs_matched": len(hits), "refs_matched_ge8_words": len(long_hits),
                               "images_with_match": len({i for i, _ in hits}), "images_with_match_ge8_words": len({i for i, _ in long_hits}),
                               "top_matches": Counter(norm(r) for _, r in hits).most_common(10),
                               "ge8_examples": [{"image_id": i, "ref": r} for i, r in long_hits[:15]]}
    rep["nocaps_refs"] = len(refs)
    print(json.dumps(rep, indent=1)[:6000])
    if args.out:
        write_json(rep, args.out)
        out = Path(args.out).with_suffix("")
        Path(f"{out}_exclude_images.txt").write_text("".join(f"{p}\n" for p in excl))
        Path(f"{out}_exclude_coco_ids.txt").write_text("".join(f"{c}\n" for c in sorted(coco_hit)))


if __name__ == "__main__":
    main()
