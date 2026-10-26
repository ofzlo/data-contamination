#!/usr/bin/env python
"""D6 audit: can the replication model have seen NoCaps val images? (B1 for the replication candidates)

Both candidates train only on public data, so their image sources can be matched against the 4,500
NoCaps val images (Open Images validation, all Flickr photos). CPU only; it reads one column of
each training-data parquet file over HTTP (no images are downloaded).

* SmolVLM-2.2B (``smolvlm``): its instruction data, The Cauldron, has a ``localized_narratives``
  subset built on Open Images, and its rows carry no image id. Localized Narratives also annotated
  Open Images validation, so we match texts instead: every Cauldron answer (normalized) against the
  official Localized Narratives captions of all Open Images validation images
  (``open_images_validation_localized_narratives.jsonl``), and of the NoCaps val images among them.
  A match means Cauldron used validation narratives, and with them the image.
* Molmo-7B-O (``molmo``): its PixMo sets list their images by URL. Every ``image_url`` is matched
  against the NoCaps val Flickr photo ids (as in ``scripts/b1_audit.py``) and Open Images ids.

Not covered: the academic datasets both models also train on (VQA v2 / COCO, TextVQA / Open Images
train, ...), which the B1 audit covered for LLaVA through the same COCO / VG checks, and the
vision encoders' pretraining data (SigLIP, CLIP: not public).

Example:
    python scripts/d6_audit.py --ln-val data/audit/open_images_validation_localized_narratives.jsonl --out data/audit/d6_audit.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from contam.config import load_config, write_json
from contam.data.nocaps import load_annotations

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_audit import flickr_id  # noqa: E402

CAULDRON = "datasets/HuggingFaceM4/the_cauldron/localized_narratives"
PIXMO = ["pixmo-cap", "pixmo-ask-model-anything", "pixmo-cap-qa", "pixmo-points", "pixmo-count", "pixmo-point-explanations"]
OI_ID = re.compile(r"\b([0-9a-f]{16})\b")


def norm(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", t.lower()).split())


def read_column(fs, path: str, column: str) -> list:
    import pyarrow.parquet as pq

    with fs.open(path, "rb") as f:
        return pq.ParquetFile(f).read(columns=[column]).column(column).to_pylist()


def parquet_files(fs, root: str) -> list[str]:
    return sorted(p for p in fs.find(root) if p.endswith(".parquet"))


def audit_smolvlm(fs, ln_val: Path, nocaps_oi: set[str]) -> dict:
    val, nocaps = {}, {}
    with open(ln_val, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            c = norm(r["caption"])
            if len(c.split()) >= 8:  # long enough not to match by chance
                val.setdefault(c, r["image_id"])
                if r["image_id"] in nocaps_oi:
                    nocaps.setdefault(c, r["image_id"])
    hits_val, hits_nc, n_rows = set(), set(), 0
    for p in parquet_files(fs, CAULDRON):
        for texts in read_column(fs, p, "texts"):
            n_rows += 1
            for t in texts:
                c = norm(t["assistant"])
                if c in val:
                    hits_val.add(val[c])
                if c in nocaps:
                    hits_nc.add(nocaps[c])
    return {
        "cauldron_ln_rows": n_rows,
        "ln_val_captions": len(val),
        "ln_val_nocaps_images_with_caption": len(set(nocaps.values())),
        "oi_val_images_matched": len(hits_val),
        "nocaps_images_matched": len(hits_nc),
        "nocaps_examples": sorted(hits_nc)[:20],
    }


def audit_molmo(fs, nc_flickr: dict[str, str], nocaps_oi: set[str]) -> dict:
    out = {}
    for name in PIXMO:
        urls, hits_f, hits_oi = 0, set(), set()
        for p in parquet_files(fs, f"datasets/allenai/{name}"):
            for u in read_column(fs, p, "image_url"):
                urls += 1
                fid = flickr_id(u)
                if fid in nc_flickr:
                    hits_f.add(nc_flickr[fid])
                hits_oi |= {m for m in OI_ID.findall(u or "") if m in nocaps_oi}
        out[name] = {"urls": urls, "nocaps_flickr_hits": len(hits_f), "nocaps_oi_id_hits": len(hits_oi), "examples": sorted(hits_f | hits_oi)[:20]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--oi-images", default="data/audit/oi_validation_images.csv", help="Open Images validation-images-with-rotation.csv")
    ap.add_argument("--ln-val", default="data/audit/open_images_validation_localized_narratives.jsonl")
    ap.add_argument("--only", choices=["smolvlm", "molmo"], default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from huggingface_hub import HfFileSystem

    cfg = load_config(args.config)
    ann = load_annotations(cfg["paths"]["annotations"])
    nocaps_oi = {a["open_images_id"] for a in ann}
    nc_flickr = {}
    with open(args.oi_images, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ImageID"] in nocaps_oi:
                fid = flickr_id(row["OriginalURL"]) or flickr_id(row["OriginalLandingURL"])
                if fid:
                    nc_flickr[fid] = row["ImageID"]
    fs = HfFileSystem()
    rep = {"nocaps_images": len(nocaps_oi), "nocaps_with_flickr_id": len(nc_flickr)}
    if args.only in (None, "smolvlm"):
        rep["smolvlm_cauldron_localized_narratives"] = audit_smolvlm(fs, Path(args.ln_val), nocaps_oi)
    if args.only in (None, "molmo"):
        rep["molmo_pixmo"] = audit_molmo(fs, nc_flickr, nocaps_oi)
    print(json.dumps(rep, indent=1))
    write_json(rep, args.out)


if __name__ == "__main__":
    main()
