"""NoCaps validation: conversion from the lmms-lab parquet mirror and annotation loading.

The lmms-lab mirror (validation-0000{0,1,2}-of-00003-*.parquet) carries images, image ids and the
10 reference captions, but NOT the in/near/out domain. The domain comes from a separate table
keyed by Open Images id; we use the ``image_domain`` column of the laicsiifes/nocaps-pt-br mirror
(same image ids and Open Images ids; its captions are Portuguese and are NOT used). The official
``nocaps_val_4500_captions.json`` would also work but its S3 bucket returns 403.

``annotations.json`` is ``{"meta": {...}, "images": [record, ...]}`` with records sorted by
``image_id``::

    {"image_id": int, "file_name": str, "open_images_id": str, "domain": "in"|"near"|"out",
     "refs": [str x 10]}

Reference normalization (recorded in ``meta.ref_normalization``):

* surrounding whitespace is stripped;
* ``strip_trailing_not_sign`` (default True): a trailing "¬" (U+00AC; 458 NoCaps val refs end
  with it, an annotation-tool artifact) is removed, e.g. "A man riding a wave¬" -> "A man riding a
  wave". Three refs contain "¬" mid-caption; those are left unchanged.

The SAME normalized references must be used as training targets (Minho) and for scoring.
"""

from __future__ import annotations

import glob
import os
import re
from collections import Counter
from pathlib import Path
from typing import Mapping

from contam.config import read_json, write_json
from contam.data.splits import normalize_domain

_TRAILING_NOT = re.compile(r"\s*\u00ac+\s*$")


def normalize_ref(text: str, strip_trailing_not_sign: bool = True) -> str:
    t = text.strip()
    if strip_trailing_not_sign:
        t = _TRAILING_NOT.sub("", t)
    return t


EXPECTED_DOMAIN_COUNTS = {"in": 417, "near": 2670, "out": 1413}
N_REFS = 10


def load_domain_table(path: str | os.PathLike) -> dict[str, str]:
    """``{open_images_id: domain}`` from a parquet (``image_open_images_id``, ``image_domain``) or JSON file.

    JSON may be the official NoCaps file (``images[*].open_images_id`` / ``domain``) or a flat dict.
    """
    path = str(path)
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        t = pq.read_table(path, columns=["image_open_images_id", "image_domain"])
        return dict(zip(t.column("image_open_images_id").to_pylist(), (normalize_domain(d) for d in t.column("image_domain").to_pylist())))
    obj = read_json(path)
    if isinstance(obj, dict) and "images" in obj:
        return {im["open_images_id"]: normalize_domain(im["domain"]) for im in obj["images"]}
    return {k: normalize_domain(v) for k, v in obj.items()}


def convert_parquet(
    parquet_dir: str | os.PathLike,
    domain_table: Mapping[str, str],
    out_annotations: str | os.PathLike,
    images_dir: str | os.PathLike | None,
    strip_trailing_not_sign: bool = True,
) -> list[dict]:
    """Convert lmms-lab NoCaps val shards to ``annotations.json`` and (optionally) an images directory.

    Images are written as the original encoded bytes (no re-encoding); existing files are kept.
    """
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(str(parquet_dir), "validation-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no validation-*.parquet in {parquet_dir}")
    if images_dir is not None:
        Path(images_dir).mkdir(parents=True, exist_ok=True)
    records = []
    cols = ["image_id", "image_file_name", "image_open_images_id", "annotations_captions"]
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=cols + (["image"] if images_dir is not None else []))
            for row in t.to_pylist():
                oi = row["image_open_images_id"]
                if oi not in domain_table:
                    raise KeyError(f"no domain for open_images_id {oi}")
                refs = [normalize_ref(c, strip_trailing_not_sign) for c in row["annotations_captions"]]
                if len(refs) != N_REFS:
                    raise ValueError(f"image {row['image_id']} has {len(refs)} refs, expected {N_REFS}")
                records.append(
                    {
                        "image_id": int(row["image_id"]),
                        "file_name": row["image_file_name"],
                        "open_images_id": oi,
                        "domain": domain_table[oi],
                        "refs": refs,
                    }
                )
                if images_dir is not None:
                    dst = Path(images_dir) / row["image_file_name"]
                    if not dst.exists():
                        tmp = dst.with_suffix(dst.suffix + ".tmp")
                        tmp.write_bytes(row["image"]["bytes"])
                        os.replace(tmp, dst)
    records.sort(key=lambda r: r["image_id"])
    ids = [r["image_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate image ids in parquet shards")
    meta = {
        "source": [os.path.basename(f) for f in files],
        "n_images": len(records),
        "ref_normalization": {"strip_whitespace": True, "strip_trailing_not_sign": bool(strip_trailing_not_sign)},
    }
    write_json({"meta": meta, "images": records}, out_annotations, indent=None)
    return records


def domain_counts(records) -> dict[str, int]:
    return dict(Counter(r["domain"] for r in records))


def load_annotations(path: str | os.PathLike) -> list[dict]:
    """Image records from annotations.json (also accepts the older bare-list format)."""
    obj = read_json(path)
    return obj["images"] if isinstance(obj, dict) else obj


def load_annotations_meta(path: str | os.PathLike) -> dict:
    obj = read_json(path)
    return obj.get("meta", {}) if isinstance(obj, dict) else {}


def refs_by_id(annotations) -> dict[int, list[str]]:
    return {int(a["image_id"]): list(a["refs"]) for a in annotations}


class ImageLoader:
    """Loads NoCaps images by image id as RGB PIL images."""

    def __init__(self, annotations, images_dir: str | os.PathLike):
        self.files = {int(a["image_id"]): Path(images_dir) / a["file_name"] for a in annotations}

    def __call__(self, image_id: int):
        from PIL import Image

        with Image.open(self.files[int(image_id)]) as im:
            return im.convert("RGB")
