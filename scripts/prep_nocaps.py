#!/usr/bin/env python
"""Convert the lmms-lab NoCaps validation parquet shards to annotations.json + an images directory.

Example:
    python scripts/prep_nocaps.py \
        --parquet-dir /scratch/$USER/data-contamination/data/nocaps \
        --domains /scratch/$USER/data-contamination/data/nocaps/domain_source/nocaps_val_domains.parquet
"""

from __future__ import annotations

import argparse
import sys

from contam.config import load_config
from contam.data.nocaps import EXPECTED_DOMAIN_COUNTS, convert_parquet, domain_counts, load_domain_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--parquet-dir", default=None, help="dir with validation-*.parquet (default: config paths.nocaps_parquet_dir)")
    ap.add_argument("--domains", required=True, help="domain table: parquet (image_open_images_id, image_domain) or NoCaps JSON")
    ap.add_argument("--out", default=None, help="annotations.json path (default: config paths.annotations)")
    ap.add_argument("--images-dir", default=None, help="image output dir (default: config paths.images_dir)")
    ap.add_argument("--no-images", action="store_true", help="only write annotations.json")
    ap.add_argument("--keep-trailing-not-sign", action="store_true", help="do NOT strip the trailing '¬' artifact from refs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    images_dir = None if args.no_images else (args.images_dir or paths["images_dir"])
    recs = convert_parquet(args.parquet_dir or paths["nocaps_parquet_dir"], load_domain_table(args.domains), args.out or paths["annotations"], images_dir, strip_trailing_not_sign=not args.keep_trailing_not_sign)
    counts = domain_counts(recs)
    print(f"wrote {len(recs)} images -> {args.out or paths['annotations']}; images -> {images_dir}")
    print(f"domain counts: {counts}; strip_trailing_not_sign={not args.keep_trailing_not_sign}")
    print(f"refs still containing '¬': {sum('¬' in r for x in recs for r in x['refs'])}")
    if counts != EXPECTED_DOMAIN_COUNTS:
        print(f"WARNING: domain counts differ from the official NoCaps val counts {EXPECTED_DOMAIN_COUNTS}", file=sys.stderr)


if __name__ == "__main__":
    main()
