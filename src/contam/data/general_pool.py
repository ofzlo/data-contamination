"""Fixed order of the LLaVA-665K general instruction pool, shared by every training run (D2, D7).

Every run has M examples. A run with K exposed (or SameDist) examples uses the first M - K examples
of this order, so the exposed examples replace the tail: d=0 uses exactly the first M, and a run at
larger d drops a superset of what a smaller-d run drops.

Sources (``coco``, ``gqa``, ``ocr_vqa``, ``textvqa``, ``vg``, and ``text`` for text-only examples)
are interleaved systematically, so every prefix keeps the 665K source mix up to rounding. Inside a
source, examples are ordered by sha256 of (``general_seed``, position in the 665K file); the
item at rank k of n sits at (k + jitter) / n and all sources are merged on that position.

``--exclude-images`` drops examples whose ``image`` path is listed (e.g. Open Images-derived images
the B1 audit flags as overlapping NoCaps). Text-only examples are never excluded this way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from contam.config import content_hash, file_sha256, load_config, read_json, write_json

GENERAL_FORMAT_VERSION = 1


def source_of(example: dict) -> str:
    img = example.get("image")
    return img.split("/", 1)[0] if img else "text"


def _u(seed: int, key: str) -> float:
    return int.from_bytes(hashlib.sha256(f"general|{int(seed)}|{key}".encode()).digest()[:8], "big") / 2**64


def order_pool(examples: Sequence[dict], seed: int, exclude_images: Iterable[str] = ()) -> list[dict]:
    """``[{order, src_index, id, source, image}]`` for the kept examples, in run order."""
    excl = set(exclude_images)
    by_src: dict[str, list[dict]] = defaultdict(list)
    for i, ex in enumerate(examples):
        if ex.get("image") and ex["image"] in excl:
            continue
        by_src[source_of(ex)].append({"src_index": i, "id": str(ex.get("id")), "source": source_of(ex), "image": ex.get("image")})
    keyed = []
    for items in by_src.values():
        items.sort(key=lambda it: (_u(seed, str(it["src_index"])), it["src_index"]))
        n = len(items)
        for rank, it in enumerate(items):
            keyed.append(((rank + _u(seed, f"jitter:{it['src_index']}")) / n, it["src_index"], it))
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [{"order": r, **t[2]} for r, t in enumerate(keyed)]


def write_order(rows: Sequence[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    tmp.replace(path)


def prefix_mix_deviation(rows: Sequence[dict], prefixes: Sequence[int]) -> dict[str, float]:
    """Max |count - proportional count| over sources, for each prefix length (checks the interleave)."""
    total = Counter(r["source"] for r in rows)
    n = len(rows)
    out = {}
    for m in prefixes:
        m = min(m, n)
        got = Counter(r["source"] for r in rows[:m])
        out[str(m)] = round(max(abs(got[s] - total[s] * m / n) for s in total), 3)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write the fixed order of the LLaVA-665K general pool.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--llava-mix", default=None, help="llava_v1_5_mix665k.json (default: config paths.llava_mix)")
    ap.add_argument("--out", default=None, help="default: <splits_dir>/general_order.jsonl")
    ap.add_argument("--exclude-images", default=None, help="text file of 'image' paths to drop, one per line")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    mix_path = args.llava_mix or cfg["paths"]["llava_mix"]
    out = Path(args.out or Path(cfg["paths"]["splits_dir"]) / "general_order.jsonl")
    exclude = []
    if args.exclude_images:
        with open(args.exclude_images) as f:
            exclude = f.read().split()
    seed = int(cfg["training"]["general_seed"])
    examples = read_json(mix_path)
    rows = order_pool(examples, seed, exclude)
    write_order(rows, out)
    summary = {
        "format_version": GENERAL_FORMAT_VERSION,
        "source": {"file": Path(mix_path).name, "sha256": file_sha256(mix_path), "n": len(examples)},
        "general_seed": seed,
        "n_excluded": len(examples) - len(rows),
        "excluded_list_hash": content_hash(sorted(set(exclude))),
        "n": len(rows),
        "source_mix": dict(Counter(r["source"] for r in rows).most_common()),
        "prefix_max_abs_dev": prefix_mix_deviation(rows, [1000, 10000, 130000, 300000]),
        "order_sha256": file_sha256(out),
    }
    write_json(summary, out.with_suffix(".summary.json"))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
