"""Deterministic exposure splits of NoCaps validation (D7).

Seeds are kept separate (D7):

* ``split_seed`` decides *which images are exposed* (seen) and the nested breadth sets.
* ``ref_seed`` decides *which of the 10 references is the training reference* of each image.
* Training seeds are not part of the split; they live with the training config.

Split A (explore) and split B (confirm, G3) use different ``split_seed``/``ref_seed`` values.

Construction
------------
For every domain stratum (in / near / out) we draw ONE permutation of that stratum's image ids
(sorted by image id first, so the result does not depend on input order). The exposure set at
fraction ``f`` is the union over strata of a prefix of that permutation. Per-stratum counts come
from ``allocate_stratified`` (largest-remainder rounding, so the total is exactly round(f * N),
e.g. 2,250 at 50%). Taking prefixes of one permutation makes the sets nested by construction
(ground rule 3: smaller exposure sets are subsets of larger ones):

    exposed(0.5) ⊃ exposed(0.125) ⊃ exposed(0.03125)

The main split uses ``exposed_fraction`` (50%): ``seen = exposed(0.5)``, ``unseen`` = the rest.
For C2, the smallest set (3.125%) is the shared seen set on which seen metrics are compared
across breadth arms; the unseen group stays the complement of the 50% set for every arm, so it
contains images that are never exposed in any C2 arm.

Every image, seen or unseen, gets a ``train_ref_idx`` drawn with ``ref_seed``. Unseen images are
never trained on, but assigning them a "training" reference too keeps the reference configuration
identical across groups: the held-out set is always 9 references and the ``train`` configuration
is always 1 reference.
"""

from __future__ import annotations

import argparse
import math
import zlib
from collections import Counter
from typing import Sequence

import numpy as np

from contam.config import content_hash, load_config, write_json

DOMAINS = ("in", "near", "out")
SPLIT_FORMAT_VERSION = 1


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def allocate_stratified(sizes: dict[str, int], fraction: float) -> dict[str, int]:
    """Per-stratum counts summing to round_half_up(fraction * N) (largest-remainder method).

    Each stratum gets floor(fraction * n_s); the remaining units go to the strata with the largest
    fractional remainders (ties broken in DOMAINS order). E.g. 50% of (417, 2670, 1413) ->
    (209, 1335, 706) = 2,250 exactly.
    """
    total = _round_half_up(fraction * sum(sizes.values()))
    base = {d: int(math.floor(fraction * sizes[d])) for d in DOMAINS}
    rem = sorted(DOMAINS, key=lambda d: (-(fraction * sizes[d] - base[d]), DOMAINS.index(d)))
    for d in rem[: max(0, total - sum(base.values()))]:
        base[d] += 1
    return base


def _stratum_rng(seed: int, stratum: str) -> np.random.Generator:
    # Per-stratum stream: adding or reordering strata does not change other strata's permutation.
    return np.random.default_rng([int(seed), zlib.crc32(stratum.encode())])


def normalize_domain(domain: str) -> str:
    d = domain.lower().replace("_", "-")
    for k in DOMAINS:
        if d.startswith(k):
            return k
    raise ValueError(f"unknown NoCaps domain {domain!r}")


def annotations_hash(annotations: Sequence[dict]) -> str:
    """Hash of the (image_id, domain, refs) content of ``annotations``, recorded in every split."""
    return content_hash([[int(a["image_id"]), normalize_domain(a["domain"]), list(a["refs"])] for a in sorted(annotations, key=lambda a: int(a["image_id"]))])


def check_annotations(split: dict, annotations: Sequence[dict]) -> None:
    """Raise unless ``annotations`` are the ones ``split`` was built from (same refs, so the same
    training-reference assignment and held-out sets)."""
    h = annotations_hash(annotations)
    if split.get("annotations_hash") != h:
        raise ValueError(f"annotations ({h[:12]}) differ from those split {split.get('name')} was built from ({str(split.get('annotations_hash'))[:12]}); re-run make_splits or use the matching annotations.json")


def make_split(
    annotations: Sequence[dict],
    split_seed: int,
    ref_seed: int,
    exposed_fraction: float = 0.5,
    nested_fractions: Sequence[float] = (0.5, 0.125, 0.03125),
    name: str = "A",
) -> dict:
    """Build a split dict from NoCaps annotations.

    Args:
        annotations: records with ``image_id`` (int), ``domain`` and ``refs`` (list of str).
        split_seed: seed for exposure (seen/unseen and nested breadth sets).
        ref_seed: seed for the training-reference assignment.
        exposed_fraction: fraction exposed in the main split; must be one of ``nested_fractions``.
        nested_fractions: breadth fractions; smaller sets are prefixes (hence subsets) of larger ones.
        name: split label recorded in the metadata ("A" explore, "B" confirm).

    Returns:
        A JSON-serializable dict; see ``validate_split`` for the invariants it satisfies.
    """
    fractions = sorted({float(f) for f in nested_fractions}, reverse=True)
    if exposed_fraction not in fractions:
        fractions = sorted(set(fractions) | {float(exposed_fraction)}, reverse=True)
    if not all(0.0 < f <= 1.0 for f in fractions):
        raise ValueError("fractions must be in (0, 1]")

    by_id = {}
    for a in annotations:
        iid = int(a["image_id"])
        if iid in by_id:
            raise ValueError(f"duplicate image_id {iid}")
        by_id[iid] = a
    ids_sorted = sorted(by_id)

    strata: dict[str, list[int]] = {d: [] for d in DOMAINS}
    for iid in ids_sorted:
        strata[normalize_domain(by_id[iid]["domain"])].append(iid)

    exposure_sets: dict[str, list[int]] = {}
    order: dict[str, list[int]] = {}
    for d, ids in strata.items():
        order[d] = [ids[i] for i in _stratum_rng(split_seed, d).permutation(len(ids))]
    sizes = {d: len(v) for d, v in strata.items()}
    prev: dict[str, int] = {d: 0 for d in DOMAINS}
    for f in sorted(fractions):  # smallest first so per-stratum counts are non-decreasing in f
        alloc = allocate_stratified(sizes, f)
        alloc = {d: max(alloc[d], prev[d]) for d in DOMAINS}
        prev = alloc
        exposure_sets[_fkey(f)] = sorted(i for d in DOMAINS for i in order[d][: alloc[d]])
    exposure_sets = {_fkey(f): exposure_sets[_fkey(f)] for f in fractions}

    seen = exposure_sets[_fkey(exposed_fraction)]
    seen_set = set(seen)
    unseen = [i for i in ids_sorted if i not in seen_set]

    ref_rng = np.random.default_rng(int(ref_seed))
    train_ref_idx = {}
    for iid in ids_sorted:
        n_refs = len(by_id[iid]["refs"])
        if n_refs < 2:
            raise ValueError(f"image {iid} has {n_refs} refs; need >= 2")
        train_ref_idx[str(iid)] = int(ref_rng.integers(0, n_refs))

    domain_of = {iid: normalize_domain(by_id[iid]["domain"]) for iid in ids_sorted}
    counts = {
        "total": dict(Counter(domain_of.values())),
        "seen": dict(Counter(domain_of[i] for i in seen)),
        "unseen": dict(Counter(domain_of[i] for i in unseen)),
        "exposure_sets": {k: dict(Counter(domain_of[i] for i in v)) for k, v in exposure_sets.items()},
    }
    body = {
        "format_version": SPLIT_FORMAT_VERSION,
        "name": name,
        "split_seed": int(split_seed),
        "ref_seed": int(ref_seed),
        "exposed_fraction": float(exposed_fraction),
        "nested_fractions": fractions,
        "shared_seen_fraction": fractions[-1],
        "annotations_hash": annotations_hash(annotations),
        "counts": counts,
        "seen": seen,
        "unseen": unseen,
        "exposure_sets": exposure_sets,
        "train_ref_idx": train_ref_idx,
    }
    body["content_hash"] = content_hash(body)
    return body


def _fkey(f: float) -> str:
    return repr(float(f))


def validate_split(split: dict) -> None:
    """Raise AssertionError if any split invariant is violated."""
    seen, unseen = set(split["seen"]), set(split["unseen"])
    assert not seen & unseen, "seen and unseen overlap"
    all_ids = {int(k) for k in split["train_ref_idx"]}
    assert seen | unseen == all_ids, "seen ∪ unseen must cover every image"
    sets = split["exposure_sets"]
    keys = sorted(sets, key=float, reverse=True)
    for big, small in zip(keys, keys[1:]):
        assert set(sets[small]) <= set(sets[big]), f"exposure set {small} not nested in {big}"
    assert set(sets[_fkey(split["exposed_fraction"])]) == seen
    body = {k: v for k, v in split.items() if k != "content_hash"}
    assert content_hash(body) == split["content_hash"], "content hash mismatch"


def group_ids(split: dict, group: str) -> list[int]:
    """Image ids of ``seen``, ``unseen``, ``shared_seen`` (smallest nested set) or an exposure fraction."""
    if group in ("seen", "unseen"):
        return list(split[group])
    if group == "shared_seen":
        return list(split["exposure_sets"][_fkey(split["shared_seen_fraction"])])
    return list(split["exposure_sets"][_fkey(float(group))])


def make_class_disjoint_split(annotations: Sequence[dict], split_seed: int, ref_seed: int) -> dict:
    """Class-disjoint split for C7 (vocabulary transfer to unseen classes). NOT IMPLEMENTED.

    Intended design: assign Open Images classes (from the NoCaps out/near-domain object labels) to
    seen/unseen so that no unseen image contains a class present in any seen image, stratified by
    domain where possible, with the same ``train_ref_idx`` scheme as ``make_split``. Needs the
    NoCaps object-class annotations, which the lmms-lab parquet mirror does not carry.
    """
    raise NotImplementedError("class-disjoint split (C7) is not implemented yet; needs NoCaps class labels")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Make deterministic NoCaps exposure splits.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--annotations", default=None, help="annotations.json (default: config paths.annotations)")
    ap.add_argument("--split", action="append", default=None, help="split name(s) from config.splits (default: A and B)")
    ap.add_argument("--split-seed", type=int, default=None, help="override split_seed (single split only)")
    ap.add_argument("--ref-seed", type=int, default=None, help="override ref_seed (single split only)")
    ap.add_argument("--out-dir", default=None, help="output directory (default: config paths.splits_dir)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    from contam.data.nocaps import load_annotations

    ann = load_annotations(args.annotations or cfg["paths"]["annotations"])
    names = args.split or ["A", "B"]
    if (args.split_seed is not None or args.ref_seed is not None) and len(names) != 1:
        ap.error("seed overrides require exactly one --split")
    out_dir = args.out_dir or cfg["paths"]["splits_dir"]
    scfg = cfg["splits"]
    built = {}
    for name in names:
        seeds = scfg.get(name, {})
        split_seed = args.split_seed if args.split_seed is not None else seeds["split_seed"]
        ref_seed = args.ref_seed if args.ref_seed is not None else seeds["ref_seed"]
        split = make_split(ann, split_seed, ref_seed, scfg["exposed_fraction"], scfg["nested_fractions"], name=name)
        validate_split(split)
        path = f"{out_dir}/split_{name}.json"
        write_json(split, path, indent=None)
        built[name] = split
        c = split["counts"]
        print(f"[split {name}] split_seed={split_seed} ref_seed={ref_seed} hash={split['content_hash'][:12]} -> {path}")
        print(f"  total={c['total']} seen={c['seen']} unseen={c['unseen']}")
        for k, v in c["exposure_sets"].items():
            print(f"  exposure {float(k):.5f}: n={sum(v.values())} {v}")
    if len(built) > 1:
        names_l = list(built)
        a, b = built[names_l[0]], built[names_l[1]]
        ov = len(set(a["seen"]) & set(b["seen"])) / max(1, len(a["seen"]))
        same_ref = np.mean([a["train_ref_idx"][k] == b["train_ref_idx"][k] for k in a["train_ref_idx"]])
        print(f"  overlap seen({names_l[0]}) ∩ seen({names_l[1]}) / |seen| = {ov:.3f}; same train ref = {same_ref:.3f}")


if __name__ == "__main__":
    main()
