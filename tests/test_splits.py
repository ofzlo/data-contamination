from collections import Counter

import pytest

from contam.data.splits import allocate_stratified, group_ids, make_class_disjoint_split, make_split, validate_split
from tests.conftest import make_annotations


def test_deterministic_and_order_invariant(annotations):
    a = make_split(annotations, 1, 2)
    b = make_split(list(reversed(annotations)), 1, 2)
    assert a == b
    validate_split(a)


def test_seeds_are_separate(annotations):
    base = make_split(annotations, 1, 2)
    other_ref = make_split(annotations, 1, 3)
    other_split = make_split(annotations, 9, 2)
    assert other_ref["seen"] == base["seen"] and other_ref["exposure_sets"] == base["exposure_sets"]
    assert other_ref["train_ref_idx"] != base["train_ref_idx"]
    assert other_split["seen"] != base["seen"]
    assert other_split["train_ref_idx"] == base["train_ref_idx"]
    assert base["content_hash"] != other_ref["content_hash"]


def test_disjoint_cover_and_stratified(annotations):
    s = make_split(annotations, 5, 6)
    seen, unseen = set(s["seen"]), set(s["unseen"])
    assert not seen & unseen
    assert seen | unseen == {a["image_id"] for a in annotations}
    dom = {a["image_id"]: a["domain"] for a in annotations}
    tot = Counter(dom.values())
    got = Counter(dom[i] for i in seen)
    for d, n in tot.items():
        assert abs(got[d] - n / 2) <= 1  # stratified 50%
    assert len(seen) == int(len(annotations) / 2 + 0.5)  # 449 images -> 225 exposed


def test_nested_breadth_sets(annotations):
    s = make_split(annotations, 5, 6)
    e = s["exposure_sets"]
    big, mid, small = (set(e[k]) for k in sorted(e, key=float, reverse=True))
    assert small < mid < big
    assert big == set(s["seen"])
    assert set(group_ids(s, "shared_seen")) == small


def test_every_image_has_train_ref(annotations):
    s = make_split(annotations, 5, 6)
    assert len(s["train_ref_idx"]) == len(annotations)
    assert all(0 <= v < 10 for v in s["train_ref_idx"].values())
    # roughly uniform over the 10 refs
    c = Counter(s["train_ref_idx"].values())
    assert len(c) == 10


def test_allocation_matches_nocaps_total():
    sizes = {"in": 417, "near": 2670, "out": 1413}
    a = allocate_stratified(sizes, 0.5)
    assert sum(a.values()) == 2250
    assert all(abs(a[d] - sizes[d] / 2) < 1 for d in sizes)
    assert sum(allocate_stratified(sizes, 0.125).values()) == 563  # round_half_up(562.5)
    assert sum(allocate_stratified(sizes, 0.03125).values()) == 141  # round_half_up(140.625)


def test_real_nocaps_sizes_nested():
    ann = make_annotations(417, 2670, 1413, n_refs=10)
    s = make_split(ann, 20261019, 20261020)
    validate_split(s)
    n = {k: len(v) for k, v in s["exposure_sets"].items()}
    assert n[repr(0.5)] == 2250
    assert 139 <= n[repr(0.03125)] <= 142


def test_validate_detects_tampering(annotations):
    s = make_split(annotations, 1, 2)
    s["seen"] = s["seen"][:-1]
    with pytest.raises(AssertionError):
        validate_split(s)


def test_class_disjoint_stub(annotations):
    with pytest.raises(NotImplementedError):
        make_class_disjoint_split(annotations, 1, 2)
