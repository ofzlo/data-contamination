import pytest

from contam.data.exposure import CONDITIONS, build_exposure, derangement, expand, exposure_ids
from contam.data.samedist import make_samedist
from contam.data.splits import make_split
from tests.conftest import make_annotations
from tests.test_manifests import make_coco

PROMPTS = {"caption": "Describe the image.", "caption_text": "Write a caption for an image."}


@pytest.fixture(scope="module")
def nocaps_like():
    ann = make_annotations(417, 2670, 1413, n_refs=10)
    return ann, make_split(ann, 20261019, 20261020)


def test_stored_and_rebuilt_exposure_sets_nest(nocaps_like):
    ann, split = nocaps_like
    sets = {n: set(exposure_ids(split, ann, n)) for n in (141, 563, 1000, 2250)}
    assert {n: len(s) for n, s in sets.items()} == {141: 141, 563: 563, 1000: 1000, 2250: 2250}
    assert sets[141] < sets[563] < sets[1000] < sets[2250]
    assert sets[2250] == set(split["seen"])
    dom = {a["image_id"]: a["domain"] for a in ann}
    got = {d: sum(dom[i] == d for i in sets[1000]) for d in ("in-domain", "near-domain", "out-domain")}
    assert got == {"in-domain": 93, "near-domain": 593, "out-domain": 314}  # 1000/4500 of 417/2670/1413, largest remainder
    with pytest.raises(ValueError):
        exposure_ids(split, ann, 2251)


def test_derangement():
    ids = list(range(50))
    p = derangement(ids, 7)
    assert sorted(p) == ids and sorted(p.values()) == ids
    assert all(p[i] != i for i in ids)
    assert p == derangement(list(reversed(ids)), 7) and p != derangement(ids, 8)


@pytest.fixture
def small():
    ann = make_annotations(30, 120, 50, seed=3)
    split = make_split(ann, 1, 2)
    return ann, split, make_samedist(split, ann, make_coco(), seed=1)


def test_conditions_share_images_and_targets(small):
    ann, split, sd = small
    n = len(split["seen"])
    refs = {a["image_id"]: a["refs"][split["train_ref_idx"][str(a["image_id"])]] for a in ann}
    base = {i: f"base caption {i}" for i in split["seen"]}
    out = {c: build_exposure(c, split, ann, n, 4, PROMPTS, shuffle_seed=5, samedist=sd, base_caps=base) for c in CONDITIONS}
    for c, ex in out.items():
        assert ex["meta"]["K"] == 4 * n and len(expand(ex)) == 4 * n
        assert [r["nocaps_image_id"] for r in ex["rows"]] == sorted(split["seen"])
    ans = lambda c: [r["conversations"][1]["value"] for r in out[c]["rows"]]
    assert ans("paired") == ans("caption") == [refs[i] for i in sorted(split["seen"])]
    assert sorted(ans("shuffled")) == sorted(ans("paired")) and ans("shuffled") != ans("paired")
    assert ans("image") == [base[i] for i in sorted(split["seen"])]
    assert ans("samedist") == [r["caption"] for r in sd["rows"]]
    assert "image" not in out["caption"]["rows"][0] and out["caption"]["rows"][0]["conversations"][0]["value"] == PROMPTS["caption_text"]
    assert out["paired"]["rows"][0]["conversations"][0]["value"] == "<image>\n" + PROMPTS["caption"]
    assert out["samedist"]["rows"][0]["image"].startswith("coco/train2017/")


def test_condition_inputs_are_required(small):
    ann, split, sd = small
    n = len(split["seen"])
    with pytest.raises(ValueError):
        build_exposure("image", split, ann, n, 1, PROMPTS)
    with pytest.raises(ValueError):
        build_exposure("image", split, ann, n, 1, PROMPTS, base_caps={})
    with pytest.raises(ValueError):
        build_exposure("samedist", split, ann, n, 1, PROMPTS, samedist={**sd, "split": {"content_hash": "x"}})
    with pytest.raises(ValueError):
        build_exposure("paired", split, ann, n, 0, PROMPTS)
