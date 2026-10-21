import random

import pytest

from contam.config import load_config, write_json
from contam.data import freeze
from contam.data.general_pool import order_pool, prefix_mix_deviation, source_of
from contam.data.samedist import make_samedist, n_words, samedist_rows, validate_samedist
from contam.data.splits import make_split


def make_coco(n_images=3000, caps_per_image=5, seed=0):
    rng = random.Random(seed)
    vocab = "a man woman dog cat sits on the grass near street with red car bus table plate".split()
    caps = []
    for iid in range(n_images):
        for j in range(caps_per_image):
            caps.append({"image_id": iid, "ann_id": iid * 10 + j, "file_name": f"{iid:012d}.jpg", "caption": " ".join(rng.choice(vocab) for _ in range(rng.randint(5, 14)))})
    return caps


@pytest.fixture
def split_and_coco(annotations):
    return make_split(annotations, 5, 6), make_coco()


def test_count_length_and_unique_images(annotations, split_and_coco):
    split, coco = split_and_coco
    sd = make_samedist(split, annotations, coco, seed=1)
    validate_samedist(sd, split)
    refs = {a["image_id"]: a["refs"] for a in annotations}
    assert len(sd["rows"]) == len(split["seen"])
    for r in sd["rows"]:
        iid = r["nocaps_image_id"]
        assert r["target_words"] == n_words(refs[iid][split["train_ref_idx"][str(iid)]])
        assert r["caption_words"] == r["target_words"]  # plenty of COCO captions of every length here
    assert len({r["coco_image_id"] for r in sd["rows"]}) == len(sd["rows"])


def test_nested_and_smaller_sets_independent_of_larger(annotations, split_and_coco):
    split, coco = split_and_coco
    sd = make_samedist(split, annotations, coco, seed=1)
    small, mid, big = (samedist_rows(sd, split, k) for k in sorted(split["exposure_sets"], key=float))
    ids = lambda rows: {r["nocaps_image_id"] for r in rows}
    assert ids(small) < ids(mid) < ids(big)
    assert ids(small) == set(split["exposure_sets"][min(split["exposure_sets"], key=float)])
    k_small = min(split["exposure_sets"], key=float)
    for r_small in small:
        assert r_small["level"] == float(k_small)
    sd2 = make_samedist(split, annotations, list(reversed(coco)), seed=1)
    assert sd2["rows"] == sd["rows"]  # input order does not matter


def test_seed_changes_choice_not_lengths(annotations, split_and_coco):
    split, coco = split_and_coco
    a = make_samedist(split, annotations, coco, seed=1)
    b = make_samedist(split, annotations, coco, seed=2)
    assert [r["caption_words"] for r in a["rows"]] == [r["caption_words"] for r in b["rows"]]
    assert [r["coco_ann_id"] for r in a["rows"]] != [r["coco_ann_id"] for r in b["rows"]]


def test_nearest_length_when_bucket_runs_out(annotations):
    split = make_split(annotations, 5, 6)
    coco = [{"image_id": i, "ann_id": i, "file_name": f"{i}.jpg", "caption": " ".join(["w"] * (3 + i % 2))} for i in range(len(split["seen"]))]
    sd = make_samedist(split, annotations, coco, seed=1)
    validate_samedist(sd, split)
    big = sd["stats"][max(sd["stats"], key=float)]
    assert big["exact_length"] == 0 and big["n"] == len(split["seen"])


def test_validate_rejects_other_split_and_tampering(annotations, split_and_coco):
    split, coco = split_and_coco
    sd = make_samedist(split, annotations, coco, seed=1)
    with pytest.raises(AssertionError):
        validate_samedist(sd, make_split(annotations, 9, 6))
    sd["rows"][0]["caption"] += " extra"
    with pytest.raises(AssertionError):
        validate_samedist(sd, split)


def test_pool_exhausted(annotations):
    split = make_split(annotations, 5, 6)
    with pytest.raises(RuntimeError):
        make_samedist(split, annotations, make_coco(n_images=10), seed=1)


# ---------------------------------------------------------------- general pool


def make_mix(n=6000, seed=0):
    rng = random.Random(seed)
    srcs = ["coco/train2017", "gqa/images", "ocr_vqa/images", "textvqa/train_images", "vg/VG_100K", None]
    w = [0.55, 0.11, 0.12, 0.03, 0.12, 0.07]
    out = []
    for i in range(n):
        s = rng.choices(srcs, w)[0]
        ex = {"id": f"x{i}", "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}
        if s:
            ex["image"] = f"{s}/{i}.jpg"
        out.append(ex)
    return out


def test_general_order_deterministic_and_mix_preserving():
    mix = make_mix()
    a = order_pool(mix, seed=3)
    assert a == order_pool(mix, seed=3)
    assert a != order_pool(mix, seed=4)
    assert sorted(r["src_index"] for r in a) == list(range(len(mix)))
    dev = prefix_mix_deviation(a, [100, 1000, 3000])
    assert all(v <= 1.5 for v in dev.values())
    assert {r["source"] for r in a} == {source_of(ex) for ex in mix}


def test_general_order_exclusions():
    mix = make_mix()
    drop = [ex["image"] for ex in mix[:50] if ex.get("image")]
    rows = order_pool(mix, seed=3, exclude_images=drop)
    assert len(rows) == len(mix) - len(drop)
    assert not {r["image"] for r in rows} & set(drop)
    assert [r["order"] for r in rows] == list(range(len(rows)))


# ---------------------------------------------------------------- freeze


def test_freeze_lock_check_and_add(tmp_path, annotations):
    cfg = load_config()
    cfg["paths"]["splits_dir"] = tmp_path / "splits"
    lock = tmp_path / "lock.json"
    split = make_split(annotations, 5, 6, name="A")
    write_json(split, cfg["paths"]["splits_dir"] / "split_A.json", indent=None)
    freeze.write_lock(cfg, lock)
    freeze.verify(cfg, lock)
    with pytest.raises(SystemExit):
        freeze.write_lock(cfg, lock)  # never overwritten

    # a new manifest is an error until added; adding never rewrites existing entries
    (cfg["paths"]["splits_dir"] / "general_order.jsonl").write_text('{"order": 0}\n')
    with pytest.raises(RuntimeError, match="not in lock"):
        freeze.verify(cfg, lock)
    with pytest.raises(RuntimeError, match="not frozen"):
        freeze.verify(cfg, lock, allow_unlocked=True, required=["general_order.jsonl"])
    freeze.write_lock(cfg, lock, add=True)
    freeze.verify(cfg, lock, required=["general_order.jsonl"])

    # edited file or changed seeds -> mismatch
    (cfg["paths"]["splits_dir"] / "general_order.jsonl").write_text('{"order": 1}\n')
    with pytest.raises(RuntimeError, match="changed general_order"):
        freeze.verify(cfg, lock)
    (cfg["paths"]["splits_dir"] / "general_order.jsonl").write_text('{"order": 0}\n')
    cfg["training"]["seeds"] = [1, 2, 3, 4, 5]
    with pytest.raises(RuntimeError, match="seed config"):
        freeze.verify(cfg, lock)
