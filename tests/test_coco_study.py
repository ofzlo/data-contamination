"""COCO image study (contam.data.coco_study): pool, halves, exposure rows, questions. CPU only, synthetic data."""

import numpy as np
import pytest

from contam.data import coco_study as cs
from contam.data.freeze import FROZEN_GLOBS

CATS = [{"id": 1, "name": "person", "supercategory": "person"}, {"id": 2, "name": "dog", "supercategory": "animal"},
        {"id": 3, "name": "car", "supercategory": "vehicle"}, {"id": 4, "name": "dining table", "supercategory": "furniture"},
        {"id": 5, "name": "cat", "supercategory": "animal"}, {"id": 6, "name": "bus", "supercategory": "vehicle"}]
PROMPTS = {"caption": "Describe the image in detail.", "caption_text": "Describe the image in detail."}


def make_data(n=40):
    images, anns, caps, qs, answers = [], [], [], [], {}
    for i in range(1, n + 1):
        images.append({"id": i, "file_name": f"{i:012d}.jpg", "flickr_url": f"http://farm1.staticflickr.com/1/{1000 + i}_abc.jpg"})
        if i != 3:  # image 3 has no object
            anns.append({"id": 10 * i, "image_id": i, "category_id": 1 + i % 3, "area": 100.0 + i, "iscrowd": 0})
            anns.append({"id": 10 * i + 1, "image_id": i, "category_id": 4, "area": 10.0, "iscrowd": 0})
        for k in range(5):
            caps.append({"id": 100 * i + k, "image_id": i, "caption": f" A dog near a table number {k} "})
        if i != 4:  # image 4 has no VQAv2 question
            for k in range(2):
                qid = 1000 * i + k
                qs.append({"question_id": qid, "image_id": i, "question": f"What is item {k}?"})
                answers[qid] = {"question_id": qid, "answer_type": "other" if k == 0 else "yes/no",
                                "multiple_choice_answer": "dogs" if k == 0 else "yes", "answers": [{"answer": "dog"}] * 10}
    return {"images": images, "annotations": anns, "categories": CATS}, {"annotations": caps}, qs, answers


def test_seen_ids_reasons():
    inst, _, _, _ = make_data(10)
    mix = [{"image": "vg/VG_100K/77.jpg"}, {"image": "gqa/images/78.jpg"}, {"image": "coco/train2017/000000000005.jpg"}, {"conversations": []}]
    vg = [{"image_id": 77, "coco_id": 1, "flickr_id": None}, {"image_id": 78, "coco_id": None, "flickr_id": 1002}, {"image_id": 79, "coco_id": 6}]
    lcs = [{"url": "http://farm2.staticflickr.com/9/1007_ff.jpg"}]
    train = [{"id": 5, "flickr_url": "http://x.staticflickr.com/1/1008_aa.jpg"}]
    seen = cs.seen_ids(inst["images"], mix, vg, lcs, train)
    assert seen == {1: ["665k_vg_gqa:coco_id"], 2: ["665k_vg_gqa:flickr_id"], 5: ["665k_coco:coco_id"], 7: ["lcs558k:flickr_id"], 8: ["665k_coco:flickr_id"]}
    assert 6 not in seen  # in VG, but no 665K example uses it


def test_pool_rules_halves_and_determinism():
    inst, caps, qs, _ = make_data()
    seen = {1: ["x"], 2: ["y"]}
    pool = cs.build_pool(inst, caps, qs, seen, split_seed=7, ref_seed=8)
    cs.validate_pool(pool)
    ids = {r["image_id"] for r in pool["images"]}
    assert not ids & {1, 2, 3, 4}  # seen, no object, no question
    assert pool["counts"]["dropped"] == {"seen_by_llava": 2, "no_object": 1, "no_vqav2_val_question": 1}
    halves = [r["half"] for r in pool["images"]]
    assert abs(halves.count("A") - halves.count("B")) <= 1
    for s in {r["stratum"] for r in pool["images"]}:
        h = [r["half"] for r in pool["images"] if r["stratum"] == s]
        assert abs(h.count("A") - h.count("B")) <= 1  # stratified
    assert all(0 <= r["train_caption_idx"] < 5 for r in pool["images"])
    assert cs.build_pool(inst, caps, qs, seen, 7, 8) == pool
    assert cs.build_pool(inst, caps, qs, seen, 9, 8)["images"] != pool["images"]
    bad = dict(pool, images=pool["images"][:-1])
    with pytest.raises(ValueError):
        cs.validate_pool(bad)


def test_exposure_levels():
    inst, caps_raw, qs, answers = make_data(12)
    pool = cs.build_pool(inst, caps_raw, qs, {}, 1, 2)
    caps = cs.captions_by_image(caps_raw)
    by_img = {}
    for q in qs:
        by_img.setdefault(q["image_id"], []).append(dict(q, multiple_choice_answer=answers[q["question_id"]]["multiple_choice_answer"]))
    n_a = sum(r["half"] == "A" for r in pool["images"])
    for level in cs.LEVELS:
        ex = cs.build_exposure(level, "A", pool, caps, by_img, 16, PROMPTS)
        assert ex["meta"]["K"] == 16 * n_a and len(cs.expand(ex)) == 16 * n_a
        r = ex["rows"][0]
        first = r["conversations"][0]["value"]
        if level == "Cap":
            assert "image" not in r and not first.startswith("<image>")
        else:
            assert r["image"].startswith(cs.IMAGE_PREFIX + "/") and first.startswith("<image>\n")
        if level == "L3":
            assert len(r["conversations"]) == 4 and first.endswith(cs.VQA_SUFFIX) and r["conversations"][1]["value"] == "dogs"
            assert not r["conversations"][2]["value"].startswith("<image>")
        if level == "L1":
            assert r["conversations"][1]["value"] == cs.UNINFORMATIVE
        if level in ("L2", "Cap"):
            row = next(p for p in pool["images"] if p["image_id"] == r["coco_image_id"])
            assert r["conversations"][1]["value"] == caps[row["image_id"]][row["train_caption_idx"]] == cs.train_caption(row, caps)
    assert cs.run_id("L3", "B", 16, 0) == "C_L3_B_d16_s0"


def test_answer_mentioned():
    assert cs.answer_mentioned("dog", "Two dogs on a couch")
    assert cs.answer_mentioned("dogs", "A dog on a couch")
    assert cs.answer_mentioned("buses", "a city bus") and cs.answer_mentioned("bus", "two buses")
    assert not cs.answer_mentioned("glass", "a gas station") and cs.answer_mentioned("houses", "a white house")
    assert cs.answer_mentioned("fire hydrant", "a red fire hydrant.")
    assert not cs.answer_mentioned("hydrant", "a red fire truck")
    assert not cs.answer_mentioned("", "anything")


def test_vqav2_cells():
    inst, caps_raw, qs, answers = make_data(10)
    pool = cs.build_pool(inst, caps_raw, qs, {}, 1, 2)
    by_img = {}
    for q in qs:
        by_img.setdefault(q["image_id"], []).append(q)
    out = cs.vqav2_questions(pool, cs.captions_by_image(caps_raw), by_img, answers)
    assert {q["cell"] for q in out} == {"mentioned", "yes/no"}  # "dogs" vs "A dog near a table ..."
    assert all(len(q["answers"]) == 10 and q["half"] in cs.HALVES for q in out)


def test_pope_negatives_absent_and_unmentioned():
    inst, caps_raw, qs, _ = make_data(20)
    pool = cs.build_pool(inst, caps_raw, qs, {}, 1, 2)
    caps = cs.captions_by_image(caps_raw)
    tagged = {c: [(t, "NOUN") for t in cs.norm_tokens(c)] for v in caps.values() for c in v}
    m = cs.CocoMatcher([c["name"] for c in CATS])
    assert m([("two", "NUM"), ("dogs", "NOUN"), ("and", "CCONJ"), ("a", "DET"), ("table", "NOUN")]) == {"dog", "dining table"}
    assert m([("people", "NOUN")]) == {"person"}
    assert m([("dog", "VERB")]) == set()
    out = cs.pope_questions(pool, inst, caps, m, tagged, seed=3)
    present = {}
    for a in inst["annotations"]:
        present.setdefault(a["image_id"], set()).add(next(c["name"] for c in CATS if c["id"] == a["category_id"]))
    for q in out:
        if q["answer"] == "no":
            assert q["object"] not in present[q["image_id"]]
            assert q["object"] not in {"dog", "dining table"}  # every caption names a dog and a table
        else:
            assert q["object"] in present[q["image_id"]]
    pos = [q for q in out if q["answer"] == "yes"]
    assert {q["cell"] for q in pos if q["object"] == "dining table"} == {"pos/mentioned"}
    assert len({q["qid"] for q in out}) == len(out)
    assert cs.pope_questions(pool, inst, caps, m, tagged, seed=3) == out
    rows = cs.review_rows(out, caps, pool, ["pos/mentioned"], n_per_cell=5)
    assert len(rows) == 5 and all(r["cell"] == "pos/mentioned" for r in rows)


def test_pool_is_a_frozen_pattern():
    assert "coco_*.json" in FROZEN_GLOBS
