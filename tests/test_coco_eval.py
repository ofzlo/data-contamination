"""COCO image-study evaluation (contam.eval.coco_eval): VQA accuracy, scoring, POPE F1, Leak. CPU only."""

import pytest

from contam.config import read_jsonl
from contam.eval import coco_eval as ce


def test_normalize_answer():
    assert ce.normalize_answer("Two") == "2"
    assert ce.normalize_answer("the dog.") == "dog"
    assert ce.normalize_answer("dont") == "don't"
    assert ce.normalize_answer(" yes! ") == "yes"
    assert ce.normalize_answer("1,000") == "1000"


def test_vqa_accuracy_official_leave_one_out():
    answers = ["2"] * 3 + ["3"] * 7
    assert ce.vqa_accuracy("two", answers) == pytest.approx(0.9)  # (3 x 2/3 + 7 x 1) / 10
    assert ce.vqa_accuracy("3", answers) == pytest.approx(1.0)
    assert ce.vqa_accuracy("4", answers) == 0.0
    assert ce.vqa_accuracy("Yes.", ["yes"] * 10) == 1.0


def _questions(n_img=6, per=3):
    qs = []
    for i in range(n_img):
        half = "A" if i < n_img // 2 else "B"
        for k in range(per):
            qs.append({"qid": f"q{i}_{k}", "image_id": i, "half": half, "cell": "mentioned" if k == 0 else "unmentioned",
                       "question": f"What is {k}?", "answers": [f"a{k}"] * 10, "answer": "yes" if k % 2 == 0 else "no",
                       "neg_type": None if k % 2 == 0 else "random"})
    return qs


def test_leak_sanity():
    qs = _questions()
    same = {q["qid"]: (1.0 if q["image_id"] % 2 else 0.0) for q in qs}
    r = ce.leak(same, same, qs, None, n_boot=300)
    assert r["estimate"] == 0.0 and r["ci_lo"] == 0.0 and r["ci_hi"] == 0.0  # M_A = M_B -> Leak 0 exactly
    a = {q["qid"]: 1.0 if q["half"] == "A" else 0.5 for q in qs}  # M_A perfect on A, 0.5 on B
    b = {q["qid"]: 1.0 if q["half"] == "B" else 0.5 for q in qs}
    r = ce.leak(a, b, qs, None, n_boot=300)
    assert r["estimate"] == pytest.approx(0.5) and r["accuracy"]["M_A_on_A"] == 1.0 and r["accuracy"]["M_A_on_B"] == 0.5
    r = ce.leak(a, b, qs, "mentioned", n_boot=300)
    assert r["n_questions"] == {"A": 3, "B": 3} and r["estimate"] == pytest.approx(0.5)
    assert ce.leak(a, b, [q for q in qs if q["half"] == "A"], None, n_boot=10) is None  # one half missing


class FakeVLM:
    model_id, adapter = "fake", "ad"

    def __init__(self):
        self.calls = []

    def generate_answers(self, images, prompts, max_new_tokens=10):
        self.calls.append(list(prompts))
        return [p.split("?")[0].split()[-1] for p in prompts]  # "What is 0?" -> "0"


def test_score_vqa_resume_and_fixed_batches(tmp_path):
    qs = ce.sort_questions(_questions())
    for q in qs:
        q["answers"] = [q["question"].split()[-1].rstrip("?")] * 10
    out = tmp_path / "s.vqa.jsonl"
    vlm = FakeVLM()
    assert ce.score_vqa(vlm, lambda i: i, qs[:5], out, "ck", "SHA", batch_size=4, progress=False) == 5
    assert ce.score_vqa(vlm, lambda i: i, qs, out, "ck", "SHA", batch_size=4, progress=False) == len(qs) - 4  # chunk [4:8] rescored whole
    rows = list(read_jsonl(out))
    assert sorted(r["qid"] for r in rows) == sorted(q["qid"] for q in qs) and all(r["acc"] == 1.0 for r in rows)
    want = [[q["question"] + ce.VQA_SUFFIX for q in c] for c in ce._chunks(qs, 4)]
    assert vlm.calls[-len(want) + 1:] == want[1:]
    acc, prov = ce.load_vqa(out, qs, "SHA")
    assert len(acc) == len(qs) and prov["max_new_tokens"] == 10 and prov["batch_order"] == "length_sorted"
    order = ce.vqa_order(qs, lambda q: len(q["question"]))
    assert [len(q["question"]) for q in order] == sorted(len(q["question"]) for q in qs)
    with pytest.raises(ValueError, match="batch_size"):
        ce.score_vqa(vlm, lambda i: i, qs, out, "ck", "SHA", batch_size=8, progress=False)
    with pytest.raises(ValueError, match="missing"):
        ce.load_vqa(out, qs + [dict(qs[0], qid="extra")], "SHA")


def test_pope_correct_and_f1():
    qs = _questions(4, 2)  # k=0 positives, k=1 random negatives
    scores = {q["qid"]: (1.0 if q["answer"] == "yes" else -1.0) for q in qs}
    assert all(v == 1.0 for v in ce.pope_correct(scores, qs).values())
    f1 = ce.pope_f1(scores, qs)
    assert f1["random"]["f1"] == 1.0 and f1["random"]["yes_rate"] == 0.5 and f1["popular"]["n"] == 4  # positives only
    always_yes = {q["qid"]: 1.0 for q in qs}
    assert ce.pope_f1(always_yes, qs)["random"]["precision"] == 0.5
