"""Phase 0 yes/no scoring and analysis (contam.eval.yesno), CPU only (fake model)."""

import hashlib
import json

import numpy as np
import pytest

from contam.config import read_jsonl
from contam.eval import yesno

VOCAB = {"▁Yes": 1, "▁yes": 2, "Yes": 3, "▁No": 4, "▁no": 5, "No": 6}


class FakeTok:
    unk_token_id = 0

    def convert_tokens_to_ids(self, t):
        return VOCAB.get(t, 0)


class FakeProc:
    tokenizer = FakeTok()


class FakeVLM:
    """Deterministic log-probs from (image, prompt); records each batch's prompts."""

    model_id, adapter = "fake", "ad"

    def __init__(self):
        self.processor = FakeProc()
        self.calls = []

    def first_token_logprobs(self, images, prompts, token_ids):
        self.calls.append(list(prompts))
        out = []
        for im, p in zip(images, prompts):
            h = int(hashlib.sha256(f"{im}|{p}".encode()).hexdigest(), 16)
            out.append([-1.0 - ((h >> (8 * k)) % 50) / 10 for k in range(len(token_ids))])
        return np.array(out)


def _q(i, img, cell, groups):
    return {"qid": f"{img}:{i}", "image_id": img, "file_name": f"{img}.jpg", "question": f"Is there a thing{i} in the image?",
            "answer": "no" if cell == "no" else "yes", "cell": cell, "group": groups}


def make_questions(n_seen=60, n_unseen=60):
    qs = []
    for img in range(n_seen + n_unseen):
        if img < n_seen:
            g = {"1000": "seen", "300": "seen" if img < 20 else "unexposed", "100": "seen" if img < 8 else "unexposed", "30": "unexposed"}
        else:
            g = {c: "unseen" for c in yesno.N_EXP_COLUMNS}
        for i, cell in enumerate(("yes/direct", "yes/none", "yes/none", "no", "no")):
            qs.append(_q(i, img, cell, dict(g)))
    return qs


def test_answer_token_ids():
    assert yesno.answer_token_ids(FakeTok(), yesno.YES_WORDS) == [1, 2, 3]
    assert yesno.answer_token_ids(FakeTok(), yesno.NO_WORDS) == [4, 5, 6]
    with pytest.raises(ValueError):
        yesno.answer_token_ids(FakeTok(), ["Maybe"])


def test_load_questions_checks(tmp_path):
    p = tmp_path / "q.share.json"
    qs = make_questions(2, 2)
    p.write_text(json.dumps({"questions": qs[::-1]}))
    got, sha = yesno.load_questions(p)
    assert [q["qid"] for q in got] == sorted((q["qid"] for q in qs), key=lambda s: (int(s.split(":")[0]), s))
    with pytest.raises(ValueError, match="preregistered"):
        yesno.load_questions(p, "0" * 64)
    bad = dict(qs[0], answer="no")  # a yes cell answered no
    p.write_text(json.dumps({"questions": [bad] + qs[1:]}))
    with pytest.raises(ValueError, match="does not fit"):
        yesno.load_questions(p)


def test_run_ids():
    assert yesno.n_exp_column("EX_Paired_d1_n30_s0") == "30"
    assert yesno.n_exp_column("EX_Caption_d2_n1000_s1") == "1000"
    assert yesno.n_exp_column("G1_Paired_d4_s0") == "1000"
    assert yesno.run_cell("G1_Paired_d4_s1") == ("G1_Paired_d4", 1)
    with pytest.raises(ValueError):
        yesno.n_exp_column("EX_Paired_d1_n50_s0")


def test_score_rows_resume_and_fixed_batches(tmp_path):
    qs = yesno.chunks(make_questions(2, 1), 100)[0][:10]
    out = tmp_path / "s.jsonl"
    vlm = FakeVLM()
    assert yesno.score_questions(vlm, lambda i: f"img{i}", qs[:6], out, "ck", "SHA", batch_size=4, progress=False) == 6
    # full run: chunk [4:8] holds stale rows 4, 5 -> rescored whole; chunk [8:10] new
    assert yesno.score_questions(vlm, lambda i: f"img{i}", qs, out, "ck", "SHA", batch_size=4, progress=False) == 6
    rows = list(read_jsonl(out))
    assert sorted(r["qid"] for r in rows) == sorted(q["qid"] for q in qs)
    want_chunks = [[yesno.question_prompt(q) for q in c] for c in yesno.chunks(qs, 4)]
    assert vlm.calls[-2:] == want_chunks[1:]  # same batch composition as a fresh run
    r = rows[0]
    lp = vlm.first_token_logprobs([f"img{r['image_id']}"], [yesno.question_prompt(next(q for q in qs if q["qid"] == r["qid"]))], [1, 2, 3, 4, 5, 6])[0]
    ly, ln = np.log(np.exp(lp[:3]).sum()), np.log(np.exp(lp[3:]).sum())
    assert r["score"] == pytest.approx(ly - ln) and r["mass"] == pytest.approx(np.exp(ly) + np.exp(ln))
    assert r["yes_ids"] == [1, 2, 3] and r["no_ids"] == [4, 5, 6] and r["batch_size"] == 4
    # nothing left to do; a different batch size is refused
    assert yesno.score_questions(vlm, lambda i: f"img{i}", qs, out, "ck", "SHA", batch_size=4, progress=False) == 0
    with pytest.raises(ValueError, match="batch_size"):
        yesno.score_questions(vlm, lambda i: f"img{i}", qs, out, "ck", "SHA", batch_size=8, progress=False)
    scores, prov = yesno.load_scores(out, qs, "SHA")
    assert len(scores) == 10 and prov["checkpoint_id"] == "ck"
    with pytest.raises(ValueError, match="missing"):
        yesno.load_scores(out, make_questions(3, 1), "SHA")


def _scores(qs, rng, shift=None, noise=0.05):
    base = {q["qid"]: (2.0 if q["answer"] == "yes" else -2.0) + (int(q["qid"].split(":")[0]) % 7) * 0.1 for q in qs}
    out = {}
    for q in qs:
        s = base[q["qid"]] + rng.normal(0, noise)
        if shift and q["group"]["1000"] == "seen":
            s += shift.get(q["cell"], 0.0)
        out[q["qid"]] = s
    return out


def test_margins_match_manual():
    qs = make_questions()
    rng = np.random.default_rng(0)
    d0 = {f"G0_d0_s{k}": _scores(qs, rng) for k in range(5)}
    m = yesno.margins(qs, d0, "1000")
    for cell in yesno.CELLS:
        sel = [q for q in qs if q["cell"] == cell]
        if not sel:
            assert m[cell]["delta"] is None
            continue
        gaps = [np.mean([s[q["qid"]] for q in sel if q["group"]["1000"] == "seen"]) - np.mean([s[q["qid"]] for q in sel if q["group"]["1000"] == "unseen"]) for s in d0.values()]
        assert m[cell]["delta"] == pytest.approx(2 * np.std(gaps, ddof=1))


@pytest.mark.parametrize("shift,expect", [
    ({"yes/direct": 1.0, "yes/none": -1.0, "no": -1.0}, "P2 (caption closure, mention-specific)"),
    ({"yes/direct": -1.0, "yes/none": -1.0, "no": -1.0}, "P2 direction, not mention-specific (seen-image shift toward no)"),
    ({"yes/direct": 1.0, "yes/none": 1.0, "no": -1.0}, "familiarity"),
    ({"yes/direct": 1.0}, "neither"),
    ({}, "neither"),
])
def test_primary_verdict(shift, expect):
    qs = make_questions()
    rng = np.random.default_rng(1)
    d0 = {f"G0_d0_s{k}": _scores(qs, rng) for k in range(5)}
    pooled = {q["qid"]: float(np.mean([s[q["qid"]] for s in d0.values()])) for q in qs}
    runs = {str(k): _scores(qs, rng, shift) for k in (0, 1)}
    m = yesno.margins(qs, d0, "1000")
    eff = {c: yesno.cell_effects(qs, runs, pooled, "1000", c, n_boot=300, boot_seed=0, ci=0.95) for c in yesno.CELLS}
    assert eff["yes/descendant"] is None
    assert eff["yes/none"]["n_questions"] == {"seen": 120, "unseen": 120}
    assert eff["yes/none"]["n_images"] == {"seen": 60, "unseen": 60}
    v = yesno.primary_verdict(eff, m)
    assert v["verdict"] == expect
    assert v["p1_control_pass"] == (shift.get("yes/direct", 0) > 0)
    if shift:
        assert eff["yes/none"]["pooled"]["estimate"] == pytest.approx(shift.get("yes/none", 0.0), abs=0.05)


def test_unexposed_left_out():
    qs = make_questions()
    rng = np.random.default_rng(2)
    d0 = {f"G0_d0_s{k}": _scores(qs, rng) for k in range(5)}
    pooled = {q["qid"]: float(np.mean([s[q["qid"]] for s in d0.values()])) for q in qs}
    eff = yesno.cell_effects(qs, {"0": pooled, "1": pooled}, pooled, "300", "no", n_boot=50, boot_seed=0, ci=0.95)
    assert eff["n_images"]["seen"] == 20 and eff["pooled"]["estimate"] == pytest.approx(0.0)
    assert yesno.cell_effects(qs, {"0": pooled}, pooled, "30", "no", n_boot=50, boot_seed=0, ci=0.95) is None  # no seen images


def test_analysis_questions_drop_body_part_no():
    qs = [dict(_q(0, 0, "no", {}), is_body_part=True), dict(_q(1, 0, "yes/none", {}), is_body_part=True),
          dict(_q(2, 0, "no", {}), is_body_part=False), _q(3, 0, "yes/direct", {})]
    assert [q["qid"] for q in yesno.analysis_questions(qs)] == ["0:1", "0:2", "0:3"]
    assert [q["qid"] for q in yesno.analysis_questions(qs, exclude_body_parts=True)] == ["0:2", "0:3"]
    assert len(yesno.analysis_questions(qs, exclude_body_part_no=False)) == 4
