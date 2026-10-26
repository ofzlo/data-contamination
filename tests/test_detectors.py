import importlib.util
import json

import numpy as np
import pytest
import yaml

from tests.conftest import locked_split, make_annotations
from contam.config import REPO_ROOT, load_config, load_prompts, prompt_sha, read_json, write_json
from contam.data.splits import make_split
from contam.eval import detectors as det


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------- CDD


def test_edit_distance():
    assert det.edit_distance([], []) == 0
    assert det.edit_distance("abc", "") == 3
    assert det.edit_distance(list("kitten"), list("sitting")) == 3
    assert det.edit_distance("a dog on a bench".split(), "a cat on a bench".split()) == 1


def test_cdd_peak_counts_samples_near_greedy():
    greedy = "a dog sits on a red bench in the park near some trees and grass today"  # 15 words
    near = greedy.replace("red", "blue")  # ED 1
    far = "two people ride bicycles along a busy city street at night with lights on"
    # l = 15 words, alpha 0.1 -> threshold 1.5: the exact copy and the one-word change count
    assert det.cdd_peak([greedy, near, far, far], greedy, alpha=0.1) == 0.5
    assert det.cdd_peak([greedy, near, far, far], greedy, alpha=0.0) == 0.25  # alpha 0: exact copies only
    # case and punctuation are normalized away before the token edit distance
    assert det.cdd_peak([greedy.upper() + "."], greedy, alpha=0.0) == 1.0


def test_cdd_peak_caps_length():
    g = " ".join(f"w{k}" for k in range(300))
    s = " ".join(f"w{k}" for k in range(100)) + " different tail " * 50  # same first 100 tokens
    assert det.cdd_peak([s], g, alpha=0.0, max_len=100) == 1.0
    assert det.cdd_peak([s], g, alpha=0.0, max_len=200) == 0.0
    with pytest.raises(ValueError):
        det.cdd_peak([], g)


def _gen(iid, T, k, cap, ck="ck", sha="P"):
    return dict(checkpoint_id=ck, prompt_sha=sha, image_id=iid, temperature=T, sample_idx=k, caption=cap)


def test_cdd_scores_checks_inputs():
    g = [_gen(1, 0.0, 0, "a b c"), _gen(2, 0.0, 0, "x y z")]
    s = [_gen(1, 1.0, k, "a b c") for k in range(3)] + [_gen(2, 1.0, k, "p q r") for k in range(3)]
    assert det.cdd_scores(g, s, [1, 2], alpha=0.0) == {1: 1.0, 2: 0.0}
    with pytest.raises(ValueError, match="lack"):
        det.cdd_scores(g, s, [1, 2, 3])
    with pytest.raises(ValueError, match="different checkpoints"):
        det.cdd_scores(g, [dict(r, checkpoint_id="other") for r in s], [1, 2])
    with pytest.raises(ValueError, match="mix"):
        det.cdd_scores(g + [_gen(3, 0.0, 0, "q", sha="Q")], s, [1, 2])


# ----------------------------------------------------------------------------- DCQ

ORIG = "A brown dog is running across a grassy field."
PERTS = ["A brown dog is sprinting across a grassy field.", "A tan dog is running across a grassy field.",
         "A brown hound is running across a grassy meadow.", "A brown dog is dashing over a grassy field."]


def test_check_perturbations():
    assert det.check_perturbations(ORIG, PERTS) == []
    assert any("equals the original" in p for p in det.check_perturbations(ORIG, PERTS[:3] + ["a brown dog is running across a grassy field"]))
    assert any("not distinct" in p for p in det.check_perturbations(ORIG, PERTS[:3] + [PERTS[0]]))
    assert any("need 4" in p for p in det.check_perturbations(ORIG, PERTS[:3]))
    assert any("length differs" in p for p in det.check_perturbations(ORIG, PERTS[:3] + ["A dog."]))


def test_quiz_options_and_prompt():
    item = det.QuizItem(7, ORIG, PERTS)
    assert det.quiz_options(item) == PERTS + [det.NONE_OPTION]
    bcq = det.quiz_options(item, 2)
    assert bcq[2] == ORIG and bcq[:2] == PERTS[:2] and bcq[3] == PERTS[3]
    with pytest.raises(ValueError):
        det.quiz_options(item, 4)
    prompt = det.dcq_prompt(load_prompts()["dcq_quiz"], bcq)
    assert f"C. {ORIG}" in prompt and f"E. {det.NONE_OPTION}" in prompt and "{options}" not in prompt


@pytest.mark.parametrize("raw, want", [("B", "B"), ("b.", "B"), ("(C)", "C"), ("Answer: D", "D"), ("E. None", "E"), ("", None), ("I am not sure", None)])
def test_parse_answer(raw, want):
    assert det.parse_answer(raw) == want


def test_dcq_estimate_picks_non_preferred_position_and_kappa():
    ids = list(range(10))
    # BDQ: A picked 6 times (preferred, >= ceil(10/5) = 2), B twice (preferred), C once, D never, E once
    bdq = dict(zip(ids, list("AAAAAABBCE")))
    bcq = {
        0: {i: "A" for i in ids},                       # 100% at A, but A is preferred: not eligible
        1: {i: ("B" if i < 9 else "A") for i in ids},   # 90% at B, preferred: not eligible
        2: {i: ("C" if i < 6 else "E") for i in ids},   # 60% at C (BDQ rate 0.1)
        3: {i: ("D" if i < 6 else None) for i in ids},  # 60% at D (BDQ rate 0.0) -> wins the tie
    }
    est = det.dcq_estimate(bdq, bcq)
    assert est["non_preferred"] == ["C", "D"]
    assert est["best_position"] == "D" and est["max"] == 0.6
    assert est["min_kappa"] == pytest.approx(0.6)
    assert est["per_image"] == {i: float(i < 6) for i in ids}
    bcq[2] = {i: ("C" if i < 8 else "E") for i in ids}  # 80% at C now beats D
    est = det.dcq_estimate(bdq, bcq)
    assert est["best_position"] == "C" and est["min_kappa"] == pytest.approx((0.8 - 0.1) / 0.9)
    with pytest.raises(ValueError, match="does not cover"):
        det.dcq_estimate(bdq, {**bcq, 3: {0: "D"}})


def test_dcq_estimate_all_preferred_uses_every_position():
    ids = list(range(8))
    bdq = dict(zip(ids, list("AABBCCDD")))  # each position 2 >= ceil(8/5) = 2
    bcq = {p: {i: det.LETTERS[p] for i in ids} for p in range(4)}
    assert det.dcq_estimate(bdq, bcq)["non_preferred"] == list("ABCD")


def test_dcq_items_requires_training_reference():
    ann = make_annotations(2, 2, 2, seed=1)
    split = make_split(ann, 1, 2, name="A")
    refs = {a["image_id"]: a["refs"] for a in ann}
    iid = split["seen"][0]
    train = refs[iid][split["train_ref_idx"][str(iid)]]
    words = train.split()
    perts = [" ".join(words[:-1] + [w]) for w in ("zebra", "lion", "tiger", "camel")]
    rows = [dict(image_id=iid, original=train, perturbations=perts)]
    assert det.dcq_items(rows, split, refs, [iid])[0].original == train
    with pytest.raises(ValueError, match="training reference"):
        det.dcq_items([dict(rows[0], original="something else")], split, refs, [iid])
    with pytest.raises(ValueError, match="no perturbations"):
        det.dcq_items(rows, split, refs, [iid, split["unseen"][0]])


def test_parse_versions():
    dp = _load_script("dcq_perturb")
    text = "Sure!\nA) one two\nB. three four\nC: five six\nD) seven eight\nA) duplicate"
    assert dp.parse_versions(text) == ["one two", "three four", "five six", "seven eight"]
    assert dp.parse_versions("no list") == []


# ----------------------------------------------------------------------------- KDS


def test_kds_zero_without_change_and_ranks_shifts():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(40, 16))
    assert det.kernel_divergence_score(Z, Z)["kds"] == 0.0
    small = det.kernel_divergence_score(Z, Z + 0.01 * rng.normal(size=Z.shape))
    large = det.kernel_divergence_score(Z, Z + 0.5 * rng.normal(size=Z.shape))
    assert small["kds"] > large["kds"]  # seen-like sets (small shift) score higher (more contaminated)
    assert small["gamma"] == pytest.approx(det.median_gamma(Z))
    # scale-free: embeddings are L2-normalized first
    assert det.kernel_divergence_score(3 * Z, 3 * Z + 0.03 * rng.normal(size=Z.shape))["n"] == 40


def test_kds_matches_direct_formula():
    rng = np.random.default_rng(1)
    Z, Z2 = rng.normal(size=(12, 5)), rng.normal(size=(12, 5))
    g = 0.7
    A, B = det.l2_normalize(Z), det.l2_normalize(Z2)
    K = np.exp(-g * ((A[:, None] - A[None]) ** 2).sum(-1))
    K2 = np.exp(-g * ((B[:, None] - B[None]) ** 2).sum(-1))
    want = -np.abs(K * np.log(K / K2)).sum() / np.sqrt(K.sum())
    assert det.kernel_divergence_score(Z, Z2, gamma=g)["kds"] == pytest.approx(want, rel=1e-9)
    with pytest.raises(ValueError):
        det.kernel_divergence_score(Z, Z2[:5])


# ----------------------------------------------------------------------------- AUROC, groups


def test_auroc():
    assert det.auroc([3, 4], [1, 2]) == 1.0
    assert det.auroc([1, 2], [3, 4]) == 0.0
    assert det.auroc([1, 1], [1, 1]) == 0.5
    assert det.auroc([1, 3], [2, 2]) == 0.5
    rng = np.random.default_rng(0)
    a, b = rng.normal(1, 1, 200), rng.normal(0, 1, 300)
    brute = np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b])
    assert det.auroc(a, b) == pytest.approx(brute)


def test_detector_groups_exclusion_and_subsets(tmp_path):
    ann = make_annotations(10, 30, 20, seed=2)
    split = make_split(ann, 1, 2, name="A")
    excl = tmp_path / "excl.txt"
    excl.write_text(f"# B1\n{split['seen'][0]}\n{split['unseen'][0]}\n")
    cfg = load_config()
    cfg["analysis"]["exclude_image_ids_file"] = str(excl)
    cfg["detectors"]["dcq"]["n_per_group"] = 5
    g, ex = det.detector_groups(cfg, split, "cdd")
    assert ex == sorted([split["seen"][0], split["unseen"][0]])
    assert len(g["seen"]) == len(split["seen"]) - 1 and split["seen"][0] not in g["seen"]
    g2, _ = det.detector_groups(cfg, split, "dcq")
    assert len(g2["seen"]) == len(g2["unseen"]) == 5 and set(g2["seen"]) <= set(g["seen"])
    assert det.detector_groups(cfg, split, "dcq") == (g2, ex)  # same images every call
    g3, ex3 = det.detector_groups(cfg, split, "cdd", keep_excluded=True)
    assert ex3 == [] and len(g3["seen"]) == len(split["seen"])


# ----------------------------------------------------------------------------- CLI


def test_cdd_cli_and_report(tmp_path):
    ann = make_annotations(3, 6, 3, seed=3)
    split = make_split(ann, 1, 2, name="A")
    write_json({"meta": {}, "images": ann}, tmp_path / "ann.json")
    cfg_path, lock, split_path, split_sha = locked_split(tmp_path, split, n_boot=50)
    raw = yaml.safe_load(cfg_path.read_text())
    raw["paths"]["annotations"] = str(tmp_path / "ann.json")
    cfg_path.write_text(yaml.safe_dump(raw))
    sha = prompt_sha(load_prompts()["caption"])
    rd = _load_script("run_detectors")
    outs = []
    for ck, memorized in (("G0_d0_s0", False), ("G0_d0_s1", False), ("paired_d64_s0", True)):
        g, s = tmp_path / f"{ck}_t0.jsonl", tmp_path / f"{ck}_t1.jsonl"
        with open(g, "w") as fg, open(s, "w") as fs:
            for a in ann:
                iid = a["image_id"]
                cap = "a dog on a red bench in a park"
                fg.write(json.dumps(dict(_gen(iid, 0.0, 0, cap, ck, sha), prompt_id="caption")) + "\n")
                copy = memorized and iid in split["seen"]
                for k in range(4):
                    fs.write(json.dumps(dict(_gen(iid, 1.0, k, cap if copy or k == 0 else f"something else entirely {k} {iid}", ck, sha), prompt_id="caption")) + "\n")
        out = tmp_path / f"cdd_{ck}.json"
        rd.main(["cdd", "--config", str(cfg_path), "--lock", str(lock), "--split", str(split_path), "--checkpoint-id", ck,
                 "--greedy", str(g), "--samples", str(s), "--out", str(out)])
        res = read_json(out)
        assert res["gate_valid"] is True and res["manifest_split_sha256"] == split_sha
        outs.append(out)
    res = read_json(outs[-1])
    assert res["summary"]["seen"]["mean"] == 1.0 and res["summary"]["unseen"]["mean"] == 0.25
    assert res["summary"]["auroc"]["estimate"] == 1.0
    assert read_json(outs[0])["summary"]["auroc"]["estimate"] == 0.5
    with pytest.raises(SystemExit, match="expected"):  # rows of another checkpoint
        rd.main(["cdd", "--config", str(cfg_path), "--lock", str(lock), "--split", str(split_path), "--checkpoint-id", "other",
                 "--greedy", str(tmp_path / "G0_d0_s0_t0.jsonl"), "--samples", str(tmp_path / "G0_d0_s0_t1.jsonl"), "--out", str(tmp_path / "x.json")])
    rep_path = tmp_path / "b6.json"
    rd.main(["report", "--inputs", *map(str, outs), "--out", str(rep_path)])
    rep = read_json(rep_path)
    spread = rep["detectors"]["cdd"]["d0_spread"]["auroc"]
    assert rep["gate_valid"] is True and spread["n_seeds"] == 2 and spread["mean"] == 0.5 and spread["sd"] == 0.0
    assert set(rep["detectors"]["cdd"]["checkpoints"]) == {"G0_d0_s0", "G0_d0_s1", "paired_d64_s0"}
