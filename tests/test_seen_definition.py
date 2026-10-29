"""Seen definition (prereg Section 2, 2026-10-28): seen = the N_exp exposure set, seen_unexposed reported only."""

import json
import os
import subprocess
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pytest
import yaml

from contam.config import REPO_ROOT, content_hash, load_config, read_json, write_json
from contam.data.exposure import exposure_ids
from contam.data.splits import make_split
from contam.eval.evaluate import analysis_ids, check_seen, compare, evaluate_checkpoint, load_analysis_exclusion, load_seen_definition, seen_record
from contam.eval.gates import g1_report, g2_report, validate_g2_inputs, validate_inputs
from contam.eval.stats import association_margin
from tests.conftest import base_metrics, locked_split, make_annotations
from tests.test_g2 import MARGINS, make_g2
from tests.test_run_g0_eval import g0

N_EXP = 60  # not a stored exposure set of the synthetic split (100 / 25 / 6): rebuilt from the annotations


@pytest.fixture
def synth():
    ann = make_annotations(30, 120, 50, seed=3)
    split = make_split(ann, 1, 2)
    return ann, split, load_seen_definition(load_config(), split, ann, N_EXP)


def _rec(m, seen):
    return dict(m, seen_definition=seen_record(seen))


# ----------------------------------------------------------------------------- the real splits


SPLIT_A = REPO_ROOT / "data/splits/split_A.json"
real = pytest.mark.skipif(not SPLIT_A.exists(), reason="needs the real splits and annotations (data/)")


@real
def test_real_split_A_seen_is_the_g1_exposure_set():
    cfg = load_config()
    from contam.data.nocaps import load_annotations

    ann = load_annotations(cfg["paths"]["annotations"])
    split = read_json(SPLIT_A)
    seen = load_seen_definition(cfg, split, ann)
    assert cfg["exposure"]["n_exp"] == 1000 and seen["n_exp"] == 1000 and seen["kind"] == "exposure"
    assert seen["image_ids"] == exposure_ids(split, ann, 1000)
    assert seen["ids_sha256"] == content_hash(seen["image_ids"]) and seen["ids_sha256"].startswith("fccfe788")
    for spec in glob(str(REPO_ROOT / "outputs/runs/G1/*.json")):  # the built G1 runs train on exactly this set
        assert read_json(spec)["exposure"]["exposure_ids_sha256"] == seen["ids_sha256"], spec
    allids = sorted(set(split["seen"]) | set(split["unseen"]))
    n = {g: len(analysis_ids(split, g, allids, None, seen)) for g in ("seen", "unseen", "seen_unexposed")}
    assert n == {"seen": 1000, "unseen": 2250, "seen_unexposed": 1250}
    excl = load_analysis_exclusion(cfg)
    b1 = set(excl["image_ids"])
    assert len(b1 & set(split["seen"])) == 9 and len(b1 & set(seen["image_ids"])) == 4  # 4 trained on, excluded from analysis
    n = {g: len(analysis_ids(split, g, allids, excl, seen)) for g in ("seen", "unseen", "seen_unexposed")}
    assert n == {"seen": 996, "unseen": 2239, "seen_unexposed": 1245}
    nested = [set(load_seen_definition(cfg, split, ann, k)["image_ids"]) for k in (141, 563, 1000, 2250)]
    assert [len(s) for s in nested] == [141, 563, 1000, 2250] and all(a < b for a, b in zip(nested, nested[1:]))
    assert nested[-1] == set(split["seen"])


# ----------------------------------------------------------------------------- groups


def test_groups_follow_the_exposure_set(synth):
    ann, split, seen = synth
    allids = sorted(set(split["seen"]) | set(split["unseen"]))
    exp = exposure_ids(split, ann, N_EXP)
    assert analysis_ids(split, "seen", allids, None, seen) == exp
    assert analysis_ids(split, "seen_unexposed", allids, None, seen) == sorted(set(split["seen"]) - set(exp))
    assert analysis_ids(split, "unseen", allids, None, seen) == sorted(split["unseen"])
    assert analysis_ids(split, "seen", allids) == sorted(split["seen"]) and analysis_ids(split, "seen_unexposed", allids) == []  # seen=None: legacy
    ex = {"file": "x", "sha256": "E", "image_ids": [exp[0], sorted(set(split["seen"]) - set(exp))[0]], "applied": True}
    assert analysis_ids(split, "seen", allids, ex, seen) == exp[1:]
    # a stored size needs no annotations and gives the stored set; the sets nest
    stored = load_seen_definition(load_config(), split, n_exp=len(split["seen"]))
    assert stored["image_ids"] == sorted(split["seen"])
    small = load_seen_definition(load_config(), split, ann, 20)["image_ids"]
    assert set(small) < set(exp) < set(split["seen"])


def test_check_seen_refuses_other_or_missing_definitions(synth):
    ann, split, seen = synth
    other = load_seen_definition(load_config(), split, ann, N_EXP + 10)
    check_seen([("ok", _rec({}, seen))], seen)
    check_seen([("legacy", {})], None)
    for bad in ({}, _rec({}, other), _rec({}, None)):
        with pytest.raises(ValueError, match="seen definition"):
            check_seen([("x", bad)], seen)


def _metrics(split, name, per):
    return {"checkpoint_id": name, "split": {"name": split["name"], "content_hash": split["content_hash"]}, "complete": True, "generations_hash": name, "per_image": per}


def test_compare_uses_the_exposure_set(synth):
    ann, split, seen = synth
    exp = set(seen["image_ids"])
    ids = sorted(set(split["seen"]) | set(split["unseen"]))
    # the model gains 3 CIDEr on the exposure set only, and adds 2 words on seen_unexposed only
    model = _metrics(split, "m", {str(i): {"cider_heldout": 3.0 if i in exp else 0.0, "word_count": 2.0 if (i in split["seen"] and i not in exp) else 0.0} for i in ids})
    base = _metrics(split, "b", {str(i): {"cider_heldout": 0.0, "word_count": 0.0} for i in ids})
    c = compare([_rec(model, seen)], [_rec(base, seen)], split, n_boot=50, seen=seen)
    assert c["n_seen"] == N_EXP and c["n_seen_unexposed"] == len(split["seen"]) - N_EXP and c["n_unseen"] == len(split["unseen"])
    assert c["A"]["cider_heldout"]["estimate"] == pytest.approx(3.0)
    assert c["A_seen_unexposed"]["cider_heldout"]["estimate"] == pytest.approx(0.0)
    assert c["S"]["word_count"]["seen"]["estimate"] == pytest.approx(0.0) and c["S"]["word_count"]["seen_unexposed"]["estimate"] == pytest.approx(2.0)
    assert c["seen_definition"] == seen_record(seen)
    legacy = compare([model], [base], split, n_boot=50)  # the split's 50% set: diluted
    assert legacy["A"]["cider_heldout"]["estimate"] == pytest.approx(3.0 * N_EXP / len(split["seen"]))
    for bad in (model, _rec(model, load_seen_definition(load_config(), split, ann, 20))):
        with pytest.raises(ValueError, match="seen definition"):
            compare([bad], [_rec(base, seen)], split, n_boot=50, seen=seen)


def test_evaluate_checkpoint_reuses_q_and_groups_by_exposure(synth):
    ann, split, seen = synth
    gens = {int(a["image_id"]): a["refs"][1] for a in ann}
    q = {i: {f"{m}_{c}": float(i) for m in ("cider", "spice") for c in ("heldout", "train", "all10")} for i in gens}
    res = evaluate_checkpoint(gens, ann, split, reuse_q=q, seen=seen, checkpoint_id="ck")
    s = res["summary"]
    assert (s["seen"]["n"], s["seen_unexposed"]["n"], s["unseen"]["n"]) == (N_EXP, len(split["seen"]) - N_EXP, len(split["unseen"]))
    assert s["seen"]["cider_heldout"] == pytest.approx(np.mean(seen["image_ids"]))
    assert res["per_image"]["5"]["spice_train"] == 5.0 and "word_count" in res["per_image"]["5"]
    assert res["seen_definition"] == seen_record(seen) and any(k.startswith("seen_unexposed/") for k in res["by_domain"])
    del q[5]["spice_all10"]
    with pytest.raises(ValueError, match="reused scores"):
        evaluate_checkpoint(gens, ann, split, reuse_q=q, seen=seen)


# ----------------------------------------------------------------------------- margins (CLI)


def _cfg_with_n_exp(tmp_path, split, ann, n_exp=N_EXP):
    cfg_path, lock, split_path, split_sha = locked_split(tmp_path, split)
    raw = yaml.safe_load(cfg_path.read_text())
    raw["exposure"]["n_exp"] = n_exp
    raw["paths"]["annotations"] = str(tmp_path / "ann.json")
    cfg_path.write_text(yaml.safe_dump(raw))
    write_json({"meta": {}, "images": ann}, tmp_path / "ann.json")
    return cfg_path, lock, split_path, split_sha


def test_margins_delta_A_uses_the_exposure_set(tmp_path, synth):
    ann, split, seen = synth
    cfg_path, lock, split_path, split_sha = _cfg_with_n_exp(tmp_path, split, ann)
    excl = load_analysis_exclusion(load_config(cfg_path))
    exp = set(seen["image_ids"])
    ids = sorted(set(split["seen"]) | set(split["unseen"]))
    e, u = [0.0, 1.0, 3.0, 2.0, 7.0], [0.0, 0.0, 0.0, 0.0, 0.5]
    prov = {"generations": {"prompt_sha": "P", "model": "m", "batch_size": 16, "max_new_tokens": 300}}
    d0 = []
    for s in range(5):
        per = {str(i): {"word_count": 10.0 + s, "boilerplate": 0.0, "cider_all10": 0.5,
                        "cider_heldout": 15.0 + e[s] if i in exp else (15.0 + u[s] if i in split["seen"] else 10.0)} for i in ids}
        m = dict(_metrics(split, f"d0_s{s}", per), manifest_split_sha256=split_sha, analysis_exclusion=excl, seen_definition=seen_record(seen), provenance=prov)
        d0.append(str(tmp_path / f"d0_s{s}.json"))
        write_json(m, d0[-1])
    write_json(base_metrics(read_json(d0[0])), tmp_path / "base.json")
    cmd = [sys.executable, str(REPO_ROOT / "scripts/compute_margins.py"), "--config", str(cfg_path), "--lock", str(lock), "--split", str(split_path), "--d0", *d0, "--base", str(tmp_path / "base.json"), "--out"]
    r = subprocess.run(cmd + [str(tmp_path / "m.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = read_json(tmp_path / "m.json")
    assert m["seen_definition"] == seen_record(seen) and m["gate_valid"] is True
    assert m["per_seed"]["seen_minus_unseen/cider_heldout"] == pytest.approx([5.0 + x for x in e])
    assert m["per_seed"]["seen_unexposed_minus_unseen/cider_heldout"] == pytest.approx([5.0 + x for x in u])
    assert m["delta_A"]["cider_heldout"] == pytest.approx(association_margin([15.0 + x for x in e], [10.0] * 5, 2.0))
    assert m["analysis_exclusion"]["n_excluded"].keys() >= {"seen", "seen_unexposed", "unseen"}
    # --n-exp overrides the config, and is then refused for inputs recorded under N_exp = 100
    r = subprocess.run(cmd + [str(tmp_path / "m2.json"), "--n-exp", "20"], capture_output=True, text=True)
    assert r.returncode != 0 and "seen definition" in r.stderr
    # metrics from before the seen definition (none recorded) are refused; so is such a base
    write_json({k: v for k, v in read_json(tmp_path / "base.json").items() if k != "seen_definition"}, tmp_path / "base.json")
    r = subprocess.run(cmd + [str(tmp_path / "m3.json")], capture_output=True, text=True)
    assert r.returncode != 0 and "records seen definition None" in r.stderr and "--reuse-scores" in r.stderr
    write_json({k: v for k, v in read_json(d0[0]).items() if k != "seen_definition"}, d0[0])
    r = subprocess.run(cmd + [str(tmp_path / "m4.json")], capture_output=True, text=True)
    assert r.returncode != 0 and "records seen definition None" in r.stderr


# ----------------------------------------------------------------------------- G1 / G2


def test_g1_uses_and_requires_the_seen_definition(g1_setup):
    split, baseline, margins, cfg, ckpt = g1_setup
    ann = make_annotations(10, 40, 20, seed=9)  # g1_setup's annotations; N_exp = 9 is rebuilt from them
    seen = load_seen_definition(cfg, split, ann, 9)
    cells = [("Paired", 4, s, ckpt(f"p{s}", 2.0)) for s in range(2)]
    cks = [{"condition": c, "d": d, "seed": s, "metrics": f"/tmp/{c}_{d}_{s}.json"} for (c, d, s, _) in cells]
    ms = [_rec(m, seen) for *_, m in cells]
    base = [_rec(b, seen) for b in baseline]
    paths = [f"/tmp/b{s}.json" for s in range(5)]
    with pytest.raises(ValueError, match="margins records seen definition None"):
        validate_inputs(split, margins, paths, base, cks, ms, 2, seen=seen)
    validate_inputs(split, _rec(margins, seen), paths, base, cks, ms, 2, seen=seen)
    with pytest.raises(ValueError, match="seen definition"):
        validate_inputs(split, _rec(margins, seen), paths, baseline, cks, ms, 2, seen=seen)  # old baseline metrics
    rep = g1_report(split, _rec(margins, seen), base, cks, ms, cfg, seen=seen)
    pooled = rep["cells"]["Paired/d4"]["pooled"]
    assert rep["seen_definition"] == seen_record(seen) and pooled["n_seen"] == 9 and pooled["n_seen_unexposed"] == len(split["seen"]) - 9
    assert set(pooled["S"]["word_count"]) == {"seen", "unseen", "seen_unexposed"}


def test_g2_tau_and_two_level_bootstrap_use_the_exposure_set():
    split, margins, g1, base_paths, baseline, cks, cms, cfg = make_g2()
    ann = make_annotations(10, 40, 20, seed=9)
    seen = load_seen_definition(cfg, split, ann, 12)
    exp = set(seen["image_ids"])
    # memory effect only on the exposure set: in the legacy (50% set) view it is diluted
    for c, m in zip(cks, cms):
        if c["condition"] == "Paired":
            for k, v in m["per_image"].items():
                if int(k) not in exp:
                    v["rmi_sum"] = baseline[0]["per_image"][k]["rmi_sum"]
    args = (split, _rec(margins, seen), MARGINS, _rec(g1, seen), base_paths, [_rec(b, seen) for b in baseline], cks, [_rec(m, seen) for m in cms], cfg)
    validate_g2_inputs(*args, seen=seen)
    with pytest.raises(ValueError, match="the G1 report records seen definition None"):
        validate_g2_inputs(*args[:3], g1, *args[4:], seen=seen)
    rep = g2_report(args[0], args[1], args[3], args[5], args[6], args[7], cfg, seen=seen)
    assert rep["seen_definition"] == seen_record(seen)
    prim = rep["criteria"]["primary"]
    cell = [m for c, m in zip(cks, cms) if c["condition"] == "Paired" and c["d"] == prim["d_style"]]
    delta = [np.mean([m["per_image"][str(i)]["rmi_sum"] for m in cell]) - np.mean([b["per_image"][str(i)]["rmi_sum"] for b in baseline]) for i in sorted(exp)]
    assert prim["v_effect"]["estimate"] == pytest.approx(np.mean(delta))  # over the 12 exposure-set images only
    assert prim["rules"]["two_level_bootstrap"]["estimate"] == pytest.approx(np.mean(delta))
    assert abs(prim["v_effect_seen_unexposed"]["estimate"]) < 0.05  # no effect there (seed noise only); reported, not gating
    mem = rep["conditions"]["Paired"]["memory"]
    legacy = g2_report(split, margins, g1, baseline, cks, cms, cfg)  # the 50% set: the d=64 memory effect is diluted
    assert mem["effect"][-1] > 1.5 * legacy["conditions"]["Paired"]["memory"]["effect"][-1]


# ----------------------------------------------------------------------------- run_g0_eval staleness, --reuse-scores


def test_json_current_requires_the_seen_definition(tmp_path, synth):
    ann, split, seen = synth
    inp, out = tmp_path / "in.json", tmp_path / "out.json"
    write_json({}, inp)
    rec = {"gate_valid": True, "manifest_split_sha256": "S", "analysis_exclusion": {"sha256": "E"}}
    write_json(_rec(rec, seen), out)
    assert g0.json_current(out, [inp], "S", "E", seen=seen)
    assert not g0.json_current(out, [inp], "S", "E", seen=load_seen_definition(load_config(), split, ann, 20))
    write_json(rec, out)  # from before the seen definition: stale
    assert not g0.json_current(out, [inp], "S", "E", seen=seen)


def test_evaluate_checkpoint_cli_reuse_scores(tmp_path, synth):
    ann, split, seen = synth
    cfg_path, lock, split_path, split_sha = _cfg_with_n_exp(tmp_path, split, ann)
    prov = dict(checkpoint_id="ck", model="m", adapter=None, prompt_sha="AAA")
    with open(tmp_path / "g.jsonl", "w") as f:
        for a in ann:
            f.write(json.dumps(dict(prov, image_id=a["image_id"], prompt_id="caption", temperature=0.0, sample_idx=0, seed=None, caption=a["refs"][0])) + "\n")
    out = tmp_path / "m.json"
    cmd = [sys.executable, str(REPO_ROOT / "scripts/evaluate_checkpoint.py"), "--config", str(cfg_path), "--lock", str(lock), "--split", str(split_path), "--checkpoint-id", "ck",
           "--generations", str(tmp_path / "g.jsonl"), "--tokenizer", "simple", "--metrics", "cider", "--no-spice-cache", "--out", str(out)]
    r = subprocess.run(cmd + ["--reuse-scores"], capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, r.stderr
    assert "scoring from scratch" in r.stdout
    m = read_json(out)
    assert m["seen_definition"] == seen_record(seen) and m["summary"]["seen"]["n"] == N_EXP - m["analysis_exclusion"]["n_excluded"]["seen"]
    # an old file: other group statistics, no seen definition, sentinel CIDEr scores -> reused, regrouped
    for v in m["per_image"].values():
        v.update(cider_heldout=-1.0, word_count=-5.0)
    old = {k: v for k, v in m.items() if k != "seen_definition"}
    old["summary"] = {}
    write_json(old, out)
    r = subprocess.run(cmd + ["--reuse-scores"], capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0, r.stderr
    assert "CIDEr / SPICE reused" in r.stdout
    m2 = read_json(out)
    assert m2["seen_definition"] == seen_record(seen) and m2["summary"]["seen"]["cider_heldout"] == -1.0
    assert m2["per_image"]["0"]["word_count"] > 0  # non-Q fields are recomputed
    # other generations: not reused
    os.utime(out, (0, 0))  # older than its generations
    r = subprocess.run(cmd + ["--reuse-scores"], capture_output=True, text=True, cwd=REPO_ROOT)
    assert r.returncode == 0 and "older than its generations" in r.stdout
    assert read_json(out)["summary"]["seen"]["cider_heldout"] != -1.0
    assert Path(out).exists()
