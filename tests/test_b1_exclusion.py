"""B1 analysis exclusion (prereg Section 2): scored with the rest, left out of group means, Δ, margins and gates."""

import subprocess
import sys

import numpy as np
import pytest
import yaml

from contam.config import REPO_ROOT, file_sha256, load_config, read_json, write_json
from contam.data.splits import group_ids, make_split
from contam.eval.evaluate import compare, evaluate_checkpoint, load_analysis_exclusion
from contam.eval.gates import g1_report, g2_report, validate_g2_inputs, validate_inputs
from tests.conftest import base_metrics, locked_split, make_annotations, seen_rec
from tests.test_g2 import make_g2

PATHS = [f"/tmp/b{s}.json" for s in range(5)]


def _excl(ids, applied=True, sha="E"):
    return {"file": "excl.txt", "sha256": sha, "image_ids": sorted(ids), "applied": applied}


def _rec(m, excl):
    return dict(m, analysis_exclusion=excl)


def test_load_analysis_exclusion_default_config():
    e = load_analysis_exclusion(load_config())
    assert len(e["image_ids"]) == 20 and e["applied"] is True and 82 in e["image_ids"]
    assert e["sha256"] == file_sha256(REPO_ROOT / "docs/results/b1_overlap_nocaps_image_ids.txt")
    assert load_analysis_exclusion(load_config(), include=True) == dict(e, applied=False)


def test_group_means_and_delta_leave_out_exactly_the_excluded_ids():
    split = make_split(make_annotations(4, 10, 6, seed=3), 1, 2)
    seen, unseen = sorted(split["seen"]), sorted(split["unseen"])
    out = {seen[0], seen[1], unseen[0]}

    def metrics(name, wc, cider):  # per-image word count / CIDEr; excluded images get extreme values
        per = {str(i): {"word_count": wc(i), "cider_heldout": cider(i)} for i in seen + unseen}
        return {"checkpoint_id": name, "split": {"name": split["name"], "content_hash": split["content_hash"]}, "complete": True, "per_image": per}

    model = metrics("m", lambda i: 1000.0 if i in out else 10.0 + i, lambda i: 100.0 if i in out else (5.0 if i in seen else 2.0))
    base = metrics("b", lambda i: 10.0, lambda i: 0.0)
    ex = _excl(out)
    c = compare([_rec(model, ex)], [_rec(base, ex)], split, n_boot=50, exclusion=ex)
    kept_unseen = [i for i in unseen if i not in out]
    assert c["n_seen"] == len(seen) - 2 and c["n_unseen"] == len(unseen) - 1
    assert c["S"]["word_count"]["unseen"]["estimate"] == pytest.approx(np.mean(kept_unseen))  # (10 + i) - 10
    assert c["A"]["cider_heldout"]["estimate"] == pytest.approx(5.0 - 2.0)
    assert c["analysis_exclusion"]["n_excluded"] == {"seen": 2, "unseen": 1, "seen_unexposed": 0, "shared_seen": len(out & set(group_ids(split, "shared_seen"))), "all": 3}
    assert c["gate_valid"] is True and c["robustness"] is False

    # robustness view: same inputs, every image kept, not gate-valid
    inc = dict(ex, applied=False)
    r = compare([_rec(model, ex)], [_rec(base, ex)], split, n_boot=50, exclusion=inc)
    assert r["n_seen"] == len(seen) and r["S"]["word_count"]["unseen"]["estimate"] == pytest.approx(np.mean([990.0 if i in out else i for i in unseen]))
    assert r["gate_valid"] is False and r["robustness"] is True and r["robustness_valid"] is True

    # mixed exclusions (or none recorded: metrics from before the exclusion) are refused
    for bad in (_rec(base, _excl(out, sha="F")), base):
        with pytest.raises(ValueError, match="B1 analysis exclusion"):
            compare([_rec(model, ex)], [bad], split, n_boot=50, exclusion=ex)


def test_evaluate_checkpoint_scores_every_image_once():
    ann = make_annotations(4, 10, 6, seed=4)
    split = make_split(ann, 1, 2)
    seen, unseen = sorted(split["seen"]), sorted(split["unseen"])
    ex = _excl({seen[0], unseen[0], unseen[1]})
    gens = {a["image_id"]: a["refs"][(split["train_ref_idx"][str(a["image_id"])] + 1) % 10] for a in ann}
    kw = dict(metrics=["cider"], tokenizer="simple", checkpoint_id="x")
    full, cut = evaluate_checkpoint(gens, ann, split, **kw), evaluate_checkpoint(gens, ann, split, exclusion=ex, **kw)
    assert cut["per_image"] == full["per_image"] and len(cut["per_image"]) == len(ann)  # one CIDEr IDF corpus
    assert cut["complete"] is True and full["analysis_exclusion"] is None
    assert cut["analysis_exclusion"]["n_excluded"]["seen"] == 1 and cut["analysis_exclusion"]["n_excluded"]["unseen"] == 2
    for g, ids in (("seen", seen[1:]), ("unseen", unseen[2:]), ("all", sorted(set(seen[1:] + unseen[2:])))):
        assert cut["summary"][g]["n"] == len(ids)
        assert cut["summary"][g]["cider_all10"] == pytest.approx(np.mean([full["per_image"][str(i)]["cider_all10"] for i in ids]))
    assert full["summary"]["seen"]["n"] == len(seen)
    assert sum(v["n"] for k, v in cut["by_domain"].items() if k.startswith("unseen/")) == len(unseen) - 2


def _gate_case(g1_setup, ex):
    split, baseline, margins, cfg, ckpt = g1_setup
    cks = [{"condition": "Paired", "d": 4, "seed": s, "metrics": f"/tmp/p{s}.json"} for s in range(2)]
    ms = [_rec(ckpt(f"p{s}", 2.0), ex) for s in range(2)]
    return split, [_rec(m, ex) for m in baseline], _rec(margins, ex), cfg, cks, ms


def test_gate_inputs_must_share_the_exclusion(g1_setup):
    split, base_, margins_, *_ = g1_setup
    ex = _excl([split["seen"][0], split["unseen"][0]])
    split, baseline, margins, cfg, cks, ms = _gate_case(g1_setup, ex)
    validate_inputs(split, margins, PATHS, baseline, cks, ms, 2, exclusion=ex)
    for bad_ms in ([ms[0], _rec(ms[1], _excl(ex["image_ids"], sha="F"))], [ms[0], {k: v for k, v in ms[1].items() if k != "analysis_exclusion"}]):
        with pytest.raises(ValueError, match="B1 analysis exclusion"):
            validate_inputs(split, margins, PATHS, baseline, cks, bad_ms, 2, exclusion=ex)
        with pytest.raises(ValueError, match="B1 analysis exclusion"):  # never relaxed
            validate_inputs(split, margins, PATHS, baseline, cks, bad_ms, 2, allow_partial=True, exclusion=ex)
    with pytest.raises(ValueError, match="margins records B1 analysis exclusion sha256 None"):  # margins from before the exclusion
        validate_inputs(split, margins_, PATHS, baseline, cks, ms, 2, exclusion=ex)
    with pytest.raises(ValueError, match="B1 analysis exclusion sha256 None"):  # baseline from before the exclusion
        validate_inputs(split, margins, PATHS, base_, cks, ms, 2, exclusion=ex)
    # robustness margins in a gate run and gate margins in a robustness run are refused
    with pytest.raises(ValueError, match="margins was made with the B1 overlap"):
        validate_inputs(split, _rec(margins, dict(ex, applied=False)), PATHS, baseline, cks, ms, 2, exclusion=ex)
    with pytest.raises(ValueError, match="margins was made without the B1 overlap"):
        validate_inputs(split, margins, PATHS, baseline, cks, ms, 2, exclusion=dict(ex, applied=False))


def test_g1_report_records_exclusion_and_robustness_view(g1_setup):
    ex = _excl(sorted(g1_setup[0]["unseen"])[:5])
    split, baseline, margins, cfg, cks, ms = _gate_case(g1_setup, ex)
    r = g1_report(split, margins, baseline, cks, ms, cfg, exclusion=ex)
    assert r["gate_valid"] is True and r["robustness"] is False and r["analysis_exclusion"]["n_excluded"]["unseen"] == 5
    assert r["cells"]["Paired/d4"]["pooled"]["n_unseen"] == len(split["unseen"]) - 5
    inc = dict(ex, applied=False)
    rr = g1_report(split, _rec(margins, inc), baseline, cks, ms, cfg, exclusion=inc)
    assert rr["gate_valid"] is False and rr["robustness"] is True and rr["cells"]["Paired/d4"]["pooled"]["n_unseen"] == len(split["unseen"])
    assert rr["criteria"]["b"]["verdict"].startswith("ROBUSTNESS (B1 overlap included)")


def test_g2_requires_the_g1_report_exclusion():
    split, margins, g1, bp, base, cks, cms, cfg = make_g2()
    ex = _excl(sorted(split["seen"])[:3] + sorted(split["unseen"])[:2])
    margins, g1 = _rec(margins, ex), _rec(g1, ex)
    base, cms = [_rec(m, ex) for m in base], [_rec(m, ex) for m in cms]
    validate_g2_inputs(split, margins, "/tmp/g2/margins.json", g1, bp, base, cks, cms, cfg, exclusion=ex)
    r = g2_report(split, margins, g1, base, cks, cms, cfg, exclusion=ex)
    assert r["gate_valid"] is True and r["verdict"] == "PASS" and r["analysis_exclusion"]["n_excluded"]["seen"] == 3
    for bad in (_rec(g1, _excl(ex["image_ids"], sha="F")), {k: v for k, v in g1.items() if k != "analysis_exclusion"}, _rec(g1, dict(ex, applied=False))):
        with pytest.raises(ValueError, match="the G1 report"):
            validate_g2_inputs(split, margins, "/tmp/g2/margins.json", bad, bp, base, cks, cms, cfg, exclusion=ex)
    inc = dict(ex, applied=False)
    rg1 = dict(_rec(g1, inc), gate_valid=False, robustness=True, robustness_valid=True)
    rg1["criteria"] = {"a": dict(g1["criteria"]["a"], verdict="ROBUSTNESS (B1 overlap included), not a gate decision: PASS")}
    rm = dict(_rec(margins, inc), gate_valid=False, robustness=True, robustness_valid=True)
    validate_g2_inputs(split, rm, "/tmp/g2/margins.json", rg1, bp, base, cks, cms, cfg, exclusion=inc)
    rr = g2_report(split, rm, rg1, base, cks, cms, cfg, exclusion=inc)
    assert rr["gate_valid"] is False and rr["robustness"] is True and rr["verdict"].startswith("ROBUSTNESS")


def test_margins_and_gate_report_cli(tmp_path, g1_setup):
    split, baseline, _, _, ckpt = g1_setup
    cfg_path, lock, split_path, split_sha = locked_split(tmp_path, split, n_boot=100)
    out_ids = sorted(split["seen"])[:2] + sorted(split["unseen"])[:3]
    (tmp_path / "excl.txt").write_text("# test exclusion\n" + "\n".join(map(str, out_ids)) + "\n")
    raw = yaml.safe_load(cfg_path.read_text())
    raw["analysis"]["exclude_image_ids_file"] = str(tmp_path / "excl.txt")
    cfg_path.write_text(yaml.safe_dump(raw))
    ex = load_analysis_exclusion(load_config(cfg_path))
    d0 = [str(tmp_path / f"d0_s{s}.json") for s in range(5)]
    for p, m in zip(d0, baseline):
        write_json(dict(m, manifest_split_sha256=split_sha, analysis_exclusion=ex, seen_definition=seen_rec(split)), p)
    write_json(base_metrics(baseline[0], manifest_split_sha256=split_sha, analysis_exclusion=ex, seen_definition=seen_rec(split)), tmp_path / "base.json")
    common = ["--config", str(cfg_path), "--lock", str(lock)]
    margins_cmd = [sys.executable, str(REPO_ROOT / "scripts/compute_margins.py"), *common, "--d0", *d0, "--base", str(tmp_path / "base.json"), "--out"]

    def run(cmd):
        return subprocess.run(cmd, capture_output=True, text=True)

    r = run(margins_cmd + [str(tmp_path / "m.json")])
    assert r.returncode == 0, r.stderr
    m = read_json(tmp_path / "m.json")
    kept = [i for i in split["unseen"] if i not in out_ids]
    want = [np.mean([b["per_image"][str(i)]["word_count"] for i in kept]) for b in baseline]
    assert m["per_seed"]["unseen_mean/word_count"] == pytest.approx(want)
    assert m["gate_valid"] is True and m["robustness"] is False and m["analysis_exclusion"]["sha256"] == ex["sha256"]
    assert m["analysis_exclusion"]["n_excluded"]["seen"] == 2 and m["analysis_exclusion"]["n_excluded"]["unseen"] == 3
    r = run(margins_cmd + [str(tmp_path / "mr.json"), "--include-b1-overlap"])
    assert r.returncode == 0, r.stderr
    mr = read_json(tmp_path / "mr.json")
    assert mr["gate_valid"] is False and mr["robustness"] is True and mr["robustness_valid"] is True
    assert mr["per_seed"]["unseen_mean/word_count"] == pytest.approx([np.mean([b["per_image"][str(i)]["word_count"] for i in split["unseen"]]) for b in baseline])

    cks = []
    for s in range(2):
        p = tmp_path / f"p{s}.json"
        write_json(dict(ckpt(f"p{s}", 2.0), manifest_split_sha256=split_sha, analysis_exclusion=ex, seen_definition=seen_rec(split)), p)
        cks.append({"condition": "Paired", "d": 4, "seed": s, "metrics": str(p)})

    def gate(margins, *extra):
        (tmp_path / "man.yaml").write_text(yaml.safe_dump({"split": str(split_path), "margins": str(margins), "baseline": d0, "checkpoints": cks}))
        return run([sys.executable, str(REPO_ROOT / "scripts/gate_report.py"), *common, "--manifest", str(tmp_path / "man.yaml"), "--out", str(tmp_path / "g1.json"), *extra])

    r = gate(tmp_path / "m.json")
    assert r.returncode == 0, r.stderr
    g1 = read_json(tmp_path / "g1.json")
    assert g1["gate_valid"] is True and g1["analysis_exclusion"]["image_ids"] == sorted(out_ids)
    assert g1["cells"]["Paired/d4"]["pooled"]["n_unseen"] == len(kept)
    assert g1["n_boot"] == 100 and g1["cells"]["Paired/d4"]["pooled"]["S"]["word_count"]["unseen"]["n_boot"] == 100
    r = gate(tmp_path / "m.json", "--n-boot", "150")  # explicit B overrides stats.n_boot and is recorded
    assert r.returncode == 0, r.stderr
    g1b = read_json(tmp_path / "g1.json")
    assert g1b["n_boot"] == 150 and g1b["cells"]["Paired/d4"]["pooled"]["S"]["word_count"]["unseen"]["n_boot"] == 150 and g1b["gate_valid"] is True
    r = gate(tmp_path / "mr.json")
    assert r.returncode != 0 and "margins was made with the B1 overlap" in r.stderr
    r = gate(tmp_path / "m.json", "--include-b1-overlap")
    assert r.returncode != 0 and "margins was made without the B1 overlap" in r.stderr
    r = gate(tmp_path / "mr.json", "--include-b1-overlap")
    assert r.returncode == 0, r.stderr
    g1r = read_json(tmp_path / "g1.json")
    assert g1r["gate_valid"] is False and g1r["robustness"] is True and g1r["robustness_valid"] is True and "ROBUSTNESS" in r.stdout
    assert g1r["cells"]["Paired/d4"]["pooled"]["n_unseen"] == len(split["unseen"])

    # metrics written before the exclusion (no record) are refused by the gate and by the margins
    old = {k: v for k, v in read_json(cks[1]["metrics"]).items() if k != "analysis_exclusion"}
    write_json(old, cks[1]["metrics"])
    r = gate(tmp_path / "m.json")
    assert r.returncode != 0 and "records B1 analysis exclusion sha256 None" in r.stderr
    write_json({k: v for k, v in read_json(d0[0]).items() if k != "analysis_exclusion"}, d0[0])
    r = run(margins_cmd + [str(tmp_path / "m2.json")])
    assert r.returncode != 0 and "records B1 analysis exclusion sha256 None" in r.stderr


def test_compute_margins_refuses_a_base_from_another_prompt(tmp_path, g1_setup):
    split, baseline, *_ = g1_setup
    cfg_path, lock, split_path, split_sha = locked_split(tmp_path, split)
    ex = load_analysis_exclusion(load_config(cfg_path))
    d0 = [str(tmp_path / f"d0_s{s}.json") for s in range(5)]
    for p, m in zip(d0, baseline):
        write_json(dict(m, manifest_split_sha256=split_sha, analysis_exclusion=ex, seen_definition=seen_rec(split)), p)
    base = base_metrics(baseline[0], manifest_split_sha256=split_sha, analysis_exclusion=ex, seen_definition=seen_rec(split))
    cmd = [sys.executable, str(REPO_ROOT / "scripts/compute_margins.py"), "--config", str(cfg_path), "--lock", str(lock), "--d0", *d0, "--base", str(tmp_path / "base.json"),
           "--out", str(tmp_path / "m.json")]
    write_json(base, tmp_path / "base.json")
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = read_json(tmp_path / "m.json")
    assert m["base_cider_check"]["cider_all10"] == 100.0 and m["base_check"] == "present" and m["gate_valid"] is True
    assert m["base_file"] == str((tmp_path / "base.json").resolve())
    old_prompt = dict(base, provenance={"generations": dict(base["provenance"]["generations"], prompt_sha="old p_cap")})
    r_msg = "regenerate and re-evaluate the base under the current p_cap"
    e_msg = "re-evaluate the base"
    no_record = {k: v for k, v in base.items() if k != "analysis_exclusion"}
    for bad, msg, fix in ((old_prompt, "generated with prompt_sha 'old p_cap'", r_msg), (no_record, "records B1 analysis exclusion sha256 None", e_msg),
                          (dict(base, complete=False), "partial image set", None), (dict(base, split={"name": "A", "content_hash": "other"}), "was not evaluated on", None),
                          (dict(base, manifest_split_sha256="stale"), "checked against split sha256 stale", e_msg)):
        write_json(bad, tmp_path / "base.json")
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode != 0 and msg in r.stderr, (msg, r.stderr)
        assert fix is None or fix in r.stderr
        if bad is no_record:
            assert "regenerate" not in r.stderr  # only the evaluation is stale
    write_json(dict(base, complete=False), tmp_path / "base.json")  # smoke tests may use a partial base
    r = subprocess.run(cmd + ["--allow-partial"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_margins_without_a_base_are_not_gate_valid(tmp_path, g1_setup):
    split, baseline, margins, cfg, ckpt = g1_setup
    cfg_path, lock, split_path, split_sha = locked_split(tmp_path, split)
    ex = load_analysis_exclusion(load_config(cfg_path))
    d0 = [str(tmp_path / f"d0_s{s}.json") for s in range(5)]
    for p, m in zip(d0, baseline):
        write_json(dict(m, manifest_split_sha256=split_sha, analysis_exclusion=ex, seen_definition=seen_rec(split)), p)
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/compute_margins.py"), "--config", str(cfg_path), "--lock", str(lock), "--d0", *d0,
                        "--out", str(tmp_path / "m.json")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "base CIDEr check is missing; margins are NOT gate-valid" in r.stderr
    m = read_json(tmp_path / "m.json")
    assert m["gate_valid"] is False and m["base_check"] == "missing" and "base_cider_check" not in m and m["delta_S"]

    # the gate refuses gate-valid margins without the base check (e.g. written before this rule) unless --allow-partial
    cks = [{"condition": "Paired", "d": 4, "seed": s, "metrics": f"/tmp/p{s}.json"} for s in range(2)]
    ms = [ckpt(f"p{s}", 2.0) for s in range(2)]
    old = {k: v for k, v in margins.items() if k != "base_cider_check"}
    with pytest.raises(ValueError, match="no G0 base CIDEr check"):
        validate_inputs(split, old, PATHS, baseline, cks, ms, 2)
    validate_inputs(split, old, PATHS, baseline, cks, ms, 2, allow_partial=True)
    validate_inputs(split, dict(margins, base_cider_check=dict(margins["base_cider_check"], in_band=False)), PATHS, baseline, cks, ms, 2)  # the band is a warning


def test_compute_margins_refuses_an_empty_group(tmp_path, g1_setup):
    split, baseline, *_ = g1_setup
    cfg_path, lock, split_path, split_sha = locked_split(tmp_path, split)
    (tmp_path / "excl.txt").write_text("\n".join(map(str, split["unseen"])) + "\n")  # every unseen image excluded
    raw = yaml.safe_load(cfg_path.read_text())
    raw["analysis"]["exclude_image_ids_file"] = str(tmp_path / "excl.txt")
    cfg_path.write_text(yaml.safe_dump(raw))
    ex = load_analysis_exclusion(load_config(cfg_path))
    d0 = [str(tmp_path / f"d0_s{s}.json") for s in range(5)]
    for p, m in zip(d0, baseline):
        write_json(dict(m, manifest_split_sha256=split_sha, analysis_exclusion=ex, seen_definition=seen_rec(split)), p)
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/compute_margins.py"), "--config", str(cfg_path), "--lock", str(lock), "--d0", *d0,
                        "--out", str(tmp_path / "m.json")], capture_output=True, text=True)
    assert r.returncode != 0 and "no unseen images left to average" in r.stderr and "Traceback" not in r.stderr
