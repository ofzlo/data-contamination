"""Eval-time manifest lock check (contam.data.freeze.check_split_frozen) and its CLI wiring."""

import json
import shutil
import subprocess
import sys

import pytest
import yaml

from contam.config import REPO_ROOT, file_sha256, load_config, read_json, write_json
from contam.data import freeze
from contam.data.splits import make_split
from contam.eval.gates import validate_inputs
from tests.conftest import base_metrics, excl_record, locked_split, make_annotations, seen_rec


@pytest.fixture
def locked(tmp_path):
    ann = make_annotations(2, 4, 2, seed=5)
    split = make_split(ann, 1, 2, name="A")
    write_json({"meta": {}, "images": ann}, tmp_path / "ann.json")
    cfg_path, lock, split_path, split_sha = locked_split(tmp_path, split)
    return ann, split, load_config(cfg_path), cfg_path, lock, split_path, split_sha


def test_locked_split_accepted_and_shas_returned(locked):
    *_, cfg, _, lock, split_path, split_sha = locked
    want = {"manifest_split_sha256": split_sha, "manifest_lock_sha256": file_sha256(lock)}
    assert freeze.check_split_frozen(cfg, split_path, lock) == want and split_sha == read_json(lock)["files"]["split_A.json"]["sha256"]
    shutil.copy(split_path, split_path.parent.parent / "split_A.json")  # identical bytes elsewhere: same data
    assert freeze.check_split_frozen(cfg, split_path.parent.parent / "split_A.json", lock) == want


def test_unlocked_split_rejected(locked):
    ann, _, cfg, _, lock, split_path, _ = locked
    other = split_path.parent / "split_B.json"
    write_json(make_split(ann, 3, 4, name="B"), other)
    with pytest.raises(RuntimeError, match="split_B.json is not in manifest lock"):
        freeze.check_split_frozen(cfg, other, lock)


def test_modified_split_rejected(locked, tmp_path):
    _, split, cfg, _, lock, split_path, _ = locked
    copy = tmp_path / "elsewhere" / "split_A.json"
    write_json(dict(split, seen=split["seen"][1:]), copy)  # same name, different bytes
    with pytest.raises(RuntimeError, match="does not match the locked split_A.json"):
        freeze.check_split_frozen(cfg, copy, lock)
    write_json(split, split_path, indent=None)  # the locked file itself re-serialized
    with pytest.raises(RuntimeError, match="changed split_A.json"):
        freeze.check_split_frozen(cfg, split_path, lock)


def test_seed_config_change_rejected(locked):
    *_, cfg, _, lock, split_path, _ = locked
    cfg["splits"]["A"]["split_seed"] += 1
    with pytest.raises(RuntimeError, match="seed config"):
        freeze.check_split_frozen(cfg, split_path, lock)


def _write_gens(path, ann):
    with open(path, "w") as f:
        for a in ann:
            f.write(json.dumps(dict(checkpoint_id="ck", model="m", adapter=None, prompt_sha="P", image_id=a["image_id"], prompt_id="caption", temperature=0.0, sample_idx=0, seed=None, caption=a["refs"][0])) + "\n")


def test_evaluate_checkpoint_cli_check_and_skip(locked, tmp_path):
    ann, split, _, cfg_path, lock, split_path, _ = locked
    _write_gens(tmp_path / "g.jsonl", ann)
    cmd = [sys.executable, "scripts/evaluate_checkpoint.py", "--config", str(cfg_path), "--lock", str(lock), "--annotations", str(tmp_path / "ann.json"), "--checkpoint-id", "ck",
           "--generations", str(tmp_path / "g.jsonl"), "--tokenizer", "simple", "--metrics", "cider", "--no-spice-cache", "--out", str(tmp_path / "m.json")]
    write_json(dict(split, seen=split["seen"][1:]), tmp_path / "x" / "split_A.json")
    r = subprocess.run(cmd + ["--split", str(tmp_path / "x" / "split_A.json")], capture_output=True, text=True)
    assert r.returncode != 0 and "does not match the locked split_A.json" in r.stderr and not (tmp_path / "m.json").exists()
    r = subprocess.run(cmd + ["--split", str(split_path), "--skip-manifest-check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = read_json(tmp_path / "m.json")
    assert m["complete"] is True and m["manifest_split_sha256"] is None and m["manifest_lock_sha256"] is None and m["gate_valid"] is False


def test_validate_inputs_split_shas(g1_setup):
    split, baseline, margins, cfg, ckpt = g1_setup
    paths = [f"/tmp/b{s}.json" for s in range(5)]
    cks = [{"condition": "Paired", "d": 4, "seed": s, "metrics": f"/tmp/p{s}.json"} for s in range(2)]
    ok = [ckpt("p0", 1.0), ckpt("p1", 1.0)]
    ms = [ok[0], dict(ok[1], manifest_split_sha256="L2")]
    with pytest.raises(ValueError, match="share one recorded manifest_split_sha256"):
        validate_inputs(split, margins, paths, baseline, cks, ms, 2)
    with pytest.raises(ValueError, match="share one recorded manifest_split_sha256"):
        validate_inputs(split, dict(margins, manifest_split_sha256=None), paths, baseline, cks, ok, 2)
    validate_inputs(split, margins, paths, baseline, cks, ok, 2, split_sha="L")
    with pytest.raises(ValueError, match="current manifest lock entry is L9"):
        validate_inputs(split, margins, paths, baseline, cks, ok, 2, split_sha="L9")
    validate_inputs(split, margins, paths, baseline, cks, ms, 2, allow_partial=True)


def _gate_inputs(tmp_path, g1_setup, split_sha):
    split, baseline, margins, _, ckpt = g1_setup
    paths = [str(tmp_path / f"d0_s{s}.json") for s in range(5)]
    for p, m in zip(paths, baseline):
        write_json(dict(m, manifest_split_sha256=split_sha, analysis_exclusion=excl_record(), seen_definition=seen_rec(split)), p)
    write_json(dict(margins, d0_files=paths, manifest_split_sha256=split_sha, analysis_exclusion=excl_record(), seen_definition=seen_rec(split)), tmp_path / "margins.json")
    cks = []
    for s in range(2):
        write_json(dict(ckpt(f"p{s}", 2.0), manifest_split_sha256=split_sha, analysis_exclusion=excl_record(), seen_definition=seen_rec(split)), tmp_path / f"p{s}.json")
        cks.append({"condition": "Paired", "d": 4, "seed": s, "metrics": str(tmp_path / f"p{s}.json")})
    return {"margins": str(tmp_path / "margins.json"), "baseline": paths, "checkpoints": cks}


def test_gate_report_cli_split_sha(tmp_path, g1_setup):
    split = g1_setup[0]
    cfg_path, lock, split_path, split_sha = locked_split(tmp_path, split, n_boot=200)
    man = dict(_gate_inputs(tmp_path, g1_setup, split_sha), split=str(split_path))
    (tmp_path / "man.yaml").write_text(yaml.safe_dump(man))
    cmd = [sys.executable, str(REPO_ROOT / "scripts/gate_report.py"), "--config", str(cfg_path), "--lock", str(lock), "--manifest", str(tmp_path / "man.yaml"), "--out", str(tmp_path / "r.json")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = read_json(tmp_path / "r.json")
    assert rep["gate_valid"] is True and rep["manifest_split_sha256"] == split_sha and rep["manifest_lock_sha256"] == file_sha256(lock)

    # an --add (e.g. general_order.jsonl before G0 training) changes the lock file but not the split
    # entry: metrics evaluated before it still gate together after it
    old_lock_sha = file_sha256(lock)
    (split_path.parent / "general_order.jsonl").write_text('{"order": 0}\n')
    freeze.write_lock(load_config(cfg_path), lock, add=True)
    assert file_sha256(lock) != old_lock_sha
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = read_json(tmp_path / "r.json")
    assert rep["gate_valid"] is True and rep["manifest_split_sha256"] == split_sha and rep["manifest_lock_sha256"] == file_sha256(lock)

    r = subprocess.run(cmd + ["--skip-manifest-check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = read_json(tmp_path / "r.json")
    assert rep["gate_valid"] is False and rep["manifest_split_sha256"] is None and rep["manifest_lock_sha256"] is None

    write_json(dict(read_json(man["checkpoints"][1]["metrics"]), manifest_split_sha256="other"), man["checkpoints"][1]["metrics"])
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0 and "share one recorded manifest_split_sha256" in r.stderr

    _gate_inputs(tmp_path, g1_setup, "stale")  # all inputs agree, but not with the current lock entry
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0 and f"current manifest lock entry is {split_sha}" in r.stderr


def test_compute_margins_cli_split_sha(tmp_path, g1_setup):
    split, baseline, *_ = g1_setup
    cfg_path, lock, split_path, split_sha = locked_split(tmp_path, split)
    paths = [str(tmp_path / f"d0_s{s}.json") for s in range(5)]

    def write_d0(sha):
        for k, (p, m) in enumerate(zip(paths, baseline)):
            summ = {g: {"word_count": 10.0 + k + i, "boilerplate": 0.0} for i, g in enumerate(("seen", "unseen", "all"))}
            write_json(dict(m, summary=summ, manifest_split_sha256=sha, analysis_exclusion=excl_record(), seen_definition=seen_rec(split)), p)

    write_d0(split_sha)
    write_json(base_metrics(baseline[0], manifest_split_sha256=split_sha, analysis_exclusion=excl_record(), seen_definition=seen_rec(split)), tmp_path / "base.json")
    cmd = [sys.executable, str(REPO_ROOT / "scripts/compute_margins.py"), "--config", str(cfg_path), "--lock", str(lock), "--out", str(tmp_path / "m.json"),
           "--base", str(tmp_path / "base.json"), "--d0"] + paths
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = read_json(tmp_path / "m.json")
    assert m["gate_valid"] is True and m["manifest_split_sha256"] == split_sha and m["manifest_lock_sha256"] == file_sha256(lock)
    r = subprocess.run(cmd + ["--skip-manifest-check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = read_json(tmp_path / "m.json")
    assert m["gate_valid"] is False and m["manifest_split_sha256"] is None
    write_d0("stale")  # shared, but not the current lock entry
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0 and f"current lock entry for split_A.json is {split_sha}" in r.stderr
    write_d0(split_sha)
    write_json(dict(read_json(paths[0]), manifest_split_sha256=None), paths[0])  # d=0 run evaluated without the check
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode != 0 and "manifest_split_sha256" in r.stderr


def test_b3_report_gate_valid(locked, tmp_path):
    ann, split, _, cfg_path, lock, split_path, split_sha = locked
    raw = yaml.safe_load(cfg_path.read_text())
    raw["paths"]["annotations"] = str(tmp_path / "ann.json")
    out_ids = sorted(split["seen"])[:2] + sorted(split["unseen"])[:1]
    (tmp_path / "excl.txt").write_text("\n".join(map(str, out_ids)) + "\n")
    raw["analysis"]["exclude_image_ids_file"] = str(tmp_path / "excl.txt")
    cfg_path.write_text(yaml.safe_dump(raw))
    b3_ids = yaml.safe_load((REPO_ROOT / "configs" / "prompts.yaml").read_text())["b3"]["prompt_ids"]
    (tmp_path / "b3").mkdir()
    with open(tmp_path / "b3" / "base.jsonl", "w") as f:
        for pid in b3_ids:
            for a in ann:
                f.write(json.dumps(dict(checkpoint_id="base", model="m", adapter=None, prompt_sha=f"P{pid}", image_id=a["image_id"], prompt_id=pid, temperature=0.0, sample_idx=0, seed=None, caption=a["refs"][0])) + "\n")
    cmd = [sys.executable, str(REPO_ROOT / "scripts/b3_prompt_intervention.py"), "evaluate", "--config", str(cfg_path), "--lock", str(lock), "--split", str(split_path),
           "--model-tag", "base", "--out-dir", str(tmp_path / "b3"), "--tokenizer", "simple", "--metrics", "cider"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = read_json(tmp_path / "b3" / "b3_report.json")
    assert rep["gate_valid"] is True and rep["manifest_split_sha256"] == split_sha
    assert rep["analysis_exclusion"]["image_ids"] == sorted(out_ids) and rep["analysis_exclusion"]["applied"] is True
    assert rep["n_analysed"] == rep["n_images"] - len(out_ids)
    r = subprocess.run(cmd + ["--include-b1-overlap"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = read_json(tmp_path / "b3" / "b3_report.json")
    assert rep["gate_valid"] is False and rep["robustness"] is True and rep["robustness_valid"] is True and rep["n_analysed"] == rep["n_images"]
    r = subprocess.run(cmd + ["--skip-manifest-check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = read_json(tmp_path / "b3" / "b3_report.json")
    assert rep["gate_valid"] is False and rep["manifest_split_sha256"] is None
