"""G2 gate (contam.eval.gates.g2_report / validate_g2_inputs, scripts/g2_report.py) on synthetic dose curves."""

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from contam.config import REPO_ROOT, load_config, read_json, write_json
from contam.data.splits import make_split
from contam.eval import stats
from contam.eval.gates import g2_report, g2_text, parse_run_id, validate_g2_inputs
from tests.conftest import locked_split, make_annotations

DELTA_S, DELTA_V = 1.0, 0.5
PROV = {"generations": {"prompt_sha": "P", "model": "m", "batch_size": 16, "max_new_tokens": 300}}
MARGINS = "/tmp/g2/margins.json"
STYLE_AT_4 = {1: 0.2, 4: 3.0, 8: 3.5, 16: 4.0, 64: 5.0}  # crosses delta_S at d=4
MEMORY_AT_64 = {1: 0.0, 4: 0.0, 8: 0.0, 16: 0.05, 64: 3.0}  # crosses delta_V at d=64


def _metrics(split, name, wc, rmi, sha="L"):
    per = {str(i): {"word_count": float(wc[i]), "rmi_sum": float(rmi[i]), "boilerplate": 0.0, "exact_match_train": 0.0} for i in wc}
    return {"checkpoint_id": name, "split": {"name": split["name"], "content_hash": split["content_hash"]}, "complete": True, "generations_hash": name,
            "manifest_split_sha256": sha, "provenance": copy.deepcopy(PROV), "per_image": per}


def make_g2(style=STYLE_AT_4, memory=MEMORY_AT_64, doses=(1, 4, 16, 64), n_seeds=3, sha="L", mem_seed_shift=(0.0, 0.0, 0.0)):
    """Synthetic G2 inputs: Caption has half of Paired's style effect and no memory effect.

    ``mem_seed_shift[s]`` is added to Paired seed s's memory effect (training-seed variance).
    """
    ann = make_annotations(10, 40, 20, seed=9)
    split = make_split(ann, 1, 2)
    ids = sorted(split["seen"] + split["unseen"])
    rng = np.random.default_rng(0)
    wc0 = {i: 30 + rng.normal(0, 3) for i in ids}
    rmi0 = {i: rng.normal(0, 1) for i in ids}

    def run(name, s_eff, m_eff):  # word count drops by s_eff (positive directional effect); rmi rises by m_eff
        return _metrics(split, name, {i: w - s_eff + rng.normal(0, 0.3) for i, w in wc0.items()}, {i: r + m_eff + rng.normal(0, 0.3) for i, r in rmi0.items()}, sha)

    baseline = [run(f"G0_d0_s{s}", 0.0, 0.0) for s in range(5)]
    base_paths = [f"/tmp/g2/G0_d0_s{s}.json" for s in range(5)]
    margins = {"gate_valid": True, "split": baseline[0]["split"], "d0_files": base_paths, "manifest_split_sha256": sha, "delta_S": {"word_count": DELTA_S, "boilerplate": 0.01}}
    g1 = {"split": split["name"], "split_hash": split["content_hash"], "gate_valid": True, "manifest_split_sha256": sha, "baseline_files": base_paths, "margins_file": MARGINS,
          "criteria": {"a": {"verdict": "PASS", "delta_V": {"rmi_sum": DELTA_V, "rmi_norm": 0.02}}}}
    cks, cms = [], []
    for cond, sf, mf in (("Caption", 0.5, 0.0), ("Paired", 1.0, 1.0)):
        for d in doses:
            for s in range(n_seeds):
                name = f"{cond.lower()}_d{d}_s{s}"
                cks.append({"condition": cond, "d": d, "seed": s, "metrics": f"/tmp/g2/{name}.json"})
                cms.append(run(name, sf * style[d], mf * (memory[d] + mem_seed_shift[s])))
    for s in range(n_seeds):
        cks.append({"condition": "SameDist", "d": 16, "seed": s, "metrics": f"/tmp/g2/samedist_d16_s{s}.json"})
        cms.append(run(f"samedist_d16_s{s}", 0.5 * style[16], 0.0))
    cfg = load_config()
    cfg["stats"]["n_boot"] = 300
    return split, margins, g1, base_paths, baseline, cks, cms, cfg


def _run(inputs, **kw):
    split, margins, g1, base_paths, baseline, cks, cms, cfg = inputs
    validate_g2_inputs(split, margins, MARGINS, g1, base_paths, baseline, cks, cms, cfg, **kw)
    return g2_report(split, margins, g1, baseline, cks, cms, cfg, kw.get("allow_partial", False))


def test_parse_run_id():
    assert parse_run_id("paired_d64_s0") == ("Paired", 64, 0)
    assert parse_run_id("G2_Caption_d8_s2") == ("Caption", 8, 2)
    assert parse_run_id("samedist_d16_s1") == ("SameDist", 16, 1)
    assert parse_run_id("G0_d0_s3") is None and parse_run_id("foo_d4_s0") is None and parse_run_id(None) is None
    for bad in ("paired_d64_s0\n", "G2_paired_d064_s0", "paired_d64_s01", "paired_d64_s0x", "same_dist_d16_s0"):
        assert parse_run_id(bad) is None, bad


def test_style_first_passes():
    r = _run(make_g2())
    assert r["d_style"] == 4 and r["d_memory"] == 64
    p, s = r["criteria"]["primary"], r["criteria"]["secondary"]
    assert p["verdict"] == "PASS" and p["d_style"] == 4 and p["v_effect"]["ci_hi"] < DELTA_V
    assert s["verdict"] == "PASS" and s["ci_lo"] > 0
    assert r["verdict"] == "PASS" and r["gate_valid"] is True and r["fallback"] is None and not r["same_start"]
    cap = r["conditions"]["Caption"]
    assert cap["style"]["tau"] == 4 and cap["memory"]["tau"] == "inf"  # half the style effect, no memory: reported only
    assert r["samedist"]["d16"]["n_seeds"] == 3
    text = g2_text(r)
    assert "Primary   PASS" in text and "Secondary PASS" in text and "d_style = d=4; d_memory = d=64" in text


def test_same_start_fails_secondary_and_reports_fallback():
    r = _run(make_g2(memory=STYLE_AT_4))
    assert r["d_style"] == r["d_memory"] == 4 and r["same_start"]
    assert r["criteria"]["secondary"]["verdict"] == "FAIL"
    assert r["criteria"]["primary"]["verdict"] == "FAIL"  # V effect at d_style is ~3 > delta_V
    assert "skip C3" in r["fallback"] and "Fallback:" in g2_text(r)


def test_no_d_style_fails_primary():
    r = _run(make_g2(style={d: 0.1 for d in STYLE_AT_4}))
    assert r["d_style"] == "inf"
    p = r["criteria"]["primary"]
    assert p["verdict"] == "FAIL" and p["note"].startswith("no d_style on this grid") and "primary not evaluable" in p["note"] and "v_effect" not in p
    assert "skip C3" not in p["note"] and r["fallback"] is None
    assert r["verdict"] == "FAIL" and "never crosses (+inf)" in g2_text(r)


def test_v_effect_at_d_style_above_delta_v_fails_primary():
    r = _run(make_g2(memory={**MEMORY_AT_64, 4: 0.6, 8: 0.6, 16: 0.6}))
    p = r["criteria"]["primary"]
    assert r["d_style"] == 4 and p["verdict"] == "FAIL" and p["v_effect"]["ci_hi"] >= DELTA_V


def test_optional_d8_is_part_of_the_grid():
    r = _run(make_g2(doses=(1, 4, 8, 16, 64)))
    assert r["conditions"]["Paired"]["doses"] == [1, 4, 8, 16, 64] and r["conditions"]["Paired"]["style"]["doses"] == [1, 4, 8, 16, 64]
    assert r["d_style"] == 4 and r["criteria"]["primary"]["verdict"] == "PASS"


def test_equivalence_rules_under_seed_variance():
    # pooled V effect 0.2 (CI well below delta_V = 0.5), but seed 2's own estimate is 0.6
    inputs = make_g2(memory={**MEMORY_AT_64, 4: 0.2, 8: 0.2, 16: 0.2}, mem_seed_shift=(-0.3, -0.1, 0.4))
    r = _run(inputs)
    p = r["criteria"]["primary"]
    assert p["equivalence_rule"] == "two_level_bootstrap" and r["d_style"] == 4  # the default
    assert p["rules"]["pooled"]["pass"] and p["rules"]["pooled"]["ci_hi"] < DELTA_V
    assert max(p["per_seed_estimates"]) > DELTA_V and not p["rules"]["pooled_and_every_seed"]["pass"]
    assert not p["rules"]["two_level_bootstrap"]["pass"] and p["rules"]["two_level_bootstrap"]["ci_hi"] >= DELTA_V and not p["rules"]["seed_t"]["pass"]
    assert p["verdict"] == "FAIL" and r["verdict"] == "FAIL"
    assert "* two_level_bootstrap" in g2_text(r)
    inputs[-1]["gates"]["g2_equivalence_rule"] = "pooled_and_every_seed"  # (a) is still selectable
    p = _run(inputs)["criteria"]["primary"]
    assert p["equivalence_rule"] == "pooled_and_every_seed" and p["verdict"] == "FAIL" and "every seed" in p["rule"]
    inputs[-1]["gates"]["g2_equivalence_rule"] = "pooled"  # the image-bootstrap-only rule would pass it
    assert _run(inputs)["criteria"]["primary"]["verdict"] == "PASS"
    inputs[-1]["gates"]["g2_equivalence_rule"] = "pooled_only"
    with pytest.raises(ValueError, match="g2_equivalence_rule"):
        _run(inputs)
    # no seed variance: every rule passes
    p = _run(make_g2(memory={**MEMORY_AT_64, 4: 0.2, 8: 0.2, 16: 0.2}))["criteria"]["primary"]
    assert all(v["pass"] for v in p["rules"].values()) and p["verdict"] == "PASS"


def test_two_level_bootstrap_and_seed_t():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, (5, 200))
    same = stats.bootstrap_two_level(base[:3] + 0.5, base, n_boot=400)
    assert abs(same.estimate - (0.5 + base[:3].mean() - base.mean())) < 1e-12 and same.lo < same.estimate < same.hi
    shifted = np.vstack([base[0] - 1.0, base[1], base[2] + 1.0])  # seed variance widens the two-level CI
    assert stats.bootstrap_two_level(shifted, base, n_boot=400).hi - stats.bootstrap_two_level(shifted, base, n_boot=400).lo > 4 * (same.hi - same.lo)
    t = stats.seed_t_interval([0.1, 0.2, 0.3])
    assert t["estimate"] == pytest.approx(0.2) and t["df"] == 2 and t["ci_hi"] == pytest.approx(0.2 + 4.302652729749462 * 0.1 / np.sqrt(3))


def test_validation_failures():
    split, margins, g1, bp, base, cks, cms, cfg = make_g2()

    def check(match, **over):
        args = dict(split=split, margins=margins, margins_path=MARGINS, g1=g1, baseline_paths=bp, baseline=base, checkpoints=cks, cell_metrics=cms, cfg=cfg)
        args.update(over)
        with pytest.raises(ValueError, match=match):
            validate_g2_inputs(**args)

    keep = [k for k, c in enumerate(cks) if not (c["condition"] == "Paired" and c["d"] == 16 and c["seed"] == 2)]
    check("has 2 seeds, expected 3", checkpoints=[cks[k] for k in keep], cell_metrics=[cms[k] for k in keep])
    check("current manifest lock entry is other", split_sha="other")
    check("G1 report is not gate-valid", g1=dict(g1, gate_valid=False))
    check("G1\\(a\\) did not pass", g1=dict(g1, criteria={"a": dict(g1["criteria"]["a"], verdict="FAIL")}))
    check("G1 report was made under split sha256", g1=dict(g1, manifest_split_sha256="L2"))
    bad = copy.deepcopy(cms)
    bad[0]["provenance"]["generations"]["prompt_sha"] = "Q"
    check("share one recorded prompt_sha", cell_metrics=bad)
    keep = [k for k, c in enumerate(cks) if not (c["condition"] in ("Caption", "Paired") and c["d"] == 1)]
    check("missing d=\\[1\\]", checkpoints=[cks[k] for k in keep], cell_metrics=[cms[k] for k in keep])
    anon = [dict(m, checkpoint_id=f"run{k}") for k, m in enumerate(cms)]  # ids that do not name a cell
    check("not a G2 dose", checkpoints=[dict(c, d=2) if c["condition"] == "Paired" and c["d"] == 1 else c for c in cks], cell_metrics=anon)
    check("SameDist only at d=16", checkpoints=[dict(c, d=4) if c["condition"] == "SameDist" else c for c in cks], cell_metrics=anon)
    check("the G1 report used d=0 files", g1=dict(g1, baseline_files=["/other/d0_s0.json"]))
    check("the G1 report used margins", g1=dict(g1, margins_file="/other/margins.json"))
    swapped = list(cms)  # Paired d=64 metrics given as the Caption d=1 cell and vice versa
    ia = [k for k, c in enumerate(cks) if c["condition"] == "Caption" and c["d"] == 1]
    ib = [k for k, c in enumerate(cks) if c["condition"] == "Paired" and c["d"] == 64]
    for a, b in zip(ia, ib):
        swapped[a], swapped[b] = swapped[b], swapped[a]
    check("is \\('Paired', 64, 0\\), but it is given as \\('Caption', 1, 0\\)", cell_metrics=swapped)
    nolp = copy.deepcopy(cms)
    for row in nolp[0]["per_image"].values():
        del row["rmi_sum"]
    check("no per-image rmi_sum", cell_metrics=nolp)
    # --allow-partial relaxes the grid / seed / G1-validity rules (the report is then not gate-valid)
    keep = [k for k, c in enumerate(cks) if c["condition"] == "Paired"]
    with pytest.warns(UserWarning):
        r = _run((split, margins, dict(g1, gate_valid=False), bp, base, [cks[k] for k in keep], [cms[k] for k in keep], cfg), allow_partial=True)
    assert r["gate_valid"] is False and r["verdict"].startswith("NOT GATE-VALID")


def test_g2_report_cli_end_to_end(tmp_path):
    split, margins, g1, _, base, cks, cms, _ = make_g2()
    cfg_path, lock, split_path, split_sha = locked_split(tmp_path, split, n_boot=300)
    bp = []
    for m in base:
        bp.append(str(tmp_path / f"{m['checkpoint_id']}.json"))
        write_json(dict(m, manifest_split_sha256=split_sha), bp[-1])
    write_json(dict(margins, d0_files=bp, manifest_split_sha256=split_sha), tmp_path / "margins.json")
    write_json(dict(g1, manifest_split_sha256=split_sha, baseline_files=bp, margins_file=str(tmp_path / "margins.json")), tmp_path / "g1.json")
    named, samedist = [], []
    for c, m in zip(cks, cms):
        p = str(tmp_path / f"{m['checkpoint_id']}.json")
        if c["condition"] == "SameDist":  # pass these with --cell, under ids that do not name the cell
            m = dict(m, checkpoint_id=f"sd16_seed{c['seed']}")
            samedist.append(p)
        else:
            named.append(p)
        write_json(dict(m, manifest_split_sha256=split_sha), p)
    cmd = [sys.executable, str(REPO_ROOT / "scripts/g2_report.py"), "--config", str(cfg_path), "--lock", str(lock), "--split", str(split_path),
           "--margins", str(tmp_path / "margins.json"), "--g1-report", str(tmp_path / "g1.json"), "--baseline", *bp, "--checkpoints", *named,
           "--cell", "SameDist:16=" + ",".join(samedist), "--out", str(tmp_path / "g2.json")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = read_json(tmp_path / "g2.json")
    assert rep["gate_valid"] is True and rep["verdict"] == "PASS" and rep["manifest_split_sha256"] == split_sha
    assert rep["d_style"] == 4 and rep["d_memory"] == 64 and rep["samedist"]["d16"]["n_seeds"] == 3
    text = (tmp_path / "g2.txt").read_text()
    assert text.strip() in r.stdout and "G2 verdict (primary): PASS; gate_valid = True" in text
    print(text)

    r = subprocess.run(cmd + ["--skip-manifest-check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = read_json(tmp_path / "g2.json")
    assert rep["gate_valid"] is False and rep["manifest_split_sha256"] is None

    # --cell for metrics whose checkpoint_id names another cell / seed slot is refused
    p16 = [p for p in named if "/paired_d16_" in p]
    rest = [p for p in named if p not in p16]
    i = cmd.index("--checkpoints")
    bad = cmd[: i + 1] + rest + ["--cell", "SameDist:16=" + ",".join(samedist), "--cell", "Paired:16=" + ",".join(reversed(p16)), "--out", str(tmp_path / "g2.json")]
    r = subprocess.run(bad, capture_output=True, text=True)
    assert r.returncode != 0 and "'paired_d16_s2' is ('Paired', 16, 2), but it is given as ('Paired', 16, 0)" in r.stderr

    bad = cmd[: cmd.index("--cell")] + ["--checkpoints", samedist[0], "--out", str(tmp_path / "g2.json")]
    r = subprocess.run(bad, capture_output=True, text=True)
    assert r.returncode != 0 and "does not name a G2 cell" in r.stderr


def test_sim_g2_null_runs():
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/sim_g2_null.py"), "--n-images", "90", "--n-boot", "30", "--reps", "2", "--sigma-seed", "0.2", "--mult", "1.0"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "pooled_and_every_seed" in r.stdout and " 1.00       0.20 " in r.stdout


def test_g1_report_paths_are_realpaths_for_g2(tmp_path):
    """gate_report run from another cwd with relative paths; g2_report from the repo root accepts its G1 report."""
    split, margins, _, _, base, cks, cms, _ = make_g2()
    cfg_path, lock, split_path, split_sha = locked_split(tmp_path, split, n_boot=300)
    mdir, work = tmp_path / "m", tmp_path / "work"
    work.mkdir()

    def put(m, name, **over):  # G1(a) needs rmi_norm and a seen exact-match change at Paired d=64
        m = copy.deepcopy(m)
        for row in m["per_image"].values():
            row["rmi_norm"] = row["rmi_sum"] / 10
            row["exact_match_train"] = 1.0 if m["checkpoint_id"].startswith("paired_d64") else 0.0
        write_json(dict(m, manifest_split_sha256=split_sha, **over), mdir / name)
        return str(mdir / name)

    bp = [put(m, f"{m['checkpoint_id']}.json") for m in base]
    write_json(dict(margins, d0_files=bp, manifest_split_sha256=split_sha), mdir / "margins.json")
    paths = {m["checkpoint_id"]: put(m, f"{m['checkpoint_id']}.json") for m in cms}
    g1_cells = [c for c in cks if c["condition"] == "Paired" and c["d"] in (4, 64) and c["seed"] < 2]
    man = {"split": "../splits/" + split_path.name, "margins": "../m/margins.json", "baseline": [f"../m/{Path(p).name}" for p in bp],
           "checkpoints": [dict(c, metrics=f"../m/paired_d{c['d']}_s{c['seed']}.json") for c in g1_cells]}
    (work / "g1.yaml").write_text(yaml.safe_dump(man))
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/gate_report.py"), "--config", str(cfg_path), "--lock", str(lock), "--manifest", "g1.yaml", "--out", "g1.json"],
                       cwd=work, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    g1 = read_json(work / "g1.json")
    assert g1["gate_valid"] and g1["criteria"]["a"]["verdict"] == "PASS"
    assert g1["baseline_files"] == [str(Path(p).resolve()) for p in bp] and g1["margins_file"] == str((mdir / "margins.json").resolve())

    cmd = [sys.executable, str(REPO_ROOT / "scripts/g2_report.py"), "--config", str(cfg_path), "--lock", str(lock), "--split", str(split_path),
           "--margins", str(mdir / "margins.json"), "--g1-report", str(work / "g1.json"), "--baseline", *bp, "--checkpoints", *paths.values(), "--out", str(tmp_path / "g2.json")]
    r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert read_json(tmp_path / "g2.json")["gate_valid"] is True
