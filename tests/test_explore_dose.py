"""scripts/analysis/explore_dose.py and scripts/run_short_eval_job.sh (CPU, synthetic split)."""

import importlib.util
import json
import os
import subprocess

import numpy as np
import pytest

from contam.config import REPO_ROOT, content_hash, load_config, write_json
from contam.data.splits import make_split
from contam.eval.evaluate import seen_record
from tests.conftest import excl_record, make_annotations

_spec = importlib.util.spec_from_file_location("explore_dose", REPO_ROOT / "scripts/analysis/explore_dose.py")
ed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ed)

KW = dict(n_boot=100, seed=0, ci=0.95)
WORDS = "dog cat man woman car tree red blue small large field street bus boat".split()


@pytest.fixture
def env(tmp_path):
    ann = make_annotations(10, 60, 30, seed=3)
    split = make_split(ann, 11, 12)
    cfg = load_config()
    main_n = len(split["seen"])  # the stored 50% set plays the main N_exp
    cfg["exposure"]["n_exp"] = main_n
    seen_defs = ed.SeenDefs(cfg, split, ann)
    refs = {a["image_id"]: a["refs"] for a in ann}
    tri = {int(k): v for k, v in split["train_ref_idx"].items()}
    return {"tmp": tmp_path, "ann": ann, "split": split, "cfg": cfg, "main_n": main_n, "seen_defs": seen_defs, "refs": refs, "tri": tri,
            "excl": {**excl_record(), "applied": True}}


def _entry(env, cid, short=0.0, memorize=0.0, rmi=0.0, prompt_id="caption", seed=0, n_exp=None):
    """Write a metrics file + captions for ``cid``: ``short`` drops the word count, ``memorize`` is the
    fraction of exposed (first ``n_exp`` seen) images whose caption is the training reference."""
    split, rng = env["split"], np.random.default_rng([seed, int(content_hash(cid)[:8], 16)])
    run = ed.parse_run(cid)
    n_exp = n_exp or (run and run["n_exp"]) or env["main_n"]
    exposed = set(env["seen_defs"](n_exp)["image_ids"])
    ids = sorted(split["seen"] + split["unseen"])
    gens, per = {}, {}
    for i in ids:
        mem = i in exposed and rng.random() < memorize
        n = max(2, int(rng.normal(30 - short, 2)))
        gens[i] = env["refs"][i][env["tri"][i]] if mem else " ".join(rng.choice(WORDS, n))
        per[str(i)] = {"word_count": float(len(gens[i].split())), "boilerplate": 0.0, "exact_match_train": float(mem), "cider_heldout": 10 + short + rng.normal(0, 1),
                       "spice_heldout": 15 + rng.normal(0, 1), "rmi_sum": (rmi if i in exposed else 0.0) + rng.normal(0, 0.1),
                       "rmi_norm": (rmi / 10 if i in exposed else 0.0) + rng.normal(0, 0.01)}
    d = env["tmp"] / ("short" if prompt_id != "caption" else "full")
    d.mkdir(exist_ok=True)
    rows = [{"checkpoint_id": cid, "model": "m", "adapter": None, "image_id": i, "prompt_id": prompt_id, "prompt_sha": f"P-{prompt_id}", "temperature": 0.0,
             "sample_idx": 0, "seed": None, "batch_size": 16, "max_new_tokens": 300, "caption": gens[i]} for i in ids]
    (d / f"{cid}_{prompt_id}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    m = {"checkpoint_id": cid, "split": {"name": split["name"], "content_hash": split["content_hash"]}, "complete": True, "prompt_id": prompt_id,
         "generations_hash": content_hash({str(i): gens[i] for i in ids}), "analysis_exclusion": env["excl"], "seen_definition": seen_record(env["seen_defs"](env["main_n"])),
         "provenance": {"generations": {"checkpoint_id": cid, "prompt_sha": f"P-{prompt_id}", "model": "m", "adapter": None, "batch_size": 16, "max_new_tokens": 300}},
         "per_image": per}
    p = d / f"{cid}_metrics.json"
    write_json(m, p)
    if run is not None:  # the run spec (scripts/build_g1.py) holds N_exp and the exposure set
        n = n_exp
        write_json({"run_id": cid, "condition": run["condition"], "d": run["d"], "seed_slot": run["slot"], "N_exp": n,
                    "exposure": {"n_exp": n, "exposure_ids_sha256": env["seen_defs"](n)["ids_sha256"]}}, env["tmp"] / "runs" / run["gate"] / f"{cid}.json")
    return ed.load_entry(p, env["tmp"] / "runs")


def _d0(env, prompt_id="caption"):
    return [_entry(env, f"G0_d0_s{s}", prompt_id=prompt_id, seed=s) for s in range(3)]


def test_parse_run():
    assert ed.parse_run("G1_Paired_d4_s0") == {"gate": "G1", "condition": "Paired", "d": 4, "n_exp": None, "slot": 0}
    assert ed.parse_run("EX_SameDist_d1_n30_s1") == {"gate": "EX", "condition": "SameDist", "d": 1, "n_exp": 30, "slot": 1}
    for bad in ("G0_d0_s0", "base", "EX_Paired_d2_s0", "G1_Paired_d4_n30_s0", "G1_Foo_d4_s0", "G1_Paired_d4_s0x"):
        assert ed.parse_run(bad) is None


def test_run_spec_gives_n_exp(env):
    e = _entry(env, "G1_Paired_d4_s0", n_exp=30)  # a G run id says nothing about N_exp: the spec does
    assert e["run"]["n_exp"] == 30 and e["run"]["exposure_ids_sha256"] == env["seen_defs"](30)["ids_sha256"]
    spec = env["tmp"] / "runs" / "G1" / "G1_Paired_d4_s0.json"
    metrics = env["tmp"] / "full" / "G1_Paired_d4_s0_metrics.json"
    s = json.load(open(spec))
    write_json({**s, "d": 64}, spec)
    with pytest.raises(SystemExit, match="disagree"):
        ed.load_entry(metrics, env["tmp"] / "runs")
    spec.unlink()
    with pytest.raises(SystemExit, match="run spec"):
        ed.load_entry(metrics, env["tmp"] / "runs")
    e = _entry(env, "EX_Paired_d1_n30_s0", n_exp=30)
    s = json.load(open(env["tmp"] / "runs" / "EX" / "EX_Paired_d1_n30_s0.json"))
    write_json({**s, "N_exp": 100, "exposure": {**s["exposure"], "n_exp": 100}}, env["tmp"] / "runs" / "EX" / "EX_Paired_d1_n30_s0.json")
    with pytest.raises(SystemExit, match="disagree"):  # the id spells N_exp = 30
        ed.load_entry(env["tmp"] / "full" / "EX_Paired_d1_n30_s0_metrics.json", env["tmp"] / "runs")


def test_load_entry_checks_generations(env):
    e = _entry(env, "G1_Paired_d4_s0")
    assert e["gens"] is not None and e["run"]["d"] == 4
    p = env["tmp"] / "full" / "G1_Paired_d4_s0_caption.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    rows[0]["caption"] += " changed"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="does not hold the generations"):
        ed.load_entry(env["tmp"] / "full" / "G1_Paired_d4_s0_metrics.json", env["tmp"] / "runs")


def test_rebind_seen(env):
    e = _entry(env, "EX_Paired_d1_n30_s0")
    small = env["seen_defs"](30)
    assert set(small["image_ids"]) < set(env["seen_defs"](env["main_n"])["image_ids"])  # K-sweep sets nest in the main one
    m = ed.rebind_seen(e["m"], small, env["seen_defs"])
    assert m["seen_definition"]["n_exp"] == 30 and m["per_image"] is e["m"]["per_image"] and "summary" not in m
    with pytest.raises(ValueError, match="not an exposure set"):
        ed.rebind_seen({**e["m"], "seen_definition": None}, small, env["seen_defs"])
    with pytest.raises(ValueError, match="not an exposure set"):
        ed.rebind_seen({**e["m"], "seen_definition": {**e["m"]["seen_definition"], "ids_sha256": "x"}}, small, env["seen_defs"])


def test_analyze(env):
    d0 = _d0(env)
    model = _entry(env, "G1_Paired_d4_s0", short=20, memorize=0.6, rmi=5.0)
    nouns = ed.Nouns(env["tmp"] / "cache", "auto")
    try:
        res = ed.analyze([model], d0, env["split"], env["excl"], env["seen_defs"](env["main_n"]), env["seen_defs"], KW, nouns=nouns, floors={"rmi_sum": 0.5, "rmi_norm": 0.02})
    except (ImportError, OSError):
        pytest.skip("no POS tagger installed")
    assert res["exploratory"] is True and res["gate_valid"] is False
    wc = res["levels"]["word_count"]["unseen"]
    assert wc["diff"]["ci_hi"] < -10 and wc["model"]["estimate"] < wc["d0"]["estimate"]
    em = res["levels"]["exact_match_train"]
    assert em["seen"]["diff"]["ci_lo"] > 0.4 and em["unseen"]["diff"]["estimate"] == 0
    assert res["rmi"]["rmi_sum"]["seen"]["ci_lo"] > 4 and res["rmi"]["rmi_sum"]["floor"] == 0.5
    assert abs(res["rmi"]["rmi_norm"]["unseen"]["estimate"]) < 0.01
    assert "cider_heldout" in res["A"] and "spice_heldout" in res["A"]
    div = res["diversity"]
    assert set(div) == {"unseen", "seen"}
    assert 0 <= div["seen"]["duplicate_rate"]["model"]["estimate"] <= 1 and div["unseen"]["distinct_1"]["diff"]["ci"] == 0.95
    assert div["unseen"]["nouns_per_caption"]["diff"]["estimate"] < 0  # shorter captions, fewer nouns
    assert len(list((env["tmp"] / "cache").glob("nouns_auto_*.json"))) == 4  # cached per backend and generations file
    assert "Duplicate rate" in ed.cell_markdown(res, "t")
    md = ed.cell_markdown(res, "t")
    assert "| word_count | unseen |" in md and "distinct_2" in md


def test_tau_split_and_delta_v(env):
    d0 = _d0(env)
    seen = env["seen_defs"](env["main_n"])
    series = {1: [_entry(env, "G2_Paired_d1_s0", short=20, memorize=0.2, rmi=2.0)], 4: [_entry(env, "G1_Paired_d4_s0", short=20, memorize=0.8, rmi=20.0)]}
    dv = ed.delta_v(None, d0, env["split"], env["excl"], seen, env["cfg"])
    assert dv["rmi_sum"]["delta"] == 0.5 and dv["rmi_sum"]["status"].startswith("pending")
    p64 = [_entry(env, "G1_Paired_d64_s0", rmi=40.0)]
    dv64 = ed.delta_v(p64, d0, env["split"], env["excl"], seen, env["cfg"])
    assert dv64["rmi_sum"]["delta"] == pytest.approx(4.0, abs=0.1) and dv64["rmi_sum"]["status"].startswith("set")
    rep = env["tmp"] / "g1_report_A.json"
    write_json({"criteria": {"a": {"delta_V": {"rmi_sum": 3.5, "rmi_norm": 0.3}}}}, rep)
    dvr = ed.delta_v(p64, d0, env["split"], env["excl"], seen, env["cfg"], str(rep))  # the G1 value wins; the recomputation is recorded
    assert dvr["rmi_sum"]["delta"] == 3.5 and dvr["rmi_norm"]["delta"] == 0.3 and "g1_report_A.json" in dvr["rmi_sum"]["status"]
    assert dvr["rmi_sum"]["recomputed"]["delta"] == pytest.approx(4.0, abs=0.1)
    deltas = {"verbatim_exact_match": (0.4, ""), "verbatim_exact_match_delta0": (0.0, ""), "latent_rmi_sum": (dv64["rmi_sum"]["delta"], ""),
              "latent_rmi_norm": (dv64["rmi_norm"]["delta"], ""), "style_word_count": (1.0, "")}
    t = ed.tau_split(series, d0, env["split"], env["excl"], seen, KW, deltas)
    c = t["components"]
    assert t["doses"] == [1, 4]
    assert c["verbatim_exact_match"]["tau"] == 4.0  # some exact matches at d=1 stay below δ_verbatim (synthetic: 0.4)...
    assert c["verbatim_exact_match_delta0"]["tau"] == 1.0  # ...but fire the δ = 0 onset
    assert c["latent_rmi_sum"]["tau"] == 4.0  # +2 at d=1 is below δ_V ≈ 4
    assert c["style_word_count"]["tau"] == 1.0 and c["style_word_count"]["direction"] == -1.0
    assert c["latent_rmi_sum"]["raw"]["1"]["estimate"] == pytest.approx(2.0, abs=0.1)
    assert "latent_rmi_sum" in ed.tau_markdown({"Paired": t}, dv64, "caption")


def _ctx(env, out, figs, written, **kw):
    def check_new(paths):
        clash = [str(p) for p in paths if os.path.exists(p)]
        if clash:
            raise SystemExit(f"refusing to overwrite {clash}")

    def write(stem, obj, md):
        check_new([f"{stem}.json", f"{stem}.md"])
        write_json(obj, f"{stem}.json")
        open(f"{stem}.md", "w").write(md)
        written.append(stem.name)

    return {"cfg": env["cfg"], "split": env["split"], "excl": env["excl"], "seen_defs": env["seen_defs"], "kw": KW, "write": write, "check_new": check_new,
            "nouns": None, "out_dir": out, "fig_dir": figs, "fig_tag": "t", "delta_S": 1.0, "delta_A": {"spice_heldout": 0.3}, "delta_verbatim": 0.05, **kw}


def test_run_prompt_writes_tables_figures_and_refuses_overwrite(env):
    d0 = _d0(env)
    entries = [_entry(env, "G2_Paired_d1_s0", short=20, rmi=2.0), _entry(env, "G1_Paired_d4_s0", short=20, memorize=0.8, rmi=20.0),
               _entry(env, "G1_Paired_d4_s1", short=20, memorize=0.7, rmi=18.0, seed=1), _entry(env, "G2_SameDist_d1_s0", short=19),
               _entry(env, "EX_Paired_d1_n30_s0", short=15, memorize=0.0, rmi=1.0, n_exp=30), _entry(env, "EX_SameDist_d1_n30_s0", short=14, n_exp=30)]
    out, figs = env["tmp"] / "explore", env["tmp"] / "fig"
    out.mkdir()
    figs.mkdir()
    written = []
    ctx = _ctx(env, out, figs, written)
    plan = ed.plan_prompt("caption", entries, ctx)
    assert len(plan["paths"]) == len(set(plan["paths"])) == 2 * (6 + 1 + 2) + 4
    r = ed.run_prompt("caption", entries, d0, None, ctx)
    assert "G1_Paired_d4_pooled" in written and "G1_Paired_d4_s0" in written and "tau_caption" in written and "summary_caption" in written
    assert sorted(str(p) for p in plan["paths"]) == sorted(str(p) for p in [*out.iterdir(), *figs.iterdir()])  # the plan is exactly what is written
    assert sorted(p.name for p in figs.iterdir()) == ["t_dose.pdf", "t_dose.png", "t_ksweep.pdf", "t_ksweep.png"]
    pooled = json.load(open(out / "G1_Paired_d4_pooled.json"))
    assert pooled["checkpoint_ids"] == ["G1_Paired_d4_s0", "G1_Paired_d4_s1"]
    ks = json.load(open(out / "EX_Paired_d1_n30_s0.json"))
    assert ks["seen_definition"]["n_exp"] == 30 and ks["n"]["seen"] <= 30
    assert r["tau"]["delta_v"]["rmi_sum"]["status"].startswith("pending")
    summ = json.load(open(out / "summary_caption.json"))
    row = next(x for x in summ["rows"] if (x["condition"], x["d"], x["n_exp"]) == ("Paired", 4, env["main_n"]))
    assert row["n_seeds"] == 2 and row["metrics"]["exact_match_seen_pct"]["exceeds_margin"] and row["metrics"]["exact_match_seen_pct"]["margin"] == 5.0
    assert row["metrics"]["word_count_unseen"]["exceeds_margin"] and row["metrics"]["rmi_sum_seen"]["exceeds_margin"]
    assert not next(x for x in summ["rows"] if x["condition"] == "SameDist" and x["n_exp"] == 30)["metrics"]["exact_match_seen_pct"]["exceeds_margin"]
    assert "▲" in (out / "summary_caption.md").read_text() and summ["tau"]["Paired"]["verbatim_exact_match"]["delta"] == 0.05
    n_files = len(list(out.iterdir()))
    with pytest.raises(SystemExit, match="refusing to overwrite"):  # checked up front: nothing is written
        ed.run_prompt("caption", entries, d0, None, ctx)
    assert len(list(out.iterdir())) == n_files
    with pytest.raises(SystemExit, match="different prompt texts"):
        ed.run_prompt("caption", entries[:1], _d0(env, "caption")[:1] + [dict(_d0(env)[1], m={**_d0(env)[1]["m"], "provenance": {"generations": {"prompt_sha": "other"}}})], None, ctx)


def test_run_prompt_checks_spec_exposure_set(env):
    d0 = _d0(env)
    e = _entry(env, "G1_Paired_d4_s0", short=20)
    e["run"]["exposure_ids_sha256"] = "0" * 64
    out = env["tmp"] / "x"
    out.mkdir()
    with pytest.raises(SystemExit, match="exposure ids sha256"):
        ed.run_prompt("caption", [e], d0, None, _ctx(env, out, out, []))


def test_short_prompt_vs_base(env):
    base = [_entry(env, "base", prompt_id="nocaps_like")]
    model = [_entry(env, "G1_Paired_d4_s0", short=5, memorize=0.5, prompt_id="nocaps_like")]
    res = ed.analyze(model, base, env["split"], env["excl"], env["seen_defs"](env["main_n"]), env["seen_defs"], KW)
    assert res["prompt_id"] == "nocaps_like" and res["baseline"]["checkpoint_ids"] == ["base"]
    assert res["levels"]["cider_heldout"]["unseen"]["diff"]["estimate"] > 0
    with pytest.raises(SystemExit, match="is prompt"):
        ed.run_prompt("caption", model, base, None, dict.fromkeys(("cfg", "split", "excl", "seen_defs", "kw", "write"), env["cfg"]))
    out = env["tmp"] / "s"
    out.mkdir()
    lab = "untrained base (short prompt; no trained d=0 baseline)"
    ed.run_prompt("nocaps_like", model, base, None, _ctx(env, out, out, [], baseline_label={"nocaps_like": lab}))
    assert lab in (out / "summary_nocaps_like.md").read_text() and json.load(open(out / "tau_nocaps_like.json"))["baseline"] == lab
    assert (out / "t_short_dose.png").exists()


# ----------------------------------------------------------------------------- scripts/run_short_eval_job.sh

FAKE_PY = """#!/bin/bash
echo "$*" >> "$FAKE_CALLS"
[ "$1" = "-m" ] && [ "$2" = "contam.data.freeze" ] && exit 0
if [ "$1" = "scripts/evaluate_checkpoint.py" ]; then
    out=$(echo "$*" | sed -E 's/.*--out ([^ ]+).*/\\1/'); echo '{}' > "$out"
fi
exit 0
"""


def test_run_short_eval_job(tmp_path):
    py, ckpt, out = tmp_path / "py", tmp_path / "ckpt", tmp_path / "short"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    (ckpt / "G1_Paired_d4_s0").mkdir(parents=True)
    (ckpt / "G1_Paired_d4_s0" / "DONE").touch()
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(CONTAM_PY=str(py), CONTAM_CKPT_DIR=str(ckpt), SHORT_OUT=str(out), FAKE_CALLS=str(tmp_path / "calls"), RUN_IDS="base G1_Paired_d4_s0")
    run = lambda: subprocess.run(["bash", str(REPO_ROOT / "scripts/run_short_eval_job.sh")], capture_output=True, text=True, env=env, timeout=60)  # noqa: E731
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    gen = [c for c in calls if "contam.generate" in c]
    assert len(gen) == 2 and "--adapter" not in gen[0] and f"--adapter {ckpt}/G1_Paired_d4_s0/adapter" in gen[1]
    assert all("--prompt-id nocaps_like" in c and "--temperature 0" in c for c in gen)
    assert f"--out {out}/base_nocaps_like.jsonl" in gen[0]
    ev = [c for c in calls if "evaluate_checkpoint.py" in c]
    assert len(ev) == 2 and all("--logprobs" not in c for c in ev) and f"--out {out}/G1_Paired_d4_s0_metrics_short_A.json" in ev[1]
    (tmp_path / "calls").unlink()
    r = run()  # resumable: finished runs are skipped
    assert r.returncode == 0 and "skipping" in r.stdout
    assert all("freeze" in c for c in (tmp_path / "calls").read_text().splitlines())
    env["RUN_IDS"] = "G1_Caption_d4_s0"
    r = run()
    assert r.returncode == 1 and "is not DONE" in r.stdout


SIGNAL_PY = """#!/bin/bash
echo "$*" >> "$FAKE_CALLS"
if [ "$1" = "-m" ] && [ "$2" = "contam.data.freeze" ]; then sleep "${FREEZE_SLEEP:-0}"; exit 0; fi
exec sleep 30
"""


@pytest.mark.parametrize("script,job_name", [("run_short_eval_job.sh", "short_eval"), ("run_gen_job.sh", "gen_eval")])
@pytest.mark.parametrize("sig,when,requeues", [("USR1", "stage", True), ("TERM", "stage", False), ("HUP", "stage", False), ("USR1", "between", True)])
def test_eval_jobs_stop_and_requeue_only_on_usr1(tmp_path, script, job_name, sig, when, requeues):
    import signal
    import time

    py, ckpt, bindir = tmp_path / "py", tmp_path / "ckpt", tmp_path / "bin"
    py.write_text(SIGNAL_PY)
    bindir.mkdir()
    (bindir / "scontrol").write_text(f'#!/bin/bash\necho "$*" >> {tmp_path}/scontrol\n')
    for p in (py, bindir / "scontrol"):
        p.chmod(0o755)
    (ckpt / "G1_Paired_d4_s0").mkdir(parents=True)
    (ckpt / "G1_Paired_d4_s0" / "DONE").touch()
    calls = tmp_path / "calls"
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(CONTAM_PY=str(py), CONTAM_CKPT_DIR=str(ckpt), SHORT_OUT=str(tmp_path / "short"), FAKE_CALLS=str(calls), RUN_IDS="G1_Paired_d4_s0",
               SLURM_JOB_ID="123", SLURM_JOB_NAME=job_name, PATH=f"{bindir}:{env['PATH']}", FREEZE_SLEEP="2" if when == "between" else "0")
    proc = subprocess.Popen(["bash", str(REPO_ROOT / "scripts" / script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    want = "contam.data.freeze" if when == "between" else "contam.generate"
    t0 = time.time()
    while not (calls.exists() and want in calls.read_text()):
        assert time.time() - t0 < 20 and proc.poll() is None
        time.sleep(0.05)
    proc.send_signal(getattr(signal, f"SIG{sig}"))  # "between": arrives during the freeze check, before the first stage
    out, _ = proc.communicate(timeout=30)
    assert proc.returncode == 75, out
    assert ((tmp_path / "scontrol").exists() and (tmp_path / "scontrol").read_text().strip() == "requeue 123") == requeues, out
    if when == "between":
        assert "contam.generate" not in calls.read_text()  # no stage started after the signal


# ----------------------------------------------------------------------------- formal short-prompt analysis (two-level bootstrap)


def test_two_level_group_means_matches_stats_bootstrap_two_level():
    from contam.eval import stats

    rng = np.random.default_rng(1)
    M, B = rng.normal(1, 1, (2, 40)), rng.normal(0, 1, (5, 40))
    ref = stats.bootstrap_two_level(M, B, n_boot=700, seed=3)  # crosses a chunk boundary (CHUNK = 500)
    pt, rp = ed.two_level_group_means({"m": M[..., None], "b": B[..., None]}, {"g": np.arange(40)}, n_boot=700, seed=3)
    np.testing.assert_allclose(rp["m"]["g"][:, 0] - rp["b"]["g"][:, 0], ref.replicates, atol=1e-12)
    assert pt["m"]["g"][0] - pt["b"]["g"][0] == pytest.approx(ref.estimate)


def test_two_level_group_means_links_and_groups():
    rng = np.random.default_rng(2)
    A = rng.normal(0, 1, (3, 30, 2))
    pt, rp = ed.two_level_group_means({"a": A + 1.0, "b": A}, {"x": np.arange(10), "y": np.arange(10, 30)}, links={"b": "a"}, n_boot=200, seed=0)
    np.testing.assert_allclose(rp["a"]["x"] - rp["b"]["x"], 1.0)  # seed-matched and image-paired: a − b is exactly the shift
    _, rp2 = ed.two_level_group_means({"a": A + 1.0, "b": A}, {"x": np.arange(10)}, n_boot=200, seed=0)
    assert np.std(rp2["a"]["x"][:, 0] - rp2["b"]["x"][:, 0]) > 0.01  # independent seed draws
    assert pt["a"]["y"].shape == (2,)
    with pytest.raises(ValueError, match="cannot share"):
        ed.two_level_group_means({"a": A, "b": A[:2]}, {"x": np.arange(10)}, links={"b": "a"}, n_boot=10)


def _with_all10(e):
    """Synthetic all10 / train scores: the training caption among the refs lifts a memorised (exact-match) caption."""
    for r in e["m"]["per_image"].values():
        r["cider_all10"] = r["cider_heldout"] + 50.0 * r["exact_match_train"]
        r["spice_all10"] = r["spice_heldout"] + 5.0 * r["exact_match_train"]
        r["cider_train"] = 5.0 + 500.0 * r["exact_match_train"]
    return e


def _short(env, cid, seed=0, **kw):
    return _with_all10(_entry(env, cid, prompt_id="nocaps_like", seed=seed, **kw))


def test_formal_cell(env):
    d0 = [_short(env, f"G0_d0_s{s}", seed=s) for s in range(3)]
    dom = {a["image_id"]: a["domain"] for a in env["ann"]}
    model = [_short(env, "EX_Paired_d4_n30_s0", short=3, memorize=0.8), _short(env, "EX_Paired_d4_n30_s1", seed=1, short=3, memorize=0.6)]
    res = ed.formal_cell(model, d0, env["split"], env["excl"], env["seen_defs"](30), env["seen_defs"], dom, KW)  # seen_unexposed = seen outside the 30
    assert {"seen", "seen_unexposed", "unseen", "unseen/in-domain", "unseen/near-domain", "unseen/out-domain"} <= set(res["n"])
    assert sum(res["n"][g] for g in res["n"] if g.startswith("unseen/")) == res["n"]["unseen"]
    lv = res["levels"]["cider_heldout"]["unseen"]
    assert lv["diff"]["ci_lo"] > 1 and lv["diff"]["estimate"] == pytest.approx(lv["model"]["estimate"] - lv["d0"]["estimate"])
    dc = res["decomposition"]["cider_all10"]
    assert dc["seen_specific"]["ci_lo"] > 20  # 50 × ~70% memorised
    assert dc["total"]["estimate"] == pytest.approx(dc["style"]["estimate"] + dc["seen_specific"]["estimate"])
    assert abs(dc["style"]["estimate"] - res["levels"]["cider_all10"]["seen_unexposed"]["diff"]["estimate"]) < 1e-9
    assert set(res["per_seed"]) == {"EX_Paired_d4_n30_s0", "EX_Paired_d4_n30_s1"}
    em = [res["per_seed"][c]["levels"]["exact_match_train"]["seen"]["diff"]["estimate"] for c in ("EX_Paired_d4_n30_s0", "EX_Paired_d4_n30_s1")]
    assert em[0] > em[1] and res["levels"]["exact_match_train"]["seen"]["diff"]["estimate"] == pytest.approx(np.mean(em))
    assert "cider_all10" in res["A"] and res["A"]["cider_all10"]["ci_lo"] > 0


def test_formal_contrast_is_seed_matched(env):
    dom = {a["image_id"]: a["domain"] for a in env["ann"]}
    seen = env["seen_defs"](env["main_n"])
    p = [_short(env, "G1_Paired_d4_s0", short=2), _short(env, "G1_Paired_d4_s1", seed=1, short=2)]
    s = [_short(env, "G1_SameDist_d4_s1", seed=5)]
    c = ed.formal_contrast(p, s, env["split"], env["excl"], seen, env["seen_defs"], dom, KW)
    assert c["slots"] == [1] and c["unmatched"] == [0] and c["a_ids"] == ["G1_Paired_d4_s1"] and "per_pair" not in c
    assert set(c["pooled"]["cider_heldout"]) == {"unseen", "unseen/in-domain", "unseen/near-domain", "unseen/out-domain"}
    assert c["pooled"]["cider_heldout"]["unseen"]["diff"]["ci_lo"] > 0.5
    # sensitivity: every seed of each side, own seed draws; the 1-seed side has no seed variance
    u = ed.formal_contrast(p, s, env["split"], env["excl"], seen, env["seen_defs"], dom, KW, seed_matched=False)
    assert u["seed_matched"] is False and u["a_slots"] == [0, 1] and u["b_slots"] == [1] and u["a_ids"] == ["G1_Paired_d4_s0", "G1_Paired_d4_s1"]
    assert "per_pair" not in u
    pa = [ed.rebind_seen(e["m"], seen, env["seen_defs"]) for e in p]
    ids = ed.analysis_ids(env["split"], "unseen", [int(i) for i in pa[0]["per_image"]], env["excl"], seen)
    sb = ed.rebind_seen(s[0]["m"], seen, env["seen_defs"])
    want = np.mean([np.mean([m["per_image"][str(i)]["cider_heldout"] for m in pa]) - sb["per_image"][str(i)]["cider_heldout"] for i in ids])
    assert u["pooled"]["cider_heldout"]["unseen"]["diff"]["estimate"] == pytest.approx(want)


def test_formal_prompt_writes_tables_and_figures(env):
    d0 = [_short(env, f"G0_d0_s{s}", seed=s) for s in range(3)]
    base = _short(env, "base", seed=9)
    entries = [_short(env, "G2_Paired_d1_s0", short=2, memorize=0.05), _short(env, "G2_Paired_d1_s1", seed=1, short=2), _short(env, "G2_SameDist_d1_s0", short=1),
               _short(env, "G1_Paired_d4_s0", short=2, memorize=0.8), _short(env, "G1_Paired_d4_s1", seed=1, short=2, memorize=0.7),
               _short(env, "G1_SameDist_d4_s0", short=1), _short(env, "G1_SameDist_d4_s1", seed=1, short=1),
               _short(env, "EX_Paired_d1_n30_s0", short=2, n_exp=30)]
    long_d0 = _d0(env)
    long = ([_entry(env, "G2_Paired_d1_s0", short=20), _entry(env, "G1_Paired_d4_s0", short=20)], long_d0)
    out, figs = env["tmp"] / "formal", env["tmp"] / "ffig"
    out.mkdir()
    figs.mkdir()
    written = []
    ctx = _ctx(env, out, figs, written, domain={a["image_id"]: a["domain"] for a in env["ann"]})
    plan = ed.plan_formal("nocaps_like", ctx)
    r = ed.formal_prompt("nocaps_like", entries, d0, base, ctx, long)
    assert sorted(str(p) for p in plan["paths"]) == sorted(str(p) for p in [*out.iterdir(), *figs.iterdir()])
    doc = json.load(open(out / "formal_nocaps_like.json"))
    assert doc["exploratory"] is True and doc["gate_valid"] is False and doc["n_boot"] == KW["n_boot"] and doc["base"]["checkpoint_ids"] == ["base"]
    keys = [(x["condition"], x["d"], x["n_exp"]) for x in doc["rows"]]
    assert keys[:3] == [("Paired", 1, env["main_n"]), ("Paired", 4, env["main_n"]), ("Paired", 1, 30)]
    assert [(c["d"], c["n_exp"], c["slots"]) for c in doc["contrasts"]] == [(1, env["main_n"], [0]), (4, env["main_n"], [0, 1])]
    assert set(doc["contrasts"][1]["per_pair"]) == {"0", "1"}
    sens = doc["contrasts_sensitivity"]
    assert [(c["d"], c["a_slots"], c["b_slots"], c["seed_matched"]) for c in sens] == [(1, [0, 1], [0], False)]
    assert doc["contrasts"][0]["unmatched"] == [1] and "SENSITIVITY" in sens[0]["label"]
    labels = {x["condition"]: x["seen_specific_label"] for x in doc["rows"]}
    assert "negative control" in labels["SameDist"] and labels["Paired"] == "seen-specific share"
    assert all("memorisation" not in x["decomposition"].get("cider_all10", {}) for x in doc["rows"])
    assert [x["d"] for x in doc["long_prompt"]["rows"]] == [1, 4] and set(doc["long_prompt"]["rows"][0]["n"]) == {"unseen"}
    md = (out / "formal_nocaps_like.md").read_text()
    assert "Paired − SameDist" in md and "seen-specific" in md and "negative control" in md and "base (untrained)" in md and "G1_Paired_d4_s1 |" in md
    assert "‡ image-only CI" in md and "exceeds the total" in md and "SENSITIVITY: Paired s0,1 pooled vs SameDist s0" in md
    assert "seed counts differ between conditions at d=1" in md
    assert len(r["figures"]) == 4
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        ed.formal_prompt("nocaps_like", entries, d0, base, ctx, long)
    with pytest.raises(SystemExit, match="is prompt"):
        ed.formal_prompt("nocaps_like", entries, long_d0, None, ctx)
