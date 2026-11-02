"""scripts/build_g1.py: exposed-run data and specs (CPU, synthetic split / pool / lock)."""

import importlib.util
import json
import os
import subprocess

import pytest
import yaml

from contam.config import DEFAULT_CONFIG, REPO_ROOT, file_sha256, load_config, load_prompts, read_json, write_json
from contam.data import freeze
from contam.data.exposure import exposure_ids
from contam.data.samedist import make_samedist
from contam.data.splits import make_split
from contam.eval.gates import parse_run_id
from contam.eval.style import words
from tests.conftest import make_annotations
from tests.test_manifests import make_coco

_spec = importlib.util.spec_from_file_location("build_g1", REPO_ROOT / "scripts/build_g1.py")
bg1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bg1)

N_POOL, M, N_EXP = 1200, 1000, 100


@pytest.fixture
def env(tmp_path):
    ann = make_annotations(40, 268, 140)  # 448 images: every nested fraction is a whole count, as with NoCaps' 4,500
    split = make_split(ann, 20261019, 20261020)
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    sd_dir = tmp_path / "splits"
    raw["paths"].update(splits_dir=str(sd_dir), annotations=str(tmp_path / "annotations.json"), llava_mix=str(tmp_path / "mix.json"),
                        llava_images=str(tmp_path / "llava"), images_dir=str(tmp_path / "nocaps_images"), outputs_dir=str(tmp_path / "outputs"))
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg = load_config(cfg_path)
    write_json({"meta": {"ref_normalization": dict(bg1.REF_NORMALIZATION)}, "images": ann}, tmp_path / "annotations.json")
    mix = [{"id": f"g{i}", "image": f"coco/train2017/{i:012d}.jpg", "conversations": [{"from": "human", "value": "<image>\nq"}, {"from": "gpt", "value": f"a{i}"}]}
           for i in range(N_POOL)]
    write_json(mix, tmp_path / "mix.json")
    write_json(split, sd_dir / "split_A.json")
    write_json(make_samedist(split, ann, make_coco(), seed=1), sd_dir / "samedist_A.json")
    order = list(reversed(range(N_POOL)))  # any fixed order: the builder must follow it, not the mix order
    (sd_dir / "general_order.jsonl").write_text("".join(json.dumps({"order": k, "src_index": i}) + "\n" for k, i in enumerate(order)))
    lock = tmp_path / "lock.json"
    freeze.write_lock(cfg, lock)
    return {"cfg": cfg, "ann": ann, "split": split, "mix": mix, "order": order, "lock": lock, "out": tmp_path / "runs", "tmp": tmp_path}


def _build(env, conditions=("Caption", "Paired", "SameDist"), doses=(1, 4), slots=(0, 1), out=None):
    return bg1.build(env["cfg"], "A", conditions, doses, slots, M, N_EXP, out or env["out"], "G1", env["lock"], check_images=False)


def _data(env, cond, d):
    return read_json(env["out"] / "data" / f"A_{bg1.CONDITIONS[cond]}_d{d}_M{M}_nexp{N_EXP}.json")


def test_counts_nesting_and_specs(env):
    res = _build(env)
    ann, split = env["ann"], env["split"]
    ids = exposure_ids(split, ann, N_EXP)
    assert set(ids) < set(split["seen"]) and len(ids) == N_EXP
    general = [env["mix"][i] for i in env["order"][:M]]
    seeds = env["cfg"]["training"]["seeds"]
    assert len(res["runs"]) == 3 * 2 * 2
    for cond in ("Caption", "Paired", "SameDist"):
        for d in (1, 4):
            K = N_EXP * d
            data = _data(env, cond, d)
            assert len(data) == M
            assert data[: M - K] == general[: M - K]  # first M - K of the frozen order, exposure replaces the tail
            exposed = data[M - K :]
            assert len(exposed) == K
            uniq = sorted({e["id"] for e in exposed})
            assert len(uniq) == N_EXP and all(sum(e["id"] == u for e in exposed) == d for u in uniq[:5])
            assert sorted(int(u.split(":")[1]) for u in uniq) == ids  # same exposed images in every condition and d
            for k in (0, 1):
                s = read_json(env["out"] / "G1" / f"G1_{cond}_d{d}_s{k}.json")
                assert parse_run_id(s["run_id"]) == (cond, d, k)
                assert (s["gate"], s["condition"], s["d"], s["K"], s["M"], s["N_exp"], s["seed_slot"], s["train_seed"]) == ("G1", cond, d, K, M, N_EXP, k, seeds[k])
                assert s["data"] == {"path": s["data"]["path"], "sha256": file_sha256(s["data"]["path"]), "n_examples": M, "general_prefix": M - K, "exposed": K}
                assert s["manifest_lock_sha256"] == file_sha256(env["lock"]) and set(s["manifest_entries"]) == {"split_A.json", "general_order.jsonl", "samedist_A.json"}
                assert s["eval_splits"] == {"A": {"path": "split_A.json", "content_hash": split["content_hash"]}}
                assert s["exposure"]["K"] == K and s["exposure"]["condition"] == bg1.CONDITIONS[cond]
    # d=1 exposed rows are exactly the unique rows of d=4 (nested doses)
    for cond in ("Caption", "Paired", "SameDist"):
        one = {e["id"]: e for e in _data(env, cond, 1)[M - N_EXP :]}
        four = {e["id"]: e for e in _data(env, cond, 4)[M - 4 * N_EXP :]}
        assert one == four
    shas = {(r["condition"], r["d"]): r["exposure_ids_sha256"] for r in res["cells"]}
    assert len(set(shas.values())) == 1


def test_condition_formats(env):
    _build(env, doses=(4,), slots=(0,))
    ann, split = env["ann"], env["split"]
    by_id = {a["image_id"]: a for a in ann}
    ref = lambda i: by_id[i]["refs"][split["train_ref_idx"][str(i)]]
    p = load_prompts()
    assert p["caption"] == p["caption_text"] == "Describe the image in detail."  # D3
    tail = lambda c: _data(env, c, 4)[M - 4 * N_EXP :]
    for e in tail("Caption"):
        i = int(e["id"].split(":")[1])
        assert "image" not in e
        assert e["conversations"] == [{"from": "human", "value": "Describe the image in detail."}, {"from": "gpt", "value": ref(i)}]
    for e in tail("Paired"):
        i = int(e["id"].split(":")[1])
        assert e["image"] == f"nocaps/{by_id[i]['file_name']}"
        assert e["conversations"] == [{"from": "human", "value": "<image>\nDescribe the image in detail."}, {"from": "gpt", "value": ref(i)}]
    sd = tail("SameDist")
    assert len(sd) == 4 * N_EXP
    for e in sd:
        i = int(e["id"].split(":")[1])
        assert e["image"].startswith("coco/train2017/") and e["conversations"][0]["value"] == "<image>\nDescribe the image in detail."
        assert len(words(e["conversations"][1]["value"])) == len(words(ref(i)))  # length-matched to the image's training ref


def test_data_sha_is_deterministic(env):
    a = _build(env, conditions=("Paired",), doses=(4,), slots=(0,))
    b = _build(env, conditions=("Paired",), doses=(4,), slots=(0,), out=env["tmp"] / "runs2")
    assert a["cells"][0]["data_sha256"] == b["cells"][0]["data_sha256"]
    assert _build(env, conditions=("Paired",), doses=(4,), slots=(0,))["cells"][0]["data_sha256"] == a["cells"][0]["data_sha256"]  # rebuild in place


def test_refuses_changed_data_file(env):
    r = _build(env, conditions=("Paired",), doses=(4,), slots=(0,))
    with open(r["cells"][0]["data"], "a") as f:
        f.write(" ")
    with pytest.raises(SystemExit, match="exists with different content"):
        _build(env, conditions=("Paired",), doses=(4,), slots=(0,))


def test_specs_written_once(env):
    _build(env, conditions=("Paired",), doses=(4,), slots=(0,))
    spec = env["out"] / "G1" / "G1_Paired_d4_s0.json"
    before = (spec.read_bytes(), spec.stat().st_mtime_ns)
    _build(env, conditions=("Paired",), doses=(4,), slots=(0,))  # identical content: no-op
    assert (spec.read_bytes(), spec.stat().st_mtime_ns) == before
    s = read_json(spec)
    write_json({**s, "train_seed": s["train_seed"] + 1}, spec)  # e.g. a spec a run was launched from, edited by hand
    with pytest.raises(SystemExit, match="exists with different content"):
        _build(env, conditions=("Paired",), doses=(4,), slots=(0,))
    assert read_json(spec)["train_seed"] == s["train_seed"] + 1 and not list(spec.parent.glob("*.new"))


def test_refuses_manifest_mismatch(env):
    p = env["tmp"] / "splits" / "samedist_A.json"
    sd = read_json(p)
    sd["rows"] = sd["rows"][::-1]
    write_json(sd, p)
    with pytest.raises(SystemExit, match="changed samedist_A.json"):
        _build(env, conditions=("Caption",))
    lock = read_json(env["lock"])
    del lock["files"]["general_order.jsonl"]
    write_json(lock, env["lock"])
    with pytest.raises(SystemExit, match="general_order.jsonl is not frozen"):
        _build(env, conditions=("Caption",))


def test_refuses_annotations_mismatch(env):
    ap = env["tmp"] / "annotations.json"
    obj = read_json(ap)
    write_json({**obj, "meta": {"ref_normalization": {"strip_whitespace": True, "strip_trailing_not_sign": False}}}, ap)
    with pytest.raises(SystemExit, match="ref_normalization"):
        _build(env, conditions=("Caption",))
    obj["images"][0]["refs"][0] += " changed"
    write_json(obj, ap)
    with pytest.raises(SystemExit, match="differ from those split"):
        _build(env, conditions=("Caption",))


def test_refuses_bad_doses_and_slots(env):
    with pytest.raises(SystemExit):
        _build(env, doses=(0,))
    with pytest.raises(SystemExit):
        _build(env, slots=(5,))
    with pytest.raises(SystemExit, match="K = N_exp x d"):
        bg1.build(env["cfg"], "A", ["Paired"], [16], [0], M, N_EXP, env["out"], "G1", env["lock"], check_images=False)


def test_exploratory_runs(env):
    res = bg1.build(env["cfg"], "A", ["Paired", "SameDist"], [2], [0], M, 30, env["out"], "G1", env["lock"], check_images=False, exploratory=True)
    assert res["exploratory"] is True and res["runs"] == ["EX_Paired_d2_n30_s0", "EX_SameDist_d2_n30_s0"]
    for run_id in res["runs"]:
        s = read_json(env["out"] / "EX" / f"{run_id}.json")
        assert (s["gate"], s["exploratory"], s["d"], s["N_exp"], s["K"]) == ("EX", True, 2, 30, 60)
        assert parse_run_id(s["run_id"]) is None  # no gate parses (or accepts) an exploratory run id
    assert set(exposure_ids(env["split"], env["ann"], 30)) < set(exposure_ids(env["split"], env["ann"], N_EXP))  # nested with the main set
    assert not (env["out"] / "G1").exists()
    plain = _build(env, conditions=("Paired",), doses=(4,), slots=(0,))
    assert plain["exploratory"] is False and "exploratory" not in read_json(env["out"] / "G1" / "G1_Paired_d4_s0.json")
    with pytest.raises(SystemExit, match="reserved"):
        bg1.build(env["cfg"], "A", ["Paired"], [4], [0], M, N_EXP, env["out"], "EX", env["lock"], check_images=False)


def test_prereg_grid_check(env):
    cfg = env["cfg"]
    bg1.check_prereg_grid(cfg, "A", [1, 4, 16, 64], cfg["exposure"]["n_exp"])
    bg1.check_prereg_grid(cfg, "A", [4], len(env["split"]["seen"]))  # a stored exposure-set size (C2 arms)
    with pytest.raises(SystemExit, match="--exploratory"):
        bg1.check_prereg_grid(cfg, "A", [2], cfg["exposure"]["n_exp"])
    with pytest.raises(SystemExit, match="--exploratory"):
        bg1.check_prereg_grid(cfg, "A", [4], 30)


def test_nocaps_link(env):
    cfg = env["cfg"]
    (env["tmp"] / "llava").mkdir()
    (env["tmp"] / "nocaps_images").mkdir()
    with pytest.raises(SystemExit, match="--link-nocaps"):
        bg1.ensure_nocaps_link(cfg, create=False)
    link = bg1.ensure_nocaps_link(cfg, create=True)
    assert link.resolve() == (env["tmp"] / "nocaps_images").resolve()
    assert bg1.ensure_nocaps_link(cfg, create=True) == link  # idempotent
    link.unlink()
    link.symlink_to(env["tmp"])
    with pytest.raises(SystemExit, match="not paths.images_dir"):
        bg1.ensure_nocaps_link(cfg, create=True)


def test_image_check_finds_missing_exposed_images(env):
    with pytest.raises(SystemExit, match="exposed images are missing"):
        bg1.build(env["cfg"], "A", ["Paired"], [4], [0], M, N_EXP, env["out"], "G1", env["lock"])


# ----------------------------------------------------------------------------- scripts/run_train.sh

FAKE_PY = """#!/bin/bash
echo "$*" >> "$FAKE_CALLS"
[ "$1" = "-m" ] && exit 0
name=$(basename "$3" .json)
rc=$(cat "$FAKE_RC/$name" 2>/dev/null || echo 0)
[ "$rc" = 0 ] && mkdir -p "$CONTAM_CKPT_DIR/$name" && touch "$CONTAM_CKPT_DIR/$name/DONE"
exit "$rc"
"""


def _run_train(tmp_path, ids, rcs=None, slurm=None):
    py, specs, ckpt, rc_dir, bin_dir = (tmp_path / n for n in ("py", "specs", "ckpt", "rc", "bin"))
    for d in (specs, ckpt, rc_dir, bin_dir):
        d.mkdir(exist_ok=True)
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    (bin_dir / "scontrol").write_text(f'#!/bin/bash\necho "$*" >> {tmp_path / "scontrol"}\n')
    (bin_dir / "scontrol").chmod(0o755)
    for i in ids:
        (specs / f"{i}.json").write_text("{}")
    for i, rc in (rcs or {}).items():
        (rc_dir / i).write_text(str(rc))
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(CONTAM_PY=str(py), CONTAM_CKPT_DIR=str(ckpt), FAKE_CALLS=str(tmp_path / "calls"), FAKE_RC=str(rc_dir),
               RUN_SPECS=" ".join(str(specs / f"{i}.json") for i in ids), PATH=f"{bin_dir}:{env['PATH']}", **(slurm or {}))
    r = subprocess.run(["bash", str(REPO_ROOT / "scripts/run_train.sh")], capture_output=True, text=True, env=env, timeout=60)
    calls_file = tmp_path / "calls"
    calls = [c.split()[-1].rsplit("/", 1)[-1] for c in (calls_file.read_text() if calls_file.exists() else "").splitlines() if "train_lora.py" in c]
    requeued = (tmp_path / "scontrol").read_text() if (tmp_path / "scontrol").exists() else ""
    return r, calls, requeued


def test_run_train_runs_specs_skips_busy_and_summarizes(tmp_path):
    ids = ["G1_Caption_d4_s0", "G1_Paired_d4_s0", "G1_SameDist_d4_s0"]
    r, calls, _ = _run_train(tmp_path, ids, rcs={"G1_Paired_d4_s0": 76})
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls == [f"{i}.json" for i in ids]
    assert "G1_Paired_d4_s0: busy (another job is training it), skipping" in r.stdout
    assert "DONE runs: G1_Caption_d4_s0 G1_SameDist_d4_s0; not DONE: G1_Paired_d4_s0" in r.stdout and "nothing left for this job" in r.stdout


def test_run_train_requeues_only_train_named_batch_jobs(tmp_path):
    ids = ["G1_Caption_d4_s0", "G1_Paired_d4_s0"]
    slurm = {"SLURM_JOB_ID": "123", "SLURM_JOB_PARTITION": "public"}
    (tmp_path / "a").mkdir()
    r, calls, requeued = _run_train(tmp_path / "a", ids, rcs={"G1_Caption_d4_s0": 75}, slurm={**slurm, "SLURM_JOB_NAME": "g1_train"})
    assert r.returncode == 75 and calls == ["G1_Caption_d4_s0.json"] and requeued.strip() == "requeue 123"
    (tmp_path / "b").mkdir()
    r, _, requeued = _run_train(tmp_path / "b", ids, rcs={"G1_Caption_d4_s0": 75}, slurm={**slurm, "SLURM_JOB_NAME": "interactive"})
    assert r.returncode == 75 and requeued == ""
    (tmp_path / "c").mkdir()
    r, _, requeued = _run_train(tmp_path / "c", ids, rcs={"G1_Caption_d4_s0": 75}, slurm={**slurm, "SLURM_JOB_NAME": "train", "NO_REQUEUE": "1"})
    assert r.returncode == 75 and requeued == ""


def test_run_train_needs_existing_specs(tmp_path):
    r, calls, _ = _run_train(tmp_path, [])
    assert r.returncode != 0 and calls == []
