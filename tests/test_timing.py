"""D2 timing run helpers (scripts/timing_run.py): sample construction, steady state, projection."""

import importlib.util
import json
from collections import Counter

import numpy as np
import pytest

from contam.config import REPO_ROOT, write_json

spec = importlib.util.spec_from_file_location("timing_run", REPO_ROOT / "scripts" / "timing_run.py")
timing_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing_run)


def _mix(tmp_path, n=400):
    from PIL import Image

    root = tmp_path / "llava"
    srcs = ["coco"] * 5 + ["gqa"] * 2 + ["vg", "text", "textvqa"]
    ex = []
    for i in range(n):
        s = srcs[i % len(srcs)]
        e = {"id": str(i), "conversations": [{"from": "human", "value": "<image>\nq" if s != "text" else "q"}, {"from": "gpt", "value": f"a {i}"}]}
        if s != "text":
            e["image"] = f"{s}/img/{i}.jpg"
            if s != "gqa":  # gqa is "not downloaded"
                p = root / e["image"]
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(p)
        ex.append(e)
    for i in range(7, n, 20):  # half the vg images half-unzipped: unreadable counts as missing
        (root / "vg" / "img" / f"{i}.jpg").write_bytes(b"truncated")
    write_json(ex, tmp_path / "mix.json")
    return {"paths": {"llava_mix": tmp_path / "mix.json", "llava_images": root, "splits_dir": tmp_path / "splits"}, "training": {"general_seed": 3}}


def test_stand_in_sample_keeps_the_mix_and_the_conversations(tmp_path):
    cfg = _mix(tmp_path)
    ex, info = timing_run.build_sample(cfg, 100, "stand-in", log=lambda m: None)
    assert len(ex) == 100 and info["sources_missing"] == ["gqa"] and info["sources_partial"] == ["vg"]
    mix = json.loads((tmp_path / "mix.json").read_text())
    for e in ex:
        assert e["conversations"] == mix[e["src_index"]]["conversations"]
        if "orig_image" in e:
            assert e["orig_image"] == mix[e["src_index"]]["image"] and not e["image"].startswith("gqa/") and e["image"] not in {f"vg/img/{i}.jpg" for i in range(7, 400, 20)}
    assert info["by_source"]["gqa"]["n_stand_in"] == info["by_source"]["gqa"]["n_sample"] == Counter(e["source"] for e in ex)["gqa"] > 0
    assert abs(info["by_source"]["coco"]["share_sample"] - 0.5) <= 0.02  # the interleaved prefix keeps the mix
    assert not (tmp_path / "splits").exists()


def test_reweight_sample_drops_missing_sources(tmp_path):
    cfg = _mix(tmp_path)
    ex, info = timing_run.build_sample(cfg, 100, "reweight", log=lambda m: None)
    got = Counter(e["source"] for e in ex)
    assert len(ex) == 100 and got["gqa"] == 0 and info["sources_missing"] == ["gqa"] and info["n_stand_in"] == 0
    assert all("orig_image" not in e for e in ex)
    assert got["coco"] / 100 == pytest.approx(5 / 7.5, abs=0.03)  # renormalized over the available examples (half of vg)


def test_steady_state_skips_warmup_per_segment_and_projects(tmp_path):
    log = tmp_path / "train_log.jsonl"
    recs = [{"step": s, "segment": 1 if s <= 4 else 2, "step_time_s": 100.0 if s in (1, 5) else 10.0, "examples": 128, "tokens": 128 * 700, "loss": 1.0, "peak_mem_gb": 50.0}
            for s in range(1, 9)]
    log.write_text("".join(json.dumps(r) + "\n" for r in recs))
    st = timing_run.steady_state(log, warmup=1)
    assert st["steps_measured"] == 6 and st["examples_per_s"] == pytest.approx(12.8) and st["tokens_per_example"] == 700
    pr = timing_run.project(12.8, ckpt_save_s=36.0, overhead_s=360.0, gbs=128, every=20)
    v = pr["300000"]
    assert v["optimizer_steps"] == 2344 and v["train_gpu_h_per_run"] == pytest.approx(300000 / 12.8 / 3600, abs=0.01)
    assert v["gpu_h_per_run"] == pytest.approx(300000 / 12.8 / 3600 + 117 * 36 / 3600 + 0.1, abs=0.01)
    assert v["P0"]["eval_gpu_h"] == pytest.approx(99 * 1.7, abs=0.1) and v["G0"]["eval_ckpts"] == 6


def test_exposure_sample_replaces_the_tail_at_the_run_share(tmp_path):
    from PIL import Image

    from contam.config import load_config
    from contam.data.splits import make_split
    from tests.conftest import make_annotations

    cfg = _mix(tmp_path)
    ann = make_annotations(30, 120, 50, seed=3)
    split = make_split(ann, 1, 2, name="A")
    (tmp_path / "splits").mkdir()
    write_json(split, tmp_path / "splits" / "split_A.json")
    write_json({"meta": {}, "images": ann}, tmp_path / "annotations.json")
    cfg["paths"]["annotations"] = tmp_path / "annotations.json"
    cfg["exposure"] = load_config()["exposure"]
    general, info = timing_run.build_sample(cfg, 100, "stand-in", log=lambda m: None)
    with pytest.raises(SystemExit, match="ln -s"):  # NoCaps images not linked under the image root yet
        timing_run.add_exposure(cfg, general, info, 100, "paired", 4, 20, 160, "A")
    for a in ann:
        p = tmp_path / "llava" / "nocaps" / a["file_name"]
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(p, format="JPEG")
    ex, info2 = timing_run.add_exposure(cfg, general, info, 100, "paired", 4, 20, 160, "A")  # K = 80 = half of M
    e = info2["exposure"]
    assert len(ex) == 100 and e["K"] == 80 and e["n_exposed_in_sample"] == 50 and e["n_general_in_sample"] == 50
    assert ex[:50] == general[:50] and all(x["source"] == "exposed_paired" for x in ex[50:])
    imgs = [x["image"] for x in ex[50:]]
    assert len(set(imgs)) == 20 and max(Counter(imgs).values()) - min(Counter(imgs).values()) <= 1  # round-robin over the 20 images


def test_d2_summary_pools_repeats_per_cell():
    spec2 = importlib.util.spec_from_file_location("d2_summary", REPO_ROOT / "scripts" / "d2_summary.py")
    d2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(d2)

    def rep(r, eps, exp=None):
        return {"gpu": "A100", "args": {"repeat": r}, "sample": {"exposure": exp},
                "timing": {"examples_per_s": eps, "step_time_s_mean": 128 / eps, "checkpoint_save_s": 30.0, "overhead_s": 300.0,
                           "steps_measured": 995, "micro_batch_size": 16, "tokens_per_example": 800, "peak_mem_gib": 30.0}}

    exp = {"condition": "paired", "d": 64, "n_exp": 1000, "share": 0.4923, "M": 130000}
    rows = d2.summarize([rep(0, 4.0), rep(1, 4.2), rep(2, 3.8), rep(0, 5.0, exp), rep(1, 5.0, exp)], M=130000, gbs=128, every=200)
    by = {r["cell"]: r for r in rows}
    g = by["general only (d=0)"]
    assert g["repeats"] == [0, 1, 2] and g["examples_per_s"] == (4.0, 0.2)
    assert g["gpu_h_per_run"] == pytest.approx(130000 / 4.0 / 3600 + (1016 // 200) * 30 / 3600 + 300 / 3600, abs=0.01)
    assert by["paired d=64 (N_exp 1000, share 0.492 of M 130,000)"]["examples_per_s"] == (5.0, 0.0)
    assert "| general only (d=0) | A100 | r0 (995), r1 (995), r2 (995) |" in d2.markdown(rows)
