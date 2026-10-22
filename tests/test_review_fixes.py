"""Regression tests for the adversarial-review findings (items 1-12). Each fails on the pre-fix code."""

import json
import multiprocessing as mp
import shutil
import subprocess
import sys

import pytest

from contam.config import load_config, prompt_sha, read_json, write_json
from contam.data.splits import make_split
from contam.eval import caption_metrics as cm
from contam.eval.evaluate import compare, evaluate_checkpoint, load_generations, load_logprobs, prepare_logprob_resume
from contam.eval.style import DEFAULT_BOILERPLATE_PATTERNS, caption_style, compile_patterns
from tests.conftest import locked_split, make_annotations

java = pytest.mark.skipif(shutil.which("java") is None, reason="needs java")


def _metrics(split, ann, fn, ids=None):
    refs = {a["image_id"]: a["refs"] for a in ann}
    ids = list(refs) if ids is None else ids
    gens = {i: fn(i, refs[i], split["train_ref_idx"][str(i)]) for i in ids}
    return evaluate_checkpoint(gens, ann, split, metrics=["cider"], tokenizer="simple", require_all=ids is None or len(ids) == len(refs))


# ----------------------------------------------------------------------------- 1, 2: compare inputs


@pytest.fixture
def two_splits():
    ann = make_annotations(6, 20, 10, seed=2)
    return ann, make_split(ann, 1, 2, name="A"), make_split(ann, 7, 8, name="B")


def test_compare_rejects_metrics_from_another_split(two_splits):
    ann, A, B = two_splits
    base_A = _metrics(A, ann, lambda i, r, t: r[0])
    model_B = _metrics(B, ann, lambda i, r, t: r[0])
    with pytest.raises(ValueError, match="split"):
        compare([model_B], [base_A], B, n_boot=50)


def test_compare_rejects_partial_metrics(two_splits):
    ann, A, _ = two_splits
    base = _metrics(A, ann, lambda i, r, t: r[0])
    part = _metrics(A, ann, lambda i, r, t: r[0], ids=A["seen"] + A["unseen"][:3])
    with pytest.raises(ValueError, match="partial"):
        compare([part], [base], A, n_boot=50)
    # allow_partial relaxes completeness but never the equal-image-set requirement
    with pytest.raises(ValueError, match="different image sets"):
        with pytest.warns(UserWarning):
            compare([part], [base], A, n_boot=50, allow_partial=True)


def test_compare_allow_partial_is_marked_not_gate_valid(two_splits):
    ann, A, _ = two_splits
    ids = A["seen"] + A["unseen"][:3]
    part = _metrics(A, ann, lambda i, r, t: r[0], ids=ids)
    part2 = _metrics(A, ann, lambda i, r, t: r[1], ids=ids)
    out = compare([part], [part2], A, n_boot=50, allow_partial=True)
    assert out["gate_valid"] is False
    full = _metrics(A, ann, lambda i, r, t: r[0])
    assert compare([full], [full], A, n_boot=50)["gate_valid"] is True


# ----------------------------------------------------------------------------- 3: PTB line separators


@java
@pytest.mark.parametrize("sep", ["\r", " ", " ", "\x85", "\x0b", "\x0c"])
def test_ptb_line_separators_do_not_misalign(sep):
    tok = cm.tokenize({"h1": [f"A dog on a beach.{sep}It is sunny."], "h2": ["A red car."], "h3": ["Two cats."]})
    assert tok == {"h1": ["a dog on a beach it is sunny"], "h2": ["a red car"], "h3": ["two cats"]}


@java
def test_ptb_empty_caption_keeps_alignment():
    tok = cm.tokenize({"a": ["", "A dog."], "b": ["    "], "c": ["Two cats."]})
    assert tok == {"a": ["", "a dog"], "b": [""], "c": ["two cats"]}


def test_ptb_line_count_mismatch_is_fatal(monkeypatch):
    class R:
        stdout = b"a dog\nsecond line\nred car\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    with pytest.raises(RuntimeError, match="misaligned"):
        cm.tokenize({"h1": ["A dog"], "h2": ["A red car"]})


def test_whitespace_normalization():
    assert cm.normalize_whitespace("a b\r\nc\x85d\x0c e  ") == "a b c d e"


# ----------------------------------------------------------------------------- 4: log-prob loading / resume


def _lp_rows(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _lp(iid, j, ck="x", lp=-1.0, **kw):
    return dict(checkpoint_id=ck, image_id=iid, ref_idx=j, prompt_id="caption", logp_sum=lp, n_tokens=5, **kw)


def test_load_logprobs_requires_all_refs(tmp_path):
    p = tmp_path / "lp.jsonl"
    _lp_rows(p, [_lp(1, j) for j in range(10)] + [_lp(2, j) for j in range(6)])
    with pytest.raises(ValueError, match="image 2"):
        load_logprobs(p)


def test_load_logprobs_rejects_duplicates_and_mixed_checkpoints(tmp_path):
    p = tmp_path / "lp.jsonl"
    _lp_rows(p, [_lp(1, j) for j in range(10)] + [_lp(1, 0)])
    with pytest.raises(ValueError, match="duplicate"):
        load_logprobs(p)
    _lp_rows(p, [_lp(1, j) for j in range(10)] + [_lp(2, j, ck="OTHER") for j in range(10)])
    with pytest.raises(ValueError, match="checkpoint_id"):
        load_logprobs(p)
    _lp_rows(p, [_lp(1, j) for j in range(10)])
    assert len(load_logprobs(p, checkpoint_id="x")[1].logp_sum) == 10
    with pytest.raises(ValueError, match="expected 'y'"):
        load_logprobs(p, checkpoint_id="y")


def test_logprob_resume_drops_partial_images(tmp_path):
    p = tmp_path / "lp.jsonl"
    prov = dict(checkpoint_id="x", model="m", adapter=None, prompt_sha="s")
    _lp_rows(p, [_lp(1, j, **{k: v for k, v in prov.items() if k != "checkpoint_id"}) for j in range(10)] + [_lp(2, j, model="m", adapter=None, prompt_sha="s") for j in range(6)])
    done = prepare_logprob_resume(p, prov, {1: 10, 2: 10})
    assert done == {1}
    assert {json.loads(line)["image_id"] for line in open(p)} == {1}
    with pytest.raises(ValueError, match="prompt_sha"):
        prepare_logprob_resume(p, dict(prov, prompt_sha="other"), {1: 10, 2: 10})


# ----------------------------------------------------------------------------- 5, 12: generation provenance / batches


class FakeVLM:
    model_id, adapter = "fake/model", None

    def __init__(self):
        self.batches = []

    def generate(self, ims, prompt, t, mnt, num_return_sequences=1, seed=None):
        self.batches.append(list(ims))
        return [f"[{prompt}] cap {i}" for i in ims for _ in range(num_return_sequences)]


def test_generation_rows_carry_provenance(tmp_path):
    from contam.generate import run_generation

    out = tmp_path / "g.jsonl"
    run_generation(FakeVLM(), lambda i: i, [2, 1], "caption", "Describe the image.", out, "ck", batch_size=4, progress=False)
    rows = [json.loads(line) for line in open(out)]
    assert all(r["prompt_sha"] == prompt_sha("Describe the image.") and r["model"] == "fake/model" and r["batch_size"] == 4 for r in rows)


def test_generation_refuses_changed_prompt_on_resume(tmp_path):
    from contam.generate import run_generation

    out = tmp_path / "g.jsonl"
    run_generation(FakeVLM(), lambda i: i, [1, 2], "caption", "Describe the image.", out, "ck", progress=False)
    with pytest.raises(ValueError, match="prompt_sha"):
        run_generation(FakeVLM(), lambda i: i, [1, 2, 3], "caption", "Describe the image in one short sentence.", out, "ck", progress=False)


def test_generation_batch_composition_fixed_on_resume(tmp_path):
    from contam.generate import run_generation

    out = tmp_path / "g.jsonl"
    v = FakeVLM()
    run_generation(v, lambda i: i, [0, 1], "caption", "P", out, "ck", batch_size=4, progress=False)  # partial first chunk
    v2 = FakeVLM()
    n = run_generation(v2, lambda i: i, list(range(9, -1, -1)), "caption", "P", out, "ck", batch_size=4, progress=False)
    assert v2.batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert n == 10  # the partial chunk's stale rows are dropped and regenerated with the chunk
    assert len(open(out).readlines()) == 10 and sorted(load_generations(out, checkpoint_id="ck")) == list(range(10))


def _tiny_dataset(tmp_path):
    ann = make_annotations(2, 4, 2, seed=5)
    split = make_split(ann, 1, 2)
    write_json({"meta": {}, "images": ann}, tmp_path / "ann.json")
    write_json(split, tmp_path / "split.json")
    return ann, split


def test_evaluate_checkpoint_checks_ids_and_prompt_sha(tmp_path):
    ann, split = _tiny_dataset(tmp_path)
    prov = dict(checkpoint_id="ck", model="m", adapter=None, prompt_sha="AAA")
    with open(tmp_path / "g.jsonl", "w") as f:
        for a in ann:
            f.write(json.dumps(dict(prov, image_id=a["image_id"], prompt_id="caption", temperature=0.0, sample_idx=0, seed=None, caption=a["refs"][0])) + "\n")
    with open(tmp_path / "lp.jsonl", "w") as f:
        for a in ann:
            for j in range(10):
                f.write(json.dumps(dict(prov, prompt_sha="BBB", image_id=a["image_id"], ref_idx=j, prompt_id="caption", logp_sum=-1.0 - j, n_tokens=3)) + "\n")
    cfg_path, lock, split_path, split_sha = locked_split(tmp_path, split)
    cmd = [sys.executable, "scripts/evaluate_checkpoint.py", "--config", str(cfg_path), "--lock", str(lock), "--annotations", str(tmp_path / "ann.json"), "--split", str(split_path),
           "--generations", str(tmp_path / "g.jsonl"), "--tokenizer", "simple", "--metrics", "cider", "--no-spice-cache", "--out", str(tmp_path / "m.json")]
    r = subprocess.run(cmd + ["--checkpoint-id", "ck", "--logprobs", str(tmp_path / "lp.jsonl")], capture_output=True, text=True)
    assert r.returncode != 0 and "prompt_sha" in r.stderr
    r = subprocess.run(cmd + ["--checkpoint-id", "other"], capture_output=True, text=True)
    assert r.returncode != 0 and "expected 'other'" in r.stderr
    r = subprocess.run(cmd + ["--checkpoint-id", "ck"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = read_json(tmp_path / "m.json")
    assert m["provenance"]["generations"]["prompt_sha"] == "AAA"
    assert m["manifest_split_sha256"] == split_sha and m["gate_valid"] is True


# ----------------------------------------------------------------------------- 6, 12: G1(b) rule and gate inputs


def _run_g1(split, baseline, margins, cfg, cells):
    from contam.eval.gates import g1_report, validate_inputs

    cks = [{"condition": c, "d": d, "seed": s, "metrics": f"/tmp/{c}_{d}_{s}.json"} for (c, d, s, _) in cells]
    ms = [m for *_, m in cells]
    validate_inputs(split, margins, [f"/tmp/b{s}.json" for s in range(5)], baseline, cks, ms, 2)
    return g1_report(split, margins, baseline, cks, ms, cfg)


def test_g1b_secondary_conditions_do_not_gate(g1_setup):
    split, baseline, margins, cfg, ckpt = g1_setup
    cells = [("Caption", 4, 0, ckpt("c0", 5.0)), ("Caption", 4, 1, ckpt("c1", 5.0)), ("Paired", 4, 0, ckpt("p0", 0.0)), ("Paired", 4, 1, ckpt("p1", 0.0))]
    b = _run_g1(split, baseline, margins, cfg, cells)["criteria"]["b"]
    assert b["tests"]["Caption/word_count"]["pass"] and b["tests"]["Caption/word_count"]["role"] == "secondary"
    assert b["verdict"] == "FAIL"


def test_g1b_requires_every_seed(g1_setup):
    split, baseline, margins, cfg, ckpt = g1_setup
    # pooled effect (3.0 + 0.5) / 2 = 1.75 > delta_S = 1 with a tight CI, but seed 1 alone is 0.5
    cells = [("Paired", 4, 0, ckpt("p0", 3.0)), ("Paired", 4, 1, ckpt("p1", 0.5))]
    b = _run_g1(split, baseline, margins, cfg, cells)["criteria"]["b"]
    t = b["tests"]["Paired/word_count"]
    assert t["ci_lo"] > 1.0 and min(t["per_seed_estimates"]) < 1.0
    assert b["verdict"] == "FAIL"
    cells = [("Paired", 4, 0, ckpt("p0", 3.0)), ("Paired", 4, 1, ckpt("p1", 2.5))]
    assert _run_g1(split, baseline, margins, cfg, cells)["criteria"]["b"]["verdict"] == "PASS"


def test_gate_inputs_validation(g1_setup):
    from contam.eval.gates import validate_inputs

    split, baseline, margins, cfg, ckpt = g1_setup
    paths = [f"/tmp/b{s}.json" for s in range(5)]
    cks = [{"condition": "Paired", "d": 4, "seed": 0, "metrics": "/tmp/p.json"}]
    ms = [ckpt("p0", 1.0)]
    with pytest.raises(ValueError, match="margins"):
        validate_inputs(split, dict(margins, split={"name": "B", "content_hash": "x"}), paths, baseline, cks, ms, 1)
    with pytest.raises(ValueError, match="duplicate metrics file"):
        validate_inputs(split, dict(margins, d0_files=paths[:4] + paths[:1]), paths[:4] + paths[:1], baseline, cks, ms, 1)
    with pytest.raises(ValueError, match="5 distinct"):
        validate_inputs(split, dict(margins, d0_files=paths[:4]), paths[:4], baseline[:4], cks, ms, 1)
    with pytest.raises(ValueError, match="identical d=0"):
        validate_inputs(split, margins, paths, baseline[:4] + [dict(baseline[0], checkpoint_id="d0_s9")], cks, ms, 1)
    validate_inputs(split, margins, paths[:4], baseline[:4], cks, ms, 1, allow_partial=True)


# ----------------------------------------------------------------------------- 7: boilerplate patterns

BOILERPLATE_POS = [
    "The image showcases a busy street.",
    "The image contains two dogs.",
    "In the center of the image, there is a vase.",
    "The image features a red car parked outside.",
    "The image depicts a man on a horse.",
    "The image captures a sunset over the ocean.",
    "An image of a bridge at night.",
    "A photo of a red bus.",
    "In this picture, we can see a cat.",
    "This is an image of a kitchen.",
    "The image is a close-up of a flower.",
    "The image is a black and white photo of a street.",
]
BOILERPLATE_NEG = [
    "A sign has a picture of a rooster on it.",
    "A man holding a photo of his family.",
    "A brown dog running across a grassy field.",
    "Two people standing next to a large picture frame.",
    "The woman is taking a photo with her phone.",
    "A framed picture is hanging on the wall.",
]


@pytest.mark.parametrize("text", BOILERPLATE_POS)
def test_boilerplate_positive(text):
    assert caption_style(text, compile_patterns(load_config()["style"]["boilerplate_patterns"]))["boilerplate"] == 1


@pytest.mark.parametrize("text", BOILERPLATE_NEG)
def test_boilerplate_negative(text):
    assert caption_style(text, compile_patterns(load_config()["style"]["boilerplate_patterns"]))["boilerplate"] == 0


def test_boilerplate_config_matches_module_default():
    assert list(DEFAULT_BOILERPLATE_PATTERNS) == load_config()["style"]["boilerplate_patterns"]


# ----------------------------------------------------------------------------- 8: ref normalization


def test_trailing_not_sign_stripped(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from contam.data.nocaps import convert_parquet, load_annotations, load_annotations_meta, normalize_ref

    assert normalize_ref("A man riding a wave¬ ") == "A man riding a wave"
    assert normalize_ref("A man riding a wave¬", strip_trailing_not_sign=False) == "A man riding a wave¬"
    assert normalize_ref("A person wearing¬ a helmet.") == "A person wearing¬ a helmet."
    t = pa.table({"image_id": [0], "image_file_name": ["0.jpg"], "image_open_images_id": ["oi0"], "annotations_captions": [[f"Ref {k}¬" for k in range(10)]]})
    pq.write_table(t, tmp_path / "validation-00000-of-00001.parquet")
    convert_parquet(tmp_path, {"oi0": "in"}, tmp_path / "ann.json", None)
    assert load_annotations(tmp_path / "ann.json")[0]["refs"][3] == "Ref 3"
    assert load_annotations_meta(tmp_path / "ann.json")["ref_normalization"]["strip_trailing_not_sign"] is True


# ----------------------------------------------------------------------------- 9: LoRA verification


def _tiny_lora_setup(tmp_path, init_lora_weights):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.q_proj(x)

    torch.manual_seed(0)
    m = peft.get_peft_model(Tiny(), peft.LoraConfig(r=2, target_modules=["q_proj"], init_lora_weights=init_lora_weights))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n and init_lora_weights is False:
                p.normal_()
    m.save_pretrained(tmp_path / "ad")
    return Tiny, tmp_path / "ad"


def test_load_lora_rejects_noop_adapter(tmp_path):
    from contam.models import load_lora

    Tiny, ad = _tiny_lora_setup(tmp_path, True)  # default init: lora_B = 0
    with pytest.raises(RuntimeError, match="no effect"):
        load_lora(Tiny(), str(ad))


def test_load_lora_accepts_effective_adapter(tmp_path):
    torch = pytest.importorskip("torch")
    from contam.models import load_lora

    Tiny, ad = _tiny_lora_setup(tmp_path, False)
    base = Tiny()
    w0 = base.q_proj.weight.detach().clone()
    merged = load_lora(base, str(ad))
    assert not torch.equal(merged.q_proj.weight, w0)


# ----------------------------------------------------------------------------- 11: SPICE cache concurrency


def _cache_worker(path, k):
    c = cm.SpiceCache(path)
    for j in range(20):
        c.put(f"hyp {k} {j}", ["ref"], float(k))
        c.save()


def test_spice_cache_merges_concurrent_writers(tmp_path):
    p = tmp_path / "c.json"
    a, b = cm.SpiceCache(p), cm.SpiceCache(p)  # both loaded before either saves
    a.put("x", ["r"], 1.0)
    a.save()
    b.put("y", ["r"], 2.0)
    b.save()
    c = cm.SpiceCache(p)
    assert c.get("x", ["r"]) == 1.0 and c.get("y", ["r"]) == 2.0


def test_spice_cache_multiprocess(tmp_path):
    p = str(tmp_path / "c.json")
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_cache_worker, args=(p, k)) for k in range(4)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    assert all(pr.exitcode == 0 for pr in procs)
    c = cm.SpiceCache(p)
    assert len(c.data) == 80
    assert not list(tmp_path.glob("*.tmp"))
