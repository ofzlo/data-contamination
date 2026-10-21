"""Regression tests for the re-verification findings (N1-N10). Each fails on the pre-fix code."""

import copy
import json
import subprocess
import sys

import pytest

from contam.config import REPO_ROOT, write_json
from contam.data.splits import annotations_hash, make_split
from contam.eval.evaluate import evaluate_checkpoint
from contam.eval.gates import g1_report, validate_inputs
from tests.conftest import make_annotations
from tests.test_review_fixes import FakeVLM, _tiny_dataset

PATHS = [f"/tmp/b{s}.json" for s in range(5)]


def _cells(ckpt, n_seeds=2, drop=3.0):
    cks = [{"condition": "Paired", "d": 4, "seed": s, "metrics": f"/tmp/Paired_4_{s}.json"} for s in range(n_seeds)]
    return cks, [ckpt(f"p{s}", drop) for s in range(n_seeds)]


# ----------------------------------------------------------------------------- N1, N2, N3, N8: gate inputs


def test_gate_rejects_non_gate_valid_margins(g1_setup):
    split, baseline, margins, cfg, ckpt = g1_setup
    cks, ms = _cells(ckpt)
    for bad in (dict(margins, gate_valid=False), {k: v for k, v in margins.items() if k != "gate_valid"}):
        with pytest.raises(ValueError, match="not gate-valid"):
            validate_inputs(split, bad, PATHS, baseline, cks, ms, 2)
    with pytest.raises(ValueError, match="d=0 files"):
        validate_inputs(split, dict(margins, d0_files=PATHS[:4] + ["/tmp/other.json"]), PATHS, baseline, cks, ms, 2)
    bad = dict(margins, gate_valid=False)
    validate_inputs(split, bad, PATHS, baseline, cks, ms, 2, allow_partial=True)
    with pytest.warns(UserWarning):
        assert g1_report(split, bad, baseline, cks, ms, cfg, allow_partial=True)["gate_valid"] is False


def test_compute_margins_records_realpaths(tmp_path):
    ann, split = _tiny_dataset(tmp_path)
    for s in range(5):
        m = evaluate_checkpoint({a["image_id"]: a["refs"][s] for a in ann}, ann, split, metrics=["cider"], tokenizer="simple", checkpoint_id=f"d0_s{s}")
        write_json(m, tmp_path / f"d0_s{s}.json")
    cmd = [sys.executable, str(REPO_ROOT / "scripts/compute_margins.py"), "--out", "m.json", "--d0"] + [f"d0_s{s}.json" for s in range(5)]
    r = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    m = json.load(open(tmp_path / "m.json"))
    assert m["gate_valid"] is True and m["d0_files"] == [str((tmp_path / f"d0_s{s}.json").resolve()) for s in range(5)]


def test_gate_requires_seeds_per_cell(g1_setup):
    split, baseline, margins, cfg, ckpt = g1_setup
    assert cfg["gates"]["g1_seeds_per_cell"] == 2
    cks, ms = _cells(ckpt, n_seeds=1)
    with pytest.raises(ValueError, match="1 seeds, expected 2"):
        validate_inputs(split, margins, PATHS, baseline, cks, ms, cfg["gates"]["g1_seeds_per_cell"])
    validate_inputs(split, margins, PATHS, baseline, cks, ms, 2, allow_partial=True)


@pytest.mark.parametrize("field", ["prompt_sha", "model", "batch_size", "max_new_tokens"])
def test_gate_requires_shared_provenance(g1_setup, field):
    split, baseline, margins, cfg, ckpt = g1_setup
    cks, ms = _cells(ckpt)
    other = copy.deepcopy(ms[1])
    other["provenance"]["generations"][field] = "OTHER"
    with pytest.raises(ValueError, match=field):
        validate_inputs(split, margins, PATHS, baseline, cks, [ms[0], other], 2)
    missing = copy.deepcopy(baseline[0])
    del missing["provenance"]["generations"][field]
    with pytest.raises(ValueError, match=field):
        validate_inputs(split, margins, PATHS, [missing] + baseline[1:], cks, ms, 2)


def test_gate_rejects_baseline_reused_as_checkpoint(g1_setup):
    split, baseline, margins, cfg, ckpt = g1_setup
    cks, ms = _cells(ckpt)
    with pytest.raises(ValueError, match="metrics file"):
        validate_inputs(split, margins, PATHS, baseline, [dict(cks[0], metrics=PATHS[0]), cks[1]], ms, 2)
    with pytest.raises(ValueError, match="checkpoint_id"):
        validate_inputs(split, margins, PATHS, baseline, cks, [dict(ms[0], checkpoint_id=baseline[0]["checkpoint_id"]), ms[1]], 2)


# ----------------------------------------------------------------------------- N4: annotations hash


def _edited(ann):
    ann2 = copy.deepcopy(ann)
    ann2[0]["refs"][0] += " edited"
    return ann2


def test_split_records_annotations_hash():
    ann = make_annotations(2, 4, 2, seed=5)
    assert make_split(ann, 1, 2)["annotations_hash"] == annotations_hash(ann) != annotations_hash(_edited(ann))


def test_evaluate_checkpoint_rejects_other_annotations():
    ann = make_annotations(2, 4, 2, seed=5)
    split = make_split(ann, 1, 2)
    gens = {a["image_id"]: a["refs"][0] for a in ann}
    with pytest.raises(ValueError, match="annotations"):
        evaluate_checkpoint(gens, _edited(ann), split, metrics=["cider"], tokenizer="simple")


def test_generate_and_score_refs_reject_other_annotations(tmp_path):
    from contam.generate import main as gen_main

    ann, split = _tiny_dataset(tmp_path)
    write_json({"meta": {}, "images": _edited(ann)}, tmp_path / "ann2.json")
    common = ["--annotations", str(tmp_path / "ann2.json"), "--split", str(tmp_path / "split.json"), "--checkpoint-id", "ck", "--out", str(tmp_path / "o.jsonl")]
    with pytest.raises(ValueError, match="annotations"):
        gen_main(common)
    r = subprocess.run([sys.executable, "scripts/score_refs.py"] + common, capture_output=True, text=True)
    assert r.returncode != 0 and "annotations" in r.stderr.splitlines()[-1]


# ----------------------------------------------------------------------------- N5: generation resume


def test_generation_resume_after_truncated_run_regenerates_partial_chunk(tmp_path):
    from contam.generate import run_generation

    out = tmp_path / "g.jsonl"
    run_generation(FakeVLM(), lambda i: i, list(range(6)), "caption", "P", out, "ck", batch_size=4, progress=False)  # like --max-images 6
    v = FakeVLM()
    assert run_generation(v, lambda i: i, list(range(10)), "caption", "P", out, "ck", batch_size=4, progress=False) == 6
    assert v.batches == [[4, 5, 6, 7], [8, 9]]
    rows = [json.loads(line) for line in open(out)]
    assert sorted(r["image_id"] for r in rows) == list(range(10))
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("kw", [dict(batch_size=8), dict(max_new_tokens=50)])
def test_generation_refuses_changed_batch_size_or_max_new_tokens(tmp_path, kw):
    from contam.generate import run_generation

    out = tmp_path / "g.jsonl"
    run_generation(FakeVLM(), lambda i: i, [0, 1], "caption", "P", out, "ck", batch_size=4, max_new_tokens=300, progress=False)
    with pytest.raises(ValueError, match=next(iter(kw))):
        run_generation(FakeVLM(), lambda i: i, [0, 1, 2], "caption", "P", out, "ck", **{"batch_size": 4, "max_new_tokens": 300, **kw}, progress=False)


# ----------------------------------------------------------------------------- N6: LoRA load result


def _two_module_adapter(tmp_path, rename=None):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from safetensors.torch import load_file, save_file

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8)
            self.v_proj = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.v_proj(self.q_proj(x))

    torch.manual_seed(0)
    m = peft.get_peft_model(Tiny(), peft.LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.normal_()
    d = tmp_path / "ad"
    m.save_pretrained(d)  # config keeps init_lora_weights=True, like a real trained adapter
    if rename:
        f = d / "adapter_model.safetensors"
        save_file({k.replace(*rename): v for k, v in load_file(f).items()}, f)
    return Tiny, d


@pytest.mark.parametrize("rename", [("v_proj", "vv_proj"), ("_proj", "_xx")])
def test_load_lora_rejects_partially_loaded_adapter_and_unloads(tmp_path, rename):
    from peft.tuners.lora import LoraLayer

    from contam.models import load_lora

    Tiny, ad = _two_module_adapter(tmp_path, rename)
    base = Tiny()
    with pytest.raises(RuntimeError, match="do not match"):
        load_lora(base, str(ad))
    assert not any(isinstance(m, LoraLayer) for m in base.modules())
    Tiny, ad = _two_module_adapter(tmp_path / "ok")
    load_lora(Tiny(), str(ad))


# ----------------------------------------------------------------------------- N10: null simulation script


def test_sim_g1b_null_runs():
    r = subprocess.run([sys.executable, "scripts/sim_g1b_null.py", "--n-images", "90", "--n-boot", "20", "--reps", "2", "--sigma-seed", "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[-1].split()[0] == "1.00"


# ----------------------------------------------------------------------------- round 2: R1, L2


def test_load_lora_accepts_trained_adapter_with_unused_modules(tmp_path):
    """LLaVA with vision_feature_layer=-2: the last CLIP layer gets no gradient, so its lora_B stay zero."""
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from peft.tuners.lora import LoraLayer
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

    from contam.models import load_lora

    cfg = LlavaConfig(
        vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=2, image_size=28, patch_size=14, projection_dim=32),
        text_config=LlamaConfig(vocab_size=200, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, pad_token_id=0),
        image_token_index=199, vision_feature_layer=-2, vision_feature_select_strategy="default",
    )

    def base():
        torch.manual_seed(0)
        return LlavaForConditionalGeneration(cfg).eval()

    m = peft.get_peft_model(base(), peft.LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    ids = torch.tensor([[1, 199] + list(range(2, 12))])
    for _ in range(3):
        m(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=torch.randn(1, 3, 28, 28), labels=ids).loss.backward()
        opt.step()
        opt.zero_grad()
    zero = [n for n, mod in m.named_modules() if isinstance(mod, LoraLayer) and all(float(b.weight.abs().max()) == 0 for b in mod.lora_B.values())]
    assert zero and all("vision_tower" in n for n in zero)
    m.save_pretrained(tmp_path / "ad")
    b = base()
    w0 = b.language_model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    with pytest.warns(UserWarning, match=f"lora_B is zero in {len(zero)} of"):
        merged = load_lora(b, str(tmp_path / "ad"))
    assert not torch.equal(merged.language_model.model.layers[0].self_attn.q_proj.weight, w0)


def test_generations_provenance_records_batch_settings(tmp_path):
    from contam.eval.evaluate import generations_provenance
    from contam.generate import run_generation

    out = tmp_path / "g.jsonl"
    run_generation(FakeVLM(), lambda i: i, [0, 1, 2], "caption", "P", out, "ck", batch_size=2, max_new_tokens=77, progress=False)
    prov = generations_provenance(out)
    assert prov["batch_size"] == 2 and prov["max_new_tokens"] == 77
