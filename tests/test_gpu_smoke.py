"""GPU smoke test of HFVLM generation + teacher-forced scoring. Skipped unless CONTAM_GPU_SMOKE=1."""

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get("CONTAM_GPU_SMOKE") != "1", reason="set CONTAM_GPU_SMOKE=1 (needs GPU + weights)")]


def test_generate_and_score_consistency():
    from PIL import Image

    from contam.models import HFVLM

    vlm = HFVLM(os.environ.get("CONTAM_SMOKE_MODEL", "llava-hf/llava-1.5-7b-hf"))
    im = Image.new("RGB", (336, 336), (200, 30, 30))
    cap = vlm.generate([im], "Describe the image in one short sentence.", 0.0, max_new_tokens=40)[0]
    assert cap
    lp, n = vlm.score([im, im, im], "Describe the image in one short sentence.", [cap, "A plate of spaghetti on a wooden table.", cap])
    assert lp[0] == pytest.approx(lp[2], abs=1e-3)  # deterministic
    assert lp[0] > lp[1]  # the model's own greedy caption is more likely than an unrelated one
    assert np.all(lp < 0) and np.all(n > 1)
    # batching must not change scores
    lp1, n1 = vlm.score([im], "Describe the image in one short sentence.", ["A plate of spaghetti on a wooden table."])
    assert n1[0] == n[1] and lp1[0] == pytest.approx(lp[1], abs=0.05)


def test_fp32_scoring_and_lora_on_one_model(tmp_path):
    """fp32 lm_head logits (item 10) and LoRA verification on the real model (item 9); one model copy."""
    import torch
    from peft import LoraConfig, get_peft_model
    from peft.tuners.lora import LoraLayer
    from PIL import Image

    from contam.models import HFVLM, load_lora

    vlm = HFVLM(os.environ.get("CONTAM_SMOKE_MODEL", "llava-hf/llava-1.5-7b-hf"))
    im = Image.new("RGB", (336, 336), (30, 120, 200))
    prompt, ref = "Describe the image.", "A plain blue square."
    lp, n = vlm.score([im], prompt, [ref])
    # manual reference: full fp32 log-softmax of the lm_head applied in fp32
    pre = vlm.prefix(prompt)
    enc = vlm._encode([im], [f"{pre} {ref}"], "right").to("cuda")
    plen = vlm._encode([im], [pre], "right")["input_ids"].shape[1]
    ids = torch.cat([enc["input_ids"], torch.tensor([[vlm.end_token_id]], device="cuda")], 1)
    head, cap = vlm.model.get_output_embeddings(), {}
    h = head.register_forward_hook(lambda m, a, o: cap.__setitem__("h", a[0]))
    with torch.inference_mode():
        out = vlm.model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=enc["pixel_values"].to(torch.bfloat16))
    h.remove()
    tgt = ids[0, plen:]
    fp32 = torch.log_softmax(cap["h"][0, plen - 1 : -1].float() @ head.weight.float().T, -1).gather(1, tgt[:, None]).sum()
    bf16 = torch.log_softmax(out.logits[0, plen - 1 : -1].float(), -1).gather(1, tgt[:, None]).sum()
    print(f"score={lp[0]:.5f} fp32-manual={float(fp32):.5f} bf16-logits={float(bf16):.5f} ntok={n[0]}")
    assert lp[0] == pytest.approx(float(fp32), abs=1e-3)
    assert n[0] == len(tgt)

    # LoRA: a zero-init adapter must be rejected, a non-zero one must change scores.
    for init, expect_ok in ((True, False), (False, True)):
        pm = get_peft_model(vlm.model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"], init_lora_weights=init))
        d = tmp_path / f"ad_{init}"
        pm.save_pretrained(d)
        vlm.model = pm.unload()
        if not expect_ok:
            with pytest.raises(RuntimeError, match="no effect"):
                load_lora(vlm.model, str(d))
            # the failed load unloads its LoRA layers, leaving the base model as it was
            assert not any(isinstance(m, LoraLayer) for m in vlm.model.modules())
            assert vlm.score([im], prompt, [ref])[0][0] == pytest.approx(lp[0], abs=1e-4)
        else:
            vlm.model = load_lora(vlm.model, str(d))
            lp2, _ = vlm.score([im], prompt, [ref])
            print(f"with random LoRA: {lp2[0]:.4f} vs base {lp[0]:.4f}")
            assert abs(lp2[0] - lp[0]) > 1e-3
