"""HF vision-language model wrapper: caption generation and teacher-forced reference scoring.

One object serves both uses so generation and scoring share EXACTLY the same prompt template:

    prefix(prompt) = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
        add_generation_prompt=True)

For llava-hf/llava-1.5-7b-hf this is ``"USER: <image>\\n{prompt} ASSISTANT:"``.

Teacher-forced scoring (``RefScorer`` protocol, ``contam.eval.memory``)
----------------------------------------------------------------------
The scored sequence is ``prefix + " " + ref.strip()`` followed by ONE end token (the tokenizer's
EOS by default, ``end_token`` overrides it, e.g. ``<end_of_utterance>`` for SmolVLM). Log-probs are
summed over the caption tokens AND the end token; prompt, BOS and image tokens are excluded.
``n_tokens`` counts the same positions (caption tokens + 1). Including the end token means a
reference scores high only if the model would also stop after it, matching what T=0 generation
must do to reproduce it verbatim. The same rule applies to every reference and every checkpoint.
The training target format (``contam.train.build_example``) uses the same prefix, separator and
end token, built from ``turn_prefix`` / ``end_token_id`` / ``configure_processor`` below.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class HFVLM:
    def __init__(
        self,
        model_id: str,
        adapter: str | None = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
        end_token: str | None = None,
        merge_adapter: bool = True,
    ):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.device = device
        self.model_id = model_id
        self.adapter = str(adapter) if adapter else None
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=getattr(torch, dtype)).to(device)
        if adapter:
            self.model = load_lora(self.model, adapter, merge_adapter)
        self.model.eval()
        configure_processor(self.processor, self.model.config)
        self.end_token_id = end_token_id(self.processor.tokenizer, end_token)

    # ------------------------------------------------------------------ prompt

    def prefix(self, prompt: str) -> str:
        return prefix_for(self.processor, prompt)

    def _encode(self, images, texts, padding_side: str):
        tok = self.processor.tokenizer
        old = tok.padding_side
        tok.padding_side = padding_side
        try:
            enc = self.processor(images=list(images), text=list(texts), return_tensors="pt", padding=True)
        finally:
            tok.padding_side = old
        return enc

    # ------------------------------------------------------------------ generation

    def generate(
        self,
        images: Sequence,
        prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 300,
        num_return_sequences: int = 1,
        seed: int | None = None,
    ) -> list[str]:
        """Captions for ``images`` (``num_return_sequences`` per image, image-major order).

        T=0 is greedy. T>0 is pure ancestral sampling (top_k=0, top_p=1) seeded with ``seed``.
        """
        torch = self.torch
        enc = self._encode(images, [self.prefix(prompt)] * len(images), "left").to(self.device)
        enc["pixel_values"] = enc["pixel_values"].to(self.model.dtype)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.processor.tokenizer.pad_token_id, num_return_sequences=num_return_sequences)
        if temperature > 0:
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            kw.update(do_sample=True, temperature=float(temperature), top_k=0, top_p=1.0)
        else:
            kw.update(do_sample=False, temperature=None, top_k=None, top_p=None)
        with torch.inference_mode():
            out = self.model.generate(**enc, **kw)
        new = out[:, enc["input_ids"].shape[1] :]
        return [t.strip() for t in self.processor.batch_decode(new, skip_special_tokens=True)]

    # ------------------------------------------------------------------ teacher forcing

    def _teacher_forced_inputs(self, images: Sequence, prompt: str, refs: Sequence[str]):
        """Model inputs for ``prefix + " " + ref`` + end token, and each row's (prompt, full) lengths.

        Row i's caption tokens are ``input_ids[i, plen[i]:flen[i]]`` and its end token sits at ``flen[i]``.
        """
        torch = self.torch
        pre = self.prefix(prompt)
        full_texts = [f"{pre} {r.strip()}" for r in refs]
        enc_p = self._encode(images, [pre] * len(refs), "right")
        enc_f = self._encode(images, full_texts, "right")
        plen = enc_p["attention_mask"].sum(1)
        flen = enc_f["attention_mask"].sum(1)
        ids = enc_f["input_ids"]
        for i in range(len(refs)):
            if not torch.equal(ids[i, : plen[i]], enc_p["input_ids"][i, : plen[i]]):
                raise RuntimeError("prompt tokens are not a prefix of prompt+caption tokens; masking would be wrong")
        # Append the end token after each row's last real token.
        B, T = ids.shape
        pad = self.processor.tokenizer.pad_token_id
        ids2 = torch.full((B, T + 1), pad, dtype=ids.dtype)
        am2 = torch.zeros((B, T + 1), dtype=enc_f["attention_mask"].dtype)
        ids2[:, :T], am2[:, :T] = ids, enc_f["attention_mask"]
        ids2[torch.arange(B), flen] = self.end_token_id
        am2[torch.arange(B), flen] = 1
        inputs = {k: v for k, v in enc_f.items() if k not in ("input_ids", "attention_mask")}
        inputs["input_ids"], inputs["attention_mask"] = ids2, am2
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        return inputs, plen, flen

    def _final_hidden(self, inputs):
        """Final-layer hidden states (the lm_head input, after the last norm) for ``inputs``."""
        torch = self.torch
        head = self.model.get_output_embeddings()
        captured = {}
        hook = head.register_forward_hook(lambda mod, args, out: captured.__setitem__("h", args[0]))
        try:
            with torch.inference_mode():
                self.model(**inputs)
        finally:
            hook.remove()
        return captured["h"]

    def score(self, images: Sequence, prompt: str, refs: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Summed log p of each ``refs[k]`` (+ end token) given ``images[k]`` and ``prompt``."""
        torch = self.torch
        inputs, plen, flen = self._teacher_forced_inputs(images, prompt, refs)
        B = len(refs)
        # Logits are recomputed in fp32 from the final hidden state (the lm_head input) so that
        # teacher-forced log-probs are not rounded to bf16.
        head = self.model.get_output_embeddings()
        hidden = self._final_hidden(inputs)
        w32 = head.weight.float()
        b32 = head.bias.float() if getattr(head, "bias", None) is not None else None
        logp = np.zeros(B)
        ntok = np.zeros(B, dtype=int)
        with torch.inference_mode():
            for i in range(B):
                s, e = int(plen[i]), int(flen[i])  # caption tokens are ids2[s:e], end token at ids2[e]
                tgt = inputs["input_ids"][i, s : e + 1]
                logits = torch.nn.functional.linear(hidden[i, s - 1 : e].float(), w32, b32)
                lp = torch.log_softmax(logits, dim=-1)
                logp[i] = float(lp.gather(1, tgt[:, None]).sum())
                ntok[i] = e + 1 - s
        return logp, ntok

    def embed(self, images: Sequence, prompt: str, refs: Sequence[str]) -> np.ndarray:
        """Sample embeddings for KDS (``contam.eval.detectors``): the mean final-layer hidden state over
        the caption tokens and the end token of ``prefix + " " + refs[k]``, the positions ``score`` scores."""
        inputs, plen, flen = self._teacher_forced_inputs(images, prompt, refs)
        hidden = self._final_hidden(inputs)
        return np.stack([hidden[i, int(plen[i]) : int(flen[i]) + 1].float().mean(0).cpu().numpy() for i in range(len(refs))])


def load_lora(model, adapter: str, merge: bool = True):
    """Attach a PEFT LoRA adapter and verify it actually changes the model.

    Raises if no LoRA module matched the base model (e.g. wrong ``target_modules`` names), if the
    adapter file's keys differ from the injected LoRA parameters (some weights did not load), or if
    every ``lora_B`` is zero / merging leaves the weights unchanged. On failure the injected LoRA
    layers are removed again, so ``model`` is left as it was. Some all-zero ``lora_B`` only warn: a
    trained adapter can legitimately have modules that got no gradient (e.g. LLaVA's last CLIP layer
    with ``vision_feature_layer=-2``).
    """
    import warnings

    import torch
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from peft.utils import get_peft_model_state_dict, load_peft_weights

    pm = PeftModel.from_pretrained(model, adapter)
    try:
        lora = [(n, m) for n, m in pm.named_modules() if isinstance(m, LoraLayer)]
        if not lora:
            raise RuntimeError(f"adapter {adapter}: no LoRA modules matched the base model")
        file_keys, model_keys = set(load_peft_weights(adapter)), set(get_peft_model_state_dict(pm))
        if file_keys != model_keys:
            missing, unexpected = sorted(model_keys - file_keys), sorted(file_keys - model_keys)
            raise RuntimeError(f"adapter {adapter}: weights do not match the injected LoRA modules ({len(missing)} missing, e.g. {missing[:2]}; {len(unexpected)} unexpected, e.g. {unexpected[:2]})")
        active = [(n, m) for n, m in lora if any(float(b.weight.abs().max()) > 0 for b in m.lora_B.values())]
        if not active:
            raise RuntimeError(f"adapter {adapter}: every lora_B is zero; the adapter has no effect")
        if len(active) < len(lora):
            zero = sorted(set(n for n, _ in lora) - set(n for n, _ in active))
            warnings.warn(f"adapter {adapter}: lora_B is zero in {len(zero)} of {len(lora)} LoRA modules (no effect there): {zero}")
    except RuntimeError:
        pm.unload()
        raise
    if not merge:
        return pm
    name, mod = active[0]
    before = mod.base_layer.weight.detach().clone()
    merged = pm.merge_and_unload()
    after = merged.get_submodule(name.removeprefix("base_model.model.")).weight
    if torch.equal(before, after):
        raise RuntimeError(f"adapter {adapter}: merging did not change {name}; the adapter has no effect")
    return merged


def configure_processor(processor, model_config) -> None:
    """Make the processor expand <image> into the full run of image tokens (needed for masking) and pad."""
    if getattr(processor, "patch_size", "missing") is None and hasattr(model_config, "vision_config"):
        processor.patch_size = model_config.vision_config.patch_size
    if getattr(processor, "vision_feature_select_strategy", "missing") is None:
        processor.vision_feature_select_strategy = getattr(model_config, "vision_feature_select_strategy", "default")
    tok = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token


def end_token_id(tokenizer, end_token: str | None = None) -> int:
    """The token appended after every scored reference / trained answer (EOS unless ``end_token``)."""
    tid = tokenizer.eos_token_id if end_token is None else tokenizer.convert_tokens_to_ids(end_token)
    if tid is None or tid == tokenizer.unk_token_id:
        raise ValueError(f"end token {end_token!r} is not in the vocabulary")
    return tid


def turn_prefix(processor, prompt: str, image: bool = True) -> str:
    """One user turn up to the assistant's answer: ``"USER: <image>\\n{prompt} ASSISTANT:"`` for llava-hf."""
    content = ([{"type": "image"}] if image else []) + [{"type": "text", "text": prompt}]
    return processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)


def prefix_for(processor, prompt: str) -> str:
    """The generation prompt string for ``prompt`` (shared by generation and scoring)."""
    return turn_prefix(processor, prompt, image=True)
