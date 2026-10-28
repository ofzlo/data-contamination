"""Single-GPU LoRA SFT of the main VLM on one LLaVA-format training file. CLI: scripts/train_lora.py.

Recipe (``training`` in configs/default.yaml; LLaVA-1.5 LoRA finetune defaults): LoRA (r 128,
alpha 256, dropout 0.05) on the language model's q/k/v/o/gate/up/down projections only; the vision
tower is frozen; the projector is trained in full (PEFT ``modules_to_save``) at ``projector_lr``.
AdamW (wd 0), cosine schedule with 3% warmup, 1 epoch, global batch 128 by gradient accumulation,
max 2048 tokens. Trainable weights are fp32 masters; on GPU the forward runs in ``models.dtype``
(bf16) under autocast, with gradient checkpointing. The training seed (``training.seeds[slot]``)
sets the LoRA init and the data order.

Loss (``training.loss_normalization``; token cross-entropy computed here, not HF's ``.loss``)
-----------------------------------------------------------------------------------------------
``per_example_mean`` (default): each example's loss is the mean over its own supervised tokens, and
the optimizer step's loss is the mean over its B examples (B = global_batch_size, or the real count
in the last, partial step). Every example weighs exactly 1/B, whatever micro-batch it lands in and
whatever its length, so an exposed example's dose does not depend on its batchmates. Alternatives:
``token_mean_global`` (sum of token losses / supervised tokens of the whole step) and
``micro_batch_mean`` (each micro-batch's token mean / gradient accumulation, the official LLaVA-1.5
recipe: a mean of per-device means). An example with no supervised token contributes 0. This
deviates from LLaVA-1.5 (prereg Section 3) and must not change between runs (it is in the config hash).

Data order (``training.micro_batch_order``): the training seed permutes the examples and the
permutation is cut into optimizer steps of global_batch_size. ``length_sorted`` (default) sorts the
examples of each step by token length (longest first) before cutting it into micro-batches, which
cuts padding (LLaVA-665K: 1.54x -> 1.11x padded/real tokens at micro-batch 16); ``random`` cuts the
permutation as is. Either way a step holds the same examples. Lengths come from
``example_stats`` (tokenizer only) and every micro-batch is checked against them.

Example format (the same template as generation and teacher forcing, ``contam.models``)
-----------------------------------------------------------------------------------------
A conversation ``[(q1, a1), (q2, a2), ...]`` becomes, turn by turn,

    turn_prefix(q1, image) + " " + a1.strip()  <end>   turn_prefix(q2) + " " + a2.strip()  <end> ...

where ``turn_prefix`` is ``"USER: <image>\\n{q} ASSISTANT:"`` for the first turn of an image example
and ``"USER: {q} ASSISTANT:"`` otherwise, and ``<end>`` is the scoring end token (EOS). Each turn is
tokenized on its own exactly as ``HFVLM.score`` tokenizes ``prefix + " " + ref`` (the first turn
through the processor, which adds BOS and expands ``<image>`` into the image tokens; later turns
without special tokens). The loss covers each answer's tokens and its end token only. So a
one-turn example (p_cap, ref) is token for token the sequence ``HFVLM.score`` scores, with the loss
on exactly the scored positions. ``<image>`` markers in the LLaVA text are removed; the image goes
in through the processor, as at eval. There is no system prompt, as at eval (``turn_prefix`` is the
llava-hf chat template, which has none). Turns are joined as in the official LLaVA-1.5 ``v1``
format: each answer ends with ``</s>`` and the next ``USER:`` follows it directly (tokenized as a
separate piece, without BOS).

Resume
------
Checkpoints (every ``checkpoint_every_steps`` optimizer steps, at the end of a ``--max-steps`` run,
after the last step (before the adapter export, so a death during the export resumes at the end),
and on SIGTERM / SIGUSR1 / SIGHUP at the next micro-batch boundary) hold the fp32 trainable weights
(LoRA + projector), the optimizer, LR scheduler and RNG states (python / numpy / torch / cuda), the
optimizer step, the micro-step inside the accumulation (with the accumulated gradients), the data
position and the config hash. They are written to a temporary directory, fsynced, marked COMPLETE
and renamed; the last ``keep_checkpoints`` are kept. A start resumes from the latest complete
checkpoint and refuses one whose config hash differs. A finished run writes ``adapter/`` (PEFT
format, in models.dtype, verified with ``contam.models.load_lora``), ``manifest.json`` and ``DONE``;
running it again is a no-op. After DONE the resume checkpoints are deleted unless
``keep_checkpoints_after_done``. The stop handlers are installed when ``train`` starts (before the slow
imports and the model load); a signal outside the training loop stops at the next safe point
without a new checkpoint. After a signal the process exits with EXIT_PREEMPTED (75). DataLoader
workers ignore the stop signals (SLURM signals the whole process group), so only the main process
reacts. A checkpoint holds no absolute paths or device ids (tensors load to CPU and move to the
model), so a run can resume in a fresh process on another node; the config hash covers the data
by sha256, not by path. The config hash also covers the manifest provenance (whether the lock was
checked, and the frozen split / general-order sha256s), the torch / transformers / peft versions, the
model snapshot commit and a hash of the processor (chat template, vocabulary, image processor), so a
resume or a DONE no-op under a different environment or provenance is refused.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import signal
import socket
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

from contam.config import REPO_ROOT, content_hash, file_sha256, read_json
from contam.models import configure_processor, end_token_id, load_lora, turn_prefix

FORMAT_VERSION = 2
EXIT_PREEMPTED = 75
EXIT_BUSY = 76
IGNORE = -100
STOP_SIGNALS = ("SIGTERM", "SIGUSR1", "SIGHUP")
# training-config keys that do not change the result (excluded from the config hash)
NON_SEMANTIC = ("checkpoint_every_steps", "keep_checkpoints", "keep_checkpoints_after_done", "num_workers", "seeds", "general_seed",
                "run_lock_stale_s")
LOSS_NORMALIZATIONS = ("per_example_mean", "token_mean_global", "micro_batch_mean")
MICRO_BATCH_ORDERS = ("length_sorted", "random")


class Preempted(Exception):
    """A stop signal (or ``stop_after_micro``) arrived; inside the training loop a checkpoint was saved."""


class RunBusy(Exception):
    """Another process holds the run lock of this out dir (or a trainer without the lock wrote to it recently)."""


# ----------------------------------------------------------------------------- example format


def conversation_turns(example: dict) -> list[tuple[str, str]]:
    """``[(question, answer)]`` from a LLaVA ``conversations`` list (human/gpt alternating)."""
    conv = example["conversations"]
    if len(conv) % 2 or not conv:
        raise ValueError(f"example {example.get('id')}: conversation has {len(conv)} messages, expected human/gpt pairs")
    turns = []
    for h, g in zip(conv[::2], conv[1::2]):
        if h["from"] != "human" or g["from"] != "gpt":
            raise ValueError(f"example {example.get('id')}: expected human/gpt pairs, got {h['from']}/{g['from']}")
        turns.append((h["value"].replace("<image>", "").strip(), g["value"].strip()))
    return turns


def _turn_ids(processor, prefix: str, answer: str, image, first: bool) -> tuple[list[int], int, dict]:
    """Token ids of ``prefix + " " + answer`` and the prompt length, tokenized as HFVLM.score does."""
    tok = processor.tokenizer
    text = f"{prefix} {answer}"
    if image is not None:
        full = processor(images=[image], text=[text], return_tensors="pt")
        pre = processor(images=[image], text=[prefix], return_tensors="pt")["input_ids"][0].tolist()
        ids, extra = full["input_ids"][0].tolist(), {"pixel_values": full["pixel_values"]}
    else:
        ids = tok(text, add_special_tokens=first)["input_ids"]
        pre = tok(prefix, add_special_tokens=first)["input_ids"]
        extra = {}
    if ids[: len(pre)] != pre:
        raise RuntimeError("prompt tokens are not a prefix of prompt+answer tokens; masking would be wrong")
    return ids, len(pre), extra


def build_example(processor, example: dict, image, end_id: int, max_len: int) -> dict:
    """``{input_ids, labels, pixel_values?}`` for one LLaVA example (see the module docstring)."""
    ids: list[int] = []
    labels: list[int] = []
    extra: dict = {}
    for k, (q, a) in enumerate(conversation_turns(example)):
        img = image if k == 0 else None
        t_ids, plen, ex = _turn_ids(processor, turn_prefix(processor, q, image=img is not None), a, img, first=k == 0)
        extra.update(ex)
        ids += t_ids + [end_id]
        labels += [IGNORE] * plen + t_ids[plen:] + [end_id]
    return {"input_ids": ids[:max_len], "labels": labels[:max_len], **extra}


def collate(batch: Sequence[dict], pad_id: int) -> dict:
    import torch

    T = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), T), IGNORE, dtype=torch.long)
    am = torch.zeros((len(batch), T), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        lab[i, :n] = torch.tensor(b["labels"])
        am[i, :n] = 1
    out = {"input_ids": ids, "attention_mask": am, "labels": lab}
    pix = [b["pixel_values"] for b in batch if "pixel_values" in b]
    if pix:
        out["pixel_values"] = torch.cat(pix)
    return out


class MicroBatches:
    """Dataset of micro-batches (lists of example indices); item k is collated micro-batch k."""

    def __init__(self, examples, image_root, processor, end_id: int, max_len: int, batches: Sequence[Sequence[int]]):
        self.examples, self.image_root, self.processor = examples, Path(image_root), processor
        self.end_id, self.max_len, self.batches = end_id, max_len, batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, k: int) -> dict:
        import torch
        from PIL import Image

        items = []
        for i in self.batches[k]:
            ex = self.examples[i]
            img = Image.open(self.image_root / ex["image"]).convert("RGB") if ex.get("image") else None
            items.append(build_example(self.processor, ex, img, self.end_id, self.max_len))
        out = collate(items, self.processor.tokenizer.pad_token_id)
        out["index"] = torch.tensor(self.batches[k])
        return out


# ----------------------------------------------------------------------------- data


def _ignore_stop_signals(worker_id: int) -> None:
    """DataLoader worker init: SLURM signals the whole process group; only the main process stops."""
    for name in STOP_SIGNALS:
        signal.signal(getattr(signal, name), signal.SIG_IGN)


def _image_decodes(path: Path) -> bool:
    """The image opens and its data decodes (a truncated JPEG from a killed unzip does not)."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.draft("RGB", (64, 64))  # JPEG: decode at 1/8 scale; the whole entropy-coded stream is still read
            im.load()
        return True
    except Exception:
        return False


def find_missing_images(examples: Sequence[dict], image_root, threads: int = 16) -> list[int]:
    """Indices of image examples whose image is missing or does not decode."""
    root = Path(image_root)
    idx = [i for i, ex in enumerate(examples) if ex.get("image")]
    with ThreadPoolExecutor(threads) as pool:
        ok = list(pool.map(lambda i: _image_decodes(root / examples[i]["image"]), idx, chunksize=64))
    return [i for i, good in zip(idx, ok) if not good]


def num_image_tokens(processor) -> int:
    """How many image tokens the processor expands one ``<image>`` into."""
    from PIL import Image

    ids = processor(images=[Image.new("RGB", (64, 64))], text=[processor.image_token], return_tensors="pt")["input_ids"][0]
    return int((ids == processor.tokenizer.convert_tokens_to_ids(processor.image_token)).sum())


def processor_hash(processor) -> str:
    """Hash of what decides the token sequences: chat template, vocabulary, special tokens, image processor."""
    tok = processor.tokenizer
    ip = {k: v for k, v in processor.image_processor.to_dict().items() if not k.startswith("_")}
    return content_hash({"chat_template": processor.chat_template, "vocab": sorted(tok.get_vocab().items()), "special": tok.special_tokens_map,
                         "class": type(tok).__name__, "image_processor": ip, "image_tokens": num_image_tokens(processor)})


def example_stats(processor, examples: Sequence[dict], max_len: int, chunk: int = 8192) -> tuple[list[int], list[int]]:
    """``(lengths, supervised)``: token length and supervised-token count of each example as
    ``build_example`` makes it, from the tokenizer alone (the processor's ``<image>`` expansion is
    replicated, no image is read). ``train`` checks every micro-batch against these."""
    tok, it = processor.tokenizer, processor.image_token
    n_img = num_image_tokens(processor)
    turns = []  # per example: [(first, prefix text, full text)]
    for ex in examples:
        rows = []
        for k, (q, a) in enumerate(conversation_turns(ex)):
            img = bool(ex.get("image")) and k == 0
            pre = turn_prefix(processor, q, image=img)
            full = f"{pre} {a}"
            if img:
                pre, full = pre.replace(it, it * n_img), full.replace(it, it * n_img)
            rows.append((k == 0, pre, full))
        turns.append(rows)
    flat = {True: [], False: []}
    for rows in turns:
        for first, pre, full in rows:
            flat[first] += [pre, full]
    n_tok = {}
    for first, texts in flat.items():
        lens = []
        for s in range(0, len(texts), chunk):
            lens += [len(x) for x in tok(texts[s : s + chunk], add_special_tokens=first)["input_ids"]]
        n_tok[first] = iter(lens)
    lengths, supervised = [], []
    for rows in turns:
        pos = sup = 0
        for first, _, _ in rows:
            lp, lf = next(n_tok[first]), next(n_tok[first])
            sup += max(0, min(pos + lf + 1, max_len) - min(pos + lp, max_len))  # answer tokens + end token, after truncation
            pos += lf + 1
        lengths.append(min(pos, max_len))
        supervised.append(sup)
    return lengths, supervised


def micro_batches(n: int, seed: int, micro_bs: int, global_bs: int | None = None, lengths: Sequence[int] | None = None) -> list[list[int]]:
    """The run's data order: a permutation of the examples (training seed), cut into optimizer steps of
    ``global_bs`` and each step into micro-batches. With ``lengths``, a step's examples are sorted by
    length (longest first, ties by permutation order) before the cut; a step holds the same examples either way."""
    perm = np.random.default_rng(seed).permutation(n)
    gbs = global_bs or micro_bs
    out = []
    for s in range(0, n, gbs):
        step = perm[s : s + gbs]
        if lengths is not None:
            step = step[np.argsort(-np.asarray(lengths)[step], kind="stable")]
        out += [step[k : k + micro_bs].tolist() for k in range(0, len(step), micro_bs)]
    return out


def forward_logits(pm, batch: dict, device: str, amp_dtype, use_amp: bool):
    """Logits of a collated micro-batch (without ``labels`` / ``index``) on ``device``."""
    import torch

    batch = {k: v.to(device) for k, v in batch.items()}
    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(amp_dtype)
    else:  # text-only micro-batch: transformers' Llava would take input_ids without image tokens for legacy (unexpanded) input
        batch["inputs_embeds"] = pm.get_input_embeddings()(batch.pop("input_ids"))
    with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
        return pm(**batch).logits


def micro_loss(logits, labels, normalization: str, step_examples: int, step_tokens: int, accum: int):
    """This micro-batch's share of its optimizer step's loss (the shares of a step sum to the step loss).

    ``step_examples`` / ``step_tokens``: examples / supervised tokens in the whole optimizer step.
    Token cross-entropy in fp32 on ``logits[:, :-1]`` vs ``labels[:, 1:]``; IGNORE positions count 0.
    """
    import torch

    tgt = labels[:, 1:]
    lg = logits[:, :-1].float()
    tok = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), ignore_index=IGNORE, reduction="none").view(tgt.shape)
    n = (tgt != IGNORE).sum(1)
    if normalization == "per_example_mean":
        return (tok.sum(1) / n.clamp(min=1)).sum() / step_examples
    if normalization == "token_mean_global":
        return tok.sum() / max(step_tokens, 1)
    if normalization == "micro_batch_mean":
        return tok.sum() / n.sum().clamp(min=1) / accum
    raise ValueError(f"loss_normalization {normalization!r} not in {LOSS_NORMALIZATIONS}")


# ----------------------------------------------------------------------------- model


def build_model(model_id: str, tcfg: dict, dtype: str, seed: int, device: str):
    """(PEFT model, processor) with fresh LoRA (init from ``seed``) and a trainable projector copy."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=getattr(torch, dtype)).to(device)
    configure_processor(processor, model.config)
    targets = "|".join(tcfg["lora_target_modules"])
    lcfg = LoraConfig(
        r=tcfg["lora_r"], lora_alpha=tcfg["lora_alpha"], lora_dropout=tcfg["lora_dropout"], bias="none",
        target_modules=rf".*language_model.*\.({targets})", modules_to_save=[tcfg["projector_module"]],
    )
    torch.manual_seed(seed)
    pm = get_peft_model(model, lcfg)
    for n, p in pm.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp32 masters
    if tcfg["gradient_checkpointing"]:
        pm.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    pm.config.use_cache = False
    bad = [n for n, p in pm.named_parameters() if p.requires_grad and "vision_tower" in n]
    if bad:
        raise RuntimeError(f"vision tower parameters are trainable: {bad[:3]}")
    return pm, processor


def trainable(pm) -> dict:
    return {n: p for n, p in pm.named_parameters() if p.requires_grad}


def make_optimizer(pm, tcfg: dict, total_steps: int):
    import torch
    from transformers import get_scheduler

    params = trainable(pm)
    proj = [p for n, p in params.items() if "modules_to_save" in n]
    lora = [p for n, p in params.items() if "modules_to_save" not in n]
    if not proj or not lora or not all("lora_" in n for n, p in params.items() if "modules_to_save" not in n):
        raise RuntimeError("unexpected trainable parameters (expected LoRA weights + the projector copy)")
    groups = [{"params": lora, "lr": tcfg["lr"], "name": "lora"}, {"params": proj, "lr": tcfg["projector_lr"], "name": "projector"}]
    opt = torch.optim.AdamW(groups, betas=tuple(tcfg["adam_betas"]), eps=tcfg["adam_eps"], weight_decay=tcfg["weight_decay"])
    warmup = math.ceil(tcfg["warmup_ratio"] * total_steps)
    sched = get_scheduler(tcfg["lr_scheduler"], opt, num_warmup_steps=warmup, num_training_steps=total_steps)
    return opt, sched, warmup


# ----------------------------------------------------------------------------- checkpoints


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_tree(d: Path) -> None:
    for p in d.iterdir():
        with open(p, "rb") as f:
            os.fsync(f.fileno())
    _fsync_dir(d)


def rng_state() -> dict:
    import torch

    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict) -> None:
    import torch

    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():  # a resume may run on another node / GPU count
        torch.cuda.set_rng_state_all(st["cuda"][: torch.cuda.device_count()])


def save_checkpoint(ckpt_root: Path, pm, opt, sched, state: dict, keep: int) -> Path:
    """Atomically write checkpoint ``step-<opt_step>-<micro>`` and prune to the last ``keep``."""
    import torch

    ckpt_root.mkdir(parents=True, exist_ok=True)
    name = f"step-{state['opt_step']:08d}-{state['micro']:04d}"
    final = ckpt_root / name
    tmp = ckpt_root / f".tmp-{name}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    params = trainable(pm)
    torch.save({n: p.detach().cpu() for n, p in params.items()}, tmp / "trainable.pt")
    if state["micro"]:
        torch.save({n: p.grad.detach().cpu() for n, p in params.items() if p.grad is not None}, tmp / "grads.pt")
    torch.save(opt.state_dict(), tmp / "optimizer.pt")
    torch.save(sched.state_dict(), tmp / "scheduler.pt")
    torch.save(rng_state(), tmp / "rng.pt")
    (tmp / "state.json").write_text(json.dumps(state))
    _fsync_tree(tmp)
    (tmp / "COMPLETE").write_text(name)
    _fsync_tree(tmp)
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    _fsync_dir(ckpt_root)
    for old in complete_checkpoints(ckpt_root)[:-keep]:
        shutil.rmtree(old)
    return final


def complete_checkpoints(ckpt_root: Path) -> list[Path]:
    if not ckpt_root.is_dir():
        return []
    return sorted(p for p in ckpt_root.iterdir() if p.name.startswith("step-") and (p / "COMPLETE").is_file())


def load_checkpoint(ckpt: Path, pm, opt, sched) -> dict:
    import torch

    params = trainable(pm)
    saved = torch.load(ckpt / "trainable.pt", map_location="cpu", weights_only=True)
    if set(saved) != set(params):
        raise RuntimeError(f"checkpoint {ckpt}: trainable parameters differ from the model's")
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(saved[n].to(p.device))
    opt.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location="cpu", weights_only=True))  # moved to the params' device
    sched.load_state_dict(torch.load(ckpt / "scheduler.pt", map_location="cpu", weights_only=True))
    state = json.loads((ckpt / "state.json").read_text())
    if state["micro"]:
        grads = torch.load(ckpt / "grads.pt", map_location="cpu", weights_only=True)
        for n, p in params.items():
            p.grad = grads[n].to(p.device) if n in grads else None
    set_rng_state(torch.load(ckpt / "rng.pt", map_location="cpu", weights_only=False))
    return state


# ----------------------------------------------------------------------------- run spec / manifest


def check_run_spec_frozen(cfg: dict, spec: dict, lock_path) -> dict:
    """Manifest lock check for a run spec: the lock holds, and general_order.jsonl and every eval
    split of the spec are locked (eval splits with the spec's content hash). Returns the provenance."""
    from contam.data.freeze import verify

    splits = {n: s["path"] for n, s in spec.get("eval_splits", {}).items()}
    lock = verify(cfg, lock_path, allow_unlocked=True, required=["general_order.jsonl", *splits.values()])
    for n, path in splits.items():
        if lock["files"][path].get("content_hash") != spec["eval_splits"][n]["content_hash"]:
            raise RuntimeError(f"run spec split {n} ({path}) has content hash {spec['eval_splits'][n]['content_hash']}, the lock has {lock['files'][path].get('content_hash')}")
    return {
        "manifest_lock_sha256": file_sha256(lock_path),
        "manifest_split_sha256": {n: lock["files"][p]["sha256"] for n, p in splits.items()},
        "general_order_sha256": lock["files"]["general_order.jsonl"]["sha256"],
    }


def _write_json_atomic(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _versions() -> dict:
    import peft
    import torch
    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}


# ----------------------------------------------------------------------------- run lock
# One trainer per out dir, also across nodes: /scratch is BeeGFS, so no fcntl / flock. ``RUN.lock`` is created with
# O_CREAT | O_EXCL and its mtime is a heartbeat; it is taken over when it belongs to an earlier instance of this SLURM
# job (another SLURM_RESTART_COUNT, heartbeat missed twice) or is stale (heartbeat and train logs older than
# training.run_lock_stale_s). A trainer whose lock was taken over stops (RunBusy) before its next write. On release it is deleted and
# ``RUN.released`` records a clean stop. A dir without a lock whose logs are fresh is busy (a trainer from before the
# lock), unless the last trainer there released its lock or its CLI logged a preemption exit.

RUN_LOCK, RUN_RELEASED = "RUN.lock", "RUN.released"
LOCK_HEARTBEAT_S, LOCK_CONFIRM_S = 60.0, 2.0
PREEMPT_LOG_MARK = f"checkpoint saved, exiting {EXIT_PREEMPTED}"  # last train.log line of a stopped scripts/train_lora.py


def _read_lock_strict(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_lock(path: Path) -> dict:
    try:
        return _read_lock_strict(path)
    except (OSError, ValueError):
        return {}


def _restart(rec: dict) -> str:
    return str(rec.get("slurm_restart_count") or 0)


def _age(path: Path, now: float) -> float:
    try:
        return now - path.stat().st_mtime
    except FileNotFoundError:
        return math.inf


def _owner(rec: dict) -> str:
    return (f"host {rec.get('host')} pid {rec.get('pid')} SLURM job {rec.get('slurm_job_id')} (restart {rec.get('slurm_restart_count')}) "
            f"since {rec.get('started_at')}")


def _stopped_cleanly(out: Path) -> bool:
    """No trainer wrote after the last lock release, or the CLI's last train.log line is a preemption exit."""
    released, jsonl = out / RUN_RELEASED, out / "train_log.jsonl"
    if released.exists() and (not jsonl.exists() or released.stat().st_mtime >= jsonl.stat().st_mtime - 1):
        return True
    try:
        with open(out / "train.log", "rb") as f:
            f.seek(max(0, f.seek(0, os.SEEK_END) - 4096))
            tail = f.read().decode(errors="replace").rstrip("\n").rsplit("\n", 1)[-1]
    except OSError:
        return False
    return PREEMPT_LOG_MARK in tail


class RunLock:
    """A held ``RUN.lock`` with its heartbeat thread; ``release()`` deletes it if it is still ours."""

    def __init__(self, path: Path, rec: dict):
        self.path, self.rec = path, rec
        self.lost, self.thief = threading.Event(), {}  # set when the lock file names another owner
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beat, name="run-lock-heartbeat", daemon=True)
        self._thread.start()

    def owned(self) -> bool:
        return _read_lock(self.path).get("token") == self.rec["token"]

    def _held(self, tries: int = 3) -> dict | None:
        """The lock record, or None if it cannot be read (``tries`` attempts, 1 s apart)."""
        for i in range(tries):
            try:
                return _read_lock_strict(self.path)
            except (OSError, ValueError):
                if i + 1 < tries:
                    time.sleep(1)
        return None

    def _beat(self) -> None:
        while not self._stop.wait(LOCK_HEARTBEAT_S):
            held = self._held()
            if held is not None and held.get("token") != self.rec["token"]:
                self.thief = held
                self.lost.set()
                return
            try:  # unreadable (transient) counts as still ours
                os.utime(self.path)
            except OSError:
                pass

    def release(self) -> None:
        self._stop.set()
        self._thread.join()
        held = self._held()
        if held is not None and held.get("token") == self.rec["token"]:
            _write_json_atomic({**self.rec, "released_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, self.path.with_name(RUN_RELEASED))
            self.path.unlink(missing_ok=True)


def acquire_run_lock(out: Path, stale_s: float) -> RunLock:
    """Take ``out/RUN.lock`` or raise RunBusy (see the comment above)."""
    path, now = out / RUN_LOCK, time.time()
    rec = {"host": socket.gethostname(), "pid": os.getpid(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT"), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "token": uuid.uuid4().hex}
    log_age = min(_age(out / "train.log", now), _age(out / "train_log.jsonl", now))
    if not path.exists() and log_age <= stale_s and not _stopped_cleanly(out):
        raise RunBusy(f"{out}: no {RUN_LOCK}, but its train log was written {log_age:.0f} s ago (a trainer without the run lock?)")
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
    except FileExistsError:
        try:
            st = path.stat()
        except FileNotFoundError:
            raise RunBusy(f"{out}: {RUN_LOCK} was released while starting; rerun") from None
        held, lock_age = _read_lock(path), now - st.st_mtime
        requeued = (rec["slurm_job_id"] is not None and held.get("slurm_job_id") == rec["slurm_job_id"] and _restart(held) != _restart(rec)
                    and lock_age > 2 * LOCK_HEARTBEAT_S)
        if not requeued and (lock_age <= stale_s or log_age <= stale_s):
            raise RunBusy(f"{out} is being trained by {_owner(held)} (heartbeat {lock_age:.0f} s ago, train log {log_age:.0f} s ago)")
        tmp = path.with_name(f"{RUN_LOCK}.{rec['token']}.tmp")
        tmp.write_text(json.dumps(rec))
        try:
            st2 = path.stat()
        except FileNotFoundError:
            st2 = None
        if st2 is None or (st2.st_ino, st2.st_mtime_ns) != (st.st_ino, st.st_mtime_ns):  # beaten, or taken over meanwhile
            tmp.unlink(missing_ok=True)
            raise RunBusy(f"{out}: {RUN_LOCK} changed while taking it over (now {_owner(_read_lock(path))})")
        os.replace(tmp, path)
        time.sleep(LOCK_CONFIRM_S)  # a concurrent takeover by another process: the last replace wins
        if _read_lock(path).get("token") != rec["token"]:
            raise RunBusy(f"{out}: lock taken over concurrently by {_owner(_read_lock(path))}")
        return RunLock(path, rec)
    _write_json_atomic(rec, path)
    return RunLock(path, rec)


# ----------------------------------------------------------------------------- training


def _cached(path: Path, key: str, compute, keep=lambda value: True):
    """``compute()`` (JSON-serializable), cached in ``path`` under ``key`` when ``keep(value)``."""
    if path.exists():
        c = read_json(path)
        if c.get("key") == key:
            return c["value"]
    value = compute()
    if keep(value):
        _write_json_atomic({"key": key, "value": value}, path)
    return value


def _append_line(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_log(path: Path) -> list[dict]:
    """train_log.jsonl records; a partial last line (killed mid-write) is dropped."""
    recs = []
    for line in open(path):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return recs


def train(
    cfg: dict,
    data_path,
    out_dir,
    seed_slot: int,
    run_id: str,
    run_spec: dict | None = None,
    run_spec_path=None,
    frozen: dict | None = None,
    image_root=None,
    max_steps: int | None = None,
    max_examples: int | None = None,
    allow_missing_images: bool = False,
    device: str | None = None,
    stop_after_micro: int | None = None,
    log=print,
) -> dict:
    """Train (or resume, or no-op if DONE) one run. Returns ``{"status", ...}``; raises Preempted on a stop signal.

    ``frozen`` is the manifest provenance from ``check_run_spec_frozen`` (None: the check was skipped).
    """
    stop = {"sig": None}
    prev = {}
    if threading.current_thread() is threading.main_thread():
        for name in STOP_SIGNALS:
            sig = getattr(signal, name)
            prev[sig] = signal.signal(sig, lambda s, f: stop.__setitem__("sig", signal.Signals(s).name))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lock = None
    try:
        if not (out / "DONE").exists():  # a DONE run is a no-op (checked in _train) without the lock
            lock = acquire_run_lock(out, float(cfg["training"].get("run_lock_stale_s", 600)))
        return _train(cfg, data_path, out_dir, seed_slot, run_id, run_spec, run_spec_path, frozen, image_root, max_steps, max_examples,
                      allow_missing_images, device, stop_after_micro, log, stop, lock)
    finally:
        if lock:
            lock.release()
        for sig, h in prev.items():
            signal.signal(sig, h)


def _train(cfg, data_path, out_dir, seed_slot, run_id, run_spec, run_spec_path, frozen, image_root, max_steps, max_examples,
           allow_missing_images, device, stop_after_micro, log, stop, lock=None) -> dict:
    def check_stop(where: str) -> None:
        if stop["sig"]:
            log(f"{run_id}: {stop['sig']} {where}; stopping without a new checkpoint")
            raise Preempted(stop["sig"])

    def check_lock(where: str) -> None:
        if lock is not None and lock.lost.is_set():
            raise RunBusy(f"{out_dir}: run lock taken over by {_owner(lock.thief)}; stopped {where} without writing")

    import torch
    from transformers import AutoConfig, AutoProcessor

    tcfg = cfg["training"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_root = out / "checkpoints"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_id = (run_spec or {}).get("model") or cfg["models"]["main"]
    dtype = cfg["models"]["dtype"]
    seed = int(tcfg["seeds"][seed_slot])
    if run_spec and run_spec.get("train_seed") is not None and int(run_spec["train_seed"]) != seed:
        raise RuntimeError(f"run spec train_seed {run_spec['train_seed']} != training.seeds[{seed_slot}] = {seed}")
    micro_bs, gbs = int(tcfg["micro_batch_size"]), int(tcfg["global_batch_size"])
    if gbs % micro_bs:
        raise ValueError(f"micro_batch_size {micro_bs} must divide global_batch_size {gbs}")
    accum = gbs // micro_bs
    if int(tcfg["epochs"]) != 1:
        raise NotImplementedError(f"training.epochs = {tcfg['epochs']}; only 1 epoch is implemented")
    norm, order = tcfg["loss_normalization"], tcfg["micro_batch_order"]
    if norm not in LOSS_NORMALIZATIONS or order not in MICRO_BATCH_ORDERS:
        raise ValueError(f"loss_normalization {norm!r} must be in {LOSS_NORMALIZATIONS}, micro_batch_order {order!r} in {MICRO_BATCH_ORDERS}")
    max_len = int(tcfg["max_len"])
    image_root = Path(image_root or cfg["paths"]["llava_images"])

    data_sha = file_sha256(data_path)
    if run_spec and run_spec.get("data", {}).get("sha256") not in (None, data_sha):
        raise RuntimeError(f"{data_path} sha256 {data_sha} != run spec data sha256 {run_spec['data']['sha256']}")
    examples = read_json(data_path)
    n_file = len(examples)
    if run_spec and run_spec.get("data", {}).get("n_examples") not in (None, n_file):
        raise RuntimeError(f"{data_path} has {n_file} examples, the run spec says {run_spec['data']['n_examples']}")
    if max_examples is not None:
        examples = examples[:max_examples]
    check_stop("before the image check")
    missing = _cached(out / "image_check.json", content_hash({"data_sha256": data_sha, "max_examples": max_examples, "image_root": str(image_root)}),
                      lambda: find_missing_images(examples, image_root), keep=lambda m: not m)  # a non-clean result is rechecked next start
    if missing and not allow_missing_images:
        raise RuntimeError(f"{len(missing)} of {len(examples)} examples have a missing or unreadable image under {image_root} "
                           f"(e.g. {examples[missing[0]]['image']}); pass --allow-missing-images (timing only) to drop them")
    if missing:
        drop = set(missing)
        missing_by_source = dict(Counter(examples[i]["image"].split("/", 1)[0] for i in missing))
        examples = [ex for i, ex in enumerate(examples) if i not in drop]
        log(f"WARNING: dropped {len(missing)} examples with missing or unreadable images {missing_by_source} (--allow-missing-images)")
    else:
        missing_by_source = {}

    processor, model_config = AutoProcessor.from_pretrained(model_id), AutoConfig.from_pretrained(model_id)
    configure_processor(processor, model_config)
    env = {**_versions(), "model_commit": getattr(model_config, "_commit_hash", None), "processor_sha": processor_hash(processor)}
    provenance = {"manifest_checked": frozen is not None, "manifest_split_sha256": (frozen or {}).get("manifest_split_sha256"),
                  "general_order_sha256": (frozen or {}).get("general_order_sha256")}
    semantic = {k: v for k, v in tcfg.items() if k not in NON_SEMANTIC}
    hashed = {"format_version": FORMAT_VERSION, "model": model_id, "dtype": dtype, "end_token": cfg["models"]["end_token"], "training": semantic, "seed": seed,
              "data_sha256": data_sha, "max_examples": max_examples, "n_examples": len(examples), "n_missing_images_dropped": len(missing),
              "dropped_sha": content_hash(sorted(missing)), "provenance": provenance, "env": env}
    config_hash = content_hash(hashed)

    done = out / "DONE"
    if done.exists():
        man = read_json(out / "manifest.json")
        if man["config_hash"] != config_hash:
            raise RuntimeError(f"{out} is DONE with config hash {man['config_hash']}, this run has {config_hash}; refusing (use a new out dir)")
        log(f"{run_id}: already DONE ({out}); nothing to do")
        return {**man, "status": "done-noop"}
    ckpts = complete_checkpoints(ckpt_root)
    if ckpts:  # refuse before the slow part
        st = json.loads((ckpts[-1] / "state.json").read_text())
        if st["config_hash"] != config_hash:
            raise RuntimeError(f"checkpoint {ckpts[-1]} has config hash {st['config_hash']}, this run has {config_hash}; refusing to resume")
    check_stop("before the model load")

    end_id = end_token_id(processor.tokenizer, cfg["models"]["end_token"])
    lengths, supervised = _cached(out / "example_stats.json", content_hash({"config_hash": config_hash}),
                                  lambda: list(example_stats(processor, examples, max_len)))
    batches = micro_batches(len(examples), seed, micro_bs, gbs, lengths if order == "length_sorted" else None)
    total_steps = math.ceil(len(batches) / accum)
    # examples / supervised tokens of the optimizer step each micro-batch belongs to
    step_ex = [sum(len(batches[k]) for k in range(s, min(s + accum, len(batches)))) for s in range(0, len(batches), accum)]
    step_sup = [sum(supervised[i] for k in range(s, min(s + accum, len(batches))) for i in batches[k]) for s in range(0, len(batches), accum)]
    pm, _ = build_model(model_id, tcfg, dtype, seed, device)
    opt, sched, warmup = make_optimizer(pm, tcfg, total_steps)

    for stale in ckpt_root.glob(".tmp-*") if ckpt_root.is_dir() else []:
        shutil.rmtree(stale, ignore_errors=True)  # a save killed before its rename
    state = {"opt_step": 0, "micro": 0, "pos": 0, "acc_loss": 0.0, "step_losses": [], "tokens": 0, "sup_tokens": 0, "examples": 0,
             "wall_s": 0.0, "segments": 0, "config_hash": config_hash}
    log_path = out / "train_log.jsonl"
    saved = {"at": None}
    if ckpts:
        state = load_checkpoint(ckpts[-1], pm, opt, sched)
        saved["at"] = state["pos"]
        if log_path.exists():  # drop steps logged after this checkpoint (they are redone) and a partial last line
            log_path.write_text("".join(json.dumps(r) + "\n" for r in _read_log(log_path) if r["step"] <= state["opt_step"]))
        log(f"{run_id}: resumed from {ckpts[-1].name} (optimizer step {state['opt_step']}, micro {state['micro']}, data position {state['pos']})")
    state["segments"] += 1
    check_stop("after the model load")

    use_amp = dtype != "float32"  # frozen base in models.dtype, fp32 trainable masters
    amp_dtype = getattr(torch, dtype)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    ds = MicroBatches(examples, image_root, processor, end_id, max_len, batches)
    sub = torch.utils.data.Subset(ds, range(state["pos"], len(ds)))
    nw = int(tcfg["num_workers"])
    # own generator: creating the iterator must not draw from the global torch RNG (dropout), or a
    # resumed run would diverge from an uninterrupted one
    loader = torch.utils.data.DataLoader(sub, batch_size=None, shuffle=False, num_workers=nw, prefetch_factor=4 if nw else None,
                                         worker_init_fn=_ignore_stop_signals, generator=torch.Generator())
    params = list(trainable(pm).values())
    log(f"{run_id}: {len(examples)} examples, {len(batches)} micro-batches of {micro_bs} ({order}), accumulation {accum}, {total_steps} optimizer steps "
        f"(warmup {warmup}), loss {norm}, seed {seed}, device {device}, config {config_hash[:12]}")

    def checkpoint(reason: str) -> Path | None:
        check_lock(f"before a checkpoint ({reason})")
        if saved["at"] == state["pos"]:  # e.g. a signal that arrived during a periodic save
            log(f"{run_id}: checkpoint at data position {state['pos']} already saved ({reason})")
            return None
        saved["at"] = state["pos"]
        t0 = time.time()
        p = save_checkpoint(ckpt_root, pm, opt, sched, state, int(tcfg["keep_checkpoints"]))
        log(f"{run_id}: checkpoint {p.name} ({reason}) in {time.time() - t0:.1f} s")
        return p

    seg_t0 = time.time()
    step_t0, n_step_ex, n_step_tok = time.time(), 0, 0
    micro_done = 0
    pm.train()
    for batch in loader:
        if max_steps is not None and state["opt_step"] >= max_steps:
            break
        k = state["pos"]
        idx = batch.pop("index").tolist()
        if idx != list(batches[k]):
            raise RuntimeError(f"micro-batch {k}: loader returned examples {idx[:4]}..., expected {batches[k][:4]}...")
        got_len, got_sup = batch["attention_mask"].sum(1).tolist(), (batch["labels"] != IGNORE).sum(1).tolist()
        if got_len != [lengths[i] for i in idx] or got_sup != [supervised[i] for i in idx]:
            bad = next(j for j, i in enumerate(idx) if (got_len[j], got_sup[j]) != (lengths[i], supervised[i]))
            raise RuntimeError(f"example {examples[idx[bad]].get('id')}: {got_len[bad]} tokens / {got_sup[bad]} supervised, "
                               f"example_stats said {lengths[idx[bad]]} / {supervised[idx[bad]]}")
        sup, tok_in = sum(got_sup), sum(got_len)
        labels = batch.pop("labels").to(device)
        logits = forward_logits(pm, batch, device, amp_dtype, use_amp)
        s = k // accum
        loss = micro_loss(logits, labels, norm, step_ex[s], step_sup[s], accum)
        loss.backward()
        del logits
        n_ex = len(idx)
        state["acc_loss"] += float(loss)
        state["micro"] += 1
        state["pos"] += 1
        state["tokens"] += tok_in
        state["sup_tokens"] += sup
        state["examples"] += n_ex
        n_step_ex, n_step_tok = n_step_ex + n_ex, n_step_tok + tok_in
        micro_done += 1
        if state["micro"] == accum or state["pos"] == len(batches):
            torch.nn.utils.clip_grad_norm_(params, tcfg["max_grad_norm"])
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            state["opt_step"] += 1
            state["step_losses"].append(state["acc_loss"])
            state["micro"], state["acc_loss"] = 0, 0.0
            check_lock(f"at optimizer step {state['opt_step']}")
            dt = time.time() - step_t0
            rec = {"step": state["opt_step"], "loss": state["step_losses"][-1], "lr": [g["lr"] for g in opt.param_groups], "step_time_s": dt,
                   "examples": n_step_ex, "examples_per_s": n_step_ex / dt, "tokens": n_step_tok, "tokens_per_s": n_step_tok / dt, "segment": state["segments"],
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else None}
            _append_line(log_path, rec)
            log(f"step {rec['step']}/{total_steps} loss {rec['loss']:.4f} {rec['examples_per_s']:.2f} ex/s {rec['tokens_per_s']:.0f} tok/s "
                f"{dt:.1f} s/step" + (f" peak {rec['peak_mem_gb']:.1f} GiB" if rec["peak_mem_gb"] is not None else ""))
            if state["opt_step"] % int(tcfg["checkpoint_every_steps"]) == 0 and state["pos"] < len(batches):
                state["wall_s"] += time.time() - seg_t0
                seg_t0 = time.time()
                checkpoint("periodic")
            step_t0, n_step_ex, n_step_tok = time.time(), 0, 0  # step times exclude checkpoint saves
        if stop["sig"] or (stop_after_micro is not None and micro_done >= stop_after_micro):
            state["wall_s"] += time.time() - seg_t0
            checkpoint(f"stop: {stop['sig'] or 'stop_after_micro'}")
            raise Preempted(stop["sig"] or "stop_after_micro")
    state["wall_s"] += time.time() - seg_t0

    if state["pos"] < len(batches):  # --max-steps
        checkpoint("max_steps")
        return {"status": "max_steps", "opt_step": state["opt_step"], "config_hash": config_hash, "log": str(log_path)}
    checkpoint("final")  # a death during the export resumes here, with nothing left to train
    check_stop("before the adapter export")
    check_lock("before the adapter export")

    adapter = out / "adapter"
    tmp = out / ".tmp-adapter"
    shutil.rmtree(tmp, ignore_errors=True)
    for p in params:  # the eval model runs in models.dtype; the fp32 masters stay in the checkpoints
        p.data = p.data.to(amp_dtype)
    pm.save_pretrained(tmp)
    _fsync_tree(tmp)
    shutil.rmtree(adapter, ignore_errors=True)
    os.replace(tmp, adapter)
    _fsync_dir(out)
    pm = opt = sched = params = loader = None  # free the training model before loading a fresh base
    import gc

    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    check_stop("before the adapter check")
    from transformers import AutoModelForVision2Seq

    base = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=getattr(torch, dtype)).to(device)
    load_lora(base, str(adapter))  # raises unless the adapter loads and changes the model
    del base
    man = {
        "run_id": run_id, "status": "done", "config_hash": config_hash, **hashed, "seed_slot": seed_slot,
        "data_path": str(data_path), "image_root": str(image_root), "missing_images_by_source": missing_by_source,
        "run_spec": str(run_spec_path) if run_spec_path else None, "run_spec_sha256": file_sha256(run_spec_path) if run_spec_path else None,
        **(frozen or {"manifest_lock_sha256": None, "manifest_split_sha256": None, "general_order_sha256": None}),
        "micro_batch_size": micro_bs, "gradient_accumulation": accum, "optimizer_steps": state["opt_step"], "warmup_steps": warmup,
        "examples_seen": state["examples"], "tokens": state["tokens"], "supervised_tokens": state["sup_tokens"], "final_loss": state["step_losses"][-1],
        "wall_time_s": state["wall_s"], "segments": state["segments"], "adapter": str(adapter), "adapter_sha256": file_sha256(adapter / "adapter_model.safetensors"),
        "versions": _versions(), "repo": str(REPO_ROOT),
    }
    check_lock("before the manifest and DONE")
    _write_json_atomic(man, out / "manifest.json")
    with open(done, "w") as f:
        f.write(config_hash + "\n")
        f.flush()
        os.fsync(f.fileno())
    _fsync_dir(out)
    if not tcfg.get("keep_checkpoints_after_done", False):  # DONE + manifest are durable; a crash mid-delete leaves a DONE no-op
        check_lock("before deleting the checkpoints")
        shutil.rmtree(ckpt_root, ignore_errors=True)
    log(f"{run_id}: DONE: {state['opt_step']} steps, {state['examples']} examples, {state['tokens']} tokens, {state['wall_s'] / 3600:.2f} h -> {adapter}")
    return {"status": "done", **man}
