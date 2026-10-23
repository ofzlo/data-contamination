"""LoRA trainer (contam.train, scripts/train_lora.py) on a tiny LLaVA with the real llava-hf processor (CPU).

Needs the llava-hf/llava-1.5-7b-hf processor files in the local HF cache (no weights); skipped otherwise.
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from contam.config import DEFAULT_CONFIG, REPO_ROOT, load_config, write_json

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

P_CAP = "Describe the image in one short sentence."


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    """A tiny random LLaVA (real tokenizer / image processor, 336px images -> 576 image tokens), saved to disk."""
    from transformers import AutoProcessor, CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        proc = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    except Exception as e:  # pragma: no cover - depends on the local cache
        pytest.skip(f"llava-hf processor not available offline: {e}")
    cfg = LlavaConfig(
        vision_config=CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, image_size=336, patch_size=14, projection_dim=16),
        text_config=LlamaConfig(vocab_size=32064, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, pad_token_id=32001),
        image_token_index=32000, vision_feature_layer=-2, vision_feature_select_strategy="default",
    )
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("tiny_llava")
    LlavaForConditionalGeneration(cfg).save_pretrained(d)
    proc.save_pretrained(d)
    return d


def _images(root: Path, n: int):
    from PIL import Image

    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        p = root / "coco" / "train2017" / f"{i:012d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (48 + i, 64, 3), dtype=np.uint8)).save(p)
        paths.append(f"coco/train2017/{i:012d}.jpg")
    return paths


def make_data(tmp_path: Path, n: int = 12) -> tuple[Path, Path]:
    root = tmp_path / "llava"
    imgs = _images(root, n)
    ex = []
    for i in range(n):
        if i % 4 == 3:
            ex.append({"id": f"t{i}", "conversations": [{"from": "human", "value": f"Say {i}."}, {"from": "gpt", "value": f"Number {i}."}]})
        elif i % 4 == 2:
            ex.append({"id": f"m{i}", "image": imgs[i], "conversations": [
                {"from": "human", "value": "What is this?\n<image>"}, {"from": "gpt", "value": "Noise."},
                {"from": "human", "value": "Colour?"}, {"from": "gpt", "value": f"Mixed {i}."}]})
        else:
            ex.append({"id": f"c{i}", "image": imgs[i], "conversations": [{"from": "human", "value": f"<image>\n{P_CAP}"}, {"from": "gpt", "value": f"A random image {i}."}]})
    write_json(ex, tmp_path / "data.json")
    return tmp_path / "data.json", root


def tiny_cfg(model_dir, **training):
    cfg = load_config()
    cfg["models"].update(main=str(model_dir), dtype="float32")
    cfg["training"].update(lora_r=4, lora_alpha=8, lr=1e-2, projector_lr=1e-3, micro_batch_size=2, global_batch_size=4, checkpoint_every_steps=2,
                           num_workers=0, gradient_checkpointing=False, warmup_ratio=0.2)
    cfg["training"].update(training)
    return cfg


def _run(cfg, data, root, out, **kw):
    from contam.train import train

    return train(cfg, data, out, 0, "tiny", image_root=root, device="cpu", log=lambda m: None, **kw)


def _losses(out):
    return [json.loads(line)["loss"] for line in open(out / "train_log.jsonl")]


def _adapter(out):
    from safetensors.torch import load_file

    return load_file(out / "adapter" / "adapter_model.safetensors")


def _same_adapter(a, b):
    A, B = _adapter(a), _adapter(b)
    assert set(A) == set(B) and any("multi_modal_projector" in k for k in A) and any("lora_A" in k for k in A)
    for k in A:
        assert torch.equal(A[k], B[k]), k


# ----------------------------------------------------------------------------- format


def test_conversation_turns_and_image_marker():
    from contam.train import conversation_turns

    ex = {"conversations": [{"from": "human", "value": "Where?\n<image>"}, {"from": "gpt", "value": " Here. "}, {"from": "human", "value": "Why?"}, {"from": "gpt", "value": "So."}]}
    assert conversation_turns(ex) == [("Where?", "Here."), ("Why?", "So.")]
    with pytest.raises(ValueError, match="human/gpt"):
        conversation_turns({"conversations": [{"from": "gpt", "value": "x"}, {"from": "human", "value": "y"}]})


def test_training_example_matches_score_sequence_and_mask(tiny_model_dir):
    """A (p_cap, ref) example is exactly the sequence HFVLM.score scores, with the loss on exactly the scored positions."""
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    from contam.models import HFVLM, configure_processor, end_token_id
    from contam.train import IGNORE, build_example, collate

    proc = AutoProcessor.from_pretrained(tiny_model_dir)
    model = AutoModelForVision2Seq.from_pretrained(tiny_model_dir).eval()
    configure_processor(proc, model.config)
    vlm = object.__new__(HFVLM)
    vlm.torch, vlm.device, vlm.model_id, vlm.adapter, vlm.processor, vlm.model = torch, "cpu", str(tiny_model_dir), None, proc, model
    vlm.end_token_id = end_token_id(proc.tokenizer)
    seen = {}
    model.register_forward_pre_hook(lambda m, a, kw: seen.update(kw), with_kwargs=True)
    img = Image.fromarray(np.random.default_rng(1).integers(0, 255, (50, 70, 3), dtype=np.uint8))
    ref = "  A small dog sits on a red couch. "
    lp, ntok = vlm.score([img], P_CAP, [ref])

    ex = build_example(proc, {"conversations": [{"from": "human", "value": f"<image>\n{P_CAP}"}, {"from": "gpt", "value": ref}]}, img, vlm.end_token_id, 2048)
    assert ex["input_ids"] == seen["input_ids"][0].tolist() and ex["input_ids"][-1] == proc.tokenizer.eos_token_id
    assert torch.equal(ex["pixel_values"], seen["pixel_values"])
    target = [i for i, x in enumerate(ex["labels"]) if x != IGNORE]
    assert len(target) == ntok[0] and target == list(range(len(ex["input_ids"]) - ntok[0], len(ex["input_ids"])))
    assert [ex["labels"][i] for i in target] == ex["input_ids"][-ntok[0]:]
    assert ex["input_ids"].count(32000) == 576  # <image> expanded by the processor
    # the training loss on this example is the mean of exactly the log-probs score() sums
    with torch.no_grad():
        loss = model(**collate([ex], proc.tokenizer.pad_token_id)).loss
    assert float(loss) == pytest.approx(-lp[0] / ntok[0], rel=1e-4)
    from contam.train import micro_loss

    b = collate([ex], proc.tokenizer.pad_token_id)
    with torch.no_grad():
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], pixel_values=b["pixel_values"]).logits
    for norm in ("per_example_mean", "token_mean_global", "micro_batch_mean"):
        assert float(micro_loss(logits, b["labels"], norm, 1, ntok[0], 1)) == pytest.approx(-lp[0] / ntok[0], rel=1e-4)


def test_multi_turn_masks_only_answers(tiny_model_dir):
    from transformers import AutoProcessor

    from contam.train import IGNORE, build_example

    proc = AutoProcessor.from_pretrained(tiny_model_dir)
    tok = proc.tokenizer
    ex = build_example(proc, {"conversations": [{"from": "human", "value": "Hi?"}, {"from": "gpt", "value": "Hello."}, {"from": "human", "value": "Bye?"}, {"from": "gpt", "value": "Bye."}]},
                       None, tok.eos_token_id, 2048)
    ids, lab = ex["input_ids"], ex["labels"]
    assert tok.decode(ids) == "<s> USER: Hi? ASSISTANT: Hello.</s> USER: Bye? ASSISTANT: Bye.</s>" or tok.decode(ids).replace("<s> ", "<s>") == "<s>USER: Hi? ASSISTANT: Hello.</s>USER: Bye? ASSISTANT: Bye.</s>"
    sup = tok.decode([x for x in lab if x != IGNORE])
    assert sup.replace(" ", "") == "Hello.</s>Bye.</s>"
    assert ids[0] == tok.bos_token_id and ids.count(tok.bos_token_id) == 1 and ids.count(tok.eos_token_id) == 2
    assert "pixel_values" not in ex
    short = build_example(proc, {"conversations": [{"from": "human", "value": "Hi?"}, {"from": "gpt", "value": "Hello there."}]}, None, tok.eos_token_id, 5)
    assert len(short["input_ids"]) == len(short["labels"]) == 5


# ----------------------------------------------------------------------------- training / resume


def test_train_done_adapter_loads_and_rerun_is_noop(tiny_model_dir, tmp_path):
    from transformers import AutoModelForVision2Seq

    from contam.models import load_lora

    data, root = make_data(tmp_path)
    cfg = tiny_cfg(tiny_model_dir)
    out = tmp_path / "run"
    res = _run(cfg, data, root, out)
    assert res["status"] == "done" and res["optimizer_steps"] == 3 and res["examples_seen"] == 12 and res["n_missing_images_dropped"] == 0
    assert (out / "DONE").exists() and not (out / "checkpoints").exists()  # deleted after DONE (keep_checkpoints_after_done: false)
    for f in ("manifest.json", "train_log.jsonl", "example_stats.json", "image_check.json", "adapter/adapter_model.safetensors"):
        assert (out / f).exists(), f
    man = json.loads((out / "manifest.json").read_text())
    assert man["config_hash"] == res["config_hash"] and man["manifest_lock_sha256"] is None and man["tokens"] > man["supervised_tokens"] > 0
    merged = load_lora(AutoModelForVision2Seq.from_pretrained(tiny_model_dir), str(out / "adapter"))
    assert merged is not None
    mtime = (out / "adapter" / "adapter_model.safetensors").stat().st_mtime_ns
    assert _run(cfg, data, root, out)["status"] == "done-noop"
    assert (out / "adapter" / "adapter_model.safetensors").stat().st_mtime_ns == mtime
    with pytest.raises(RuntimeError, match="is DONE with config hash"):
        _run(tiny_cfg(tiny_model_dir, lr=5e-3), data, root, out)
    log = [json.loads(line) for line in open(out / "train_log.jsonl")]
    assert all(r["examples_per_s"] > 0 and r["tokens_per_s"] > 0 and r["step_time_s"] > 0 for r in log)


@pytest.mark.parametrize("stop_after", [2, 3])  # 2: at an optimizer-step boundary; 3: mid-accumulation
def test_resume_is_identical_to_an_uninterrupted_run(tiny_model_dir, tmp_path, stop_after):
    from contam.train import Preempted

    data, root = make_data(tmp_path)
    cfg = tiny_cfg(tiny_model_dir, lora_dropout=0.3)  # dropout: the RNG state must be restored too
    _run(cfg, data, root, tmp_path / "straight")
    with pytest.raises(Preempted):
        _run(cfg, data, root, tmp_path / "resumed", stop_after_micro=stop_after)
    state = json.loads(sorted((tmp_path / "resumed" / "checkpoints").glob("step-*"))[-1].joinpath("state.json").read_text())
    assert state["pos"] == stop_after and state["micro"] == stop_after % 2
    res = _run(cfg, data, root, tmp_path / "resumed")
    assert res["status"] == "done" and res["segments"] == 2 and res["examples_seen"] == 12
    assert _losses(tmp_path / "straight") == _losses(tmp_path / "resumed")
    _same_adapter(tmp_path / "straight", tmp_path / "resumed")


def test_kill_during_checkpoint_save_keeps_the_previous_one(tiny_model_dir, tmp_path, monkeypatch):
    import contam.train as tr

    data, root = make_data(tmp_path)
    cfg = tiny_cfg(tiny_model_dir, checkpoint_every_steps=1)
    _run(cfg, data, root, tmp_path / "straight")
    real_save, calls = torch.save, {"n": 0}

    def dying_save(obj, f, *a, **k):  # the 2nd checkpoint dies after writing its weights
        real_save(obj, f, *a, **k)
        if str(f).endswith("trainable.pt"):
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt("killed mid-save")

    monkeypatch.setattr(torch, "save", dying_save)
    with pytest.raises(KeyboardInterrupt):
        _run(cfg, data, root, tmp_path / "run")
    monkeypatch.undo()
    ck = tmp_path / "run" / "checkpoints"
    assert [p.name for p in tr.complete_checkpoints(ck)] == ["step-00000001-0000"] and list(ck.glob(".tmp-*"))
    res = _run(cfg, data, root, tmp_path / "run")
    assert res["status"] == "done" and not list(ck.glob(".tmp-*"))
    _same_adapter(tmp_path / "straight", tmp_path / "run")
    assert _losses(tmp_path / "straight") == _losses(tmp_path / "run")  # steps after the resumed checkpoint are logged once


def test_resume_refuses_a_different_config(tiny_model_dir, tmp_path):
    from contam.train import Preempted

    data, root = make_data(tmp_path)
    with pytest.raises(Preempted):
        _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "run", stop_after_micro=2)
    with pytest.raises(RuntimeError, match="refusing to resume"):
        _run(tiny_cfg(tiny_model_dir, lr=5e-3), data, root, tmp_path / "run")
    with pytest.raises(RuntimeError, match="refusing to resume"):
        _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "run", max_examples=8)


def test_missing_images_refused_unless_allowed(tiny_model_dir, tmp_path):
    data, root = make_data(tmp_path)
    (root / "coco" / "train2017" / f"{0:012d}.jpg").unlink()
    with pytest.raises(RuntimeError, match="1 of 12 examples have a missing or unreadable image"):
        _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "run")
    res = _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "run", allow_missing_images=True)
    assert res["n_missing_images_dropped"] == 1 and res["examples_seen"] == 11 and res["missing_images_by_source"] == {"coco": 1}


def test_max_steps_stops_with_a_checkpoint(tiny_model_dir, tmp_path):
    data, root = make_data(tmp_path)
    res = _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "run", max_steps=1)
    assert res["status"] == "max_steps" and res["opt_step"] == 1 and not (tmp_path / "run" / "DONE").exists()
    res = _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "run")
    assert res["status"] == "done" and res["optimizer_steps"] == 3


def _cli_config(tmp_path, model_dir, **training) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["models"].update(main=str(model_dir), dtype="float32")
    raw["training"].update(lora_r=4, lora_alpha=8, lr=1e-2, projector_lr=1e-3, micro_batch_size=2, global_batch_size=4, checkpoint_every_steps=1000,
                           num_workers=0, gradient_checkpointing=True, warmup_ratio=0.2)
    raw["training"].update(training)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_sigterm_saves_a_checkpoint_and_resume_matches(tiny_model_dir, tmp_path):
    """Real process with DataLoader workers, SIGTERM to its whole process group mid-run (as SLURM does): exit 75
    with a checkpoint; the rerun equals an uninterrupted run."""
    data, root = make_data(tmp_path, n=48)
    cfg_path = _cli_config(tmp_path, tiny_model_dir, num_workers=2)
    out = tmp_path / "cli"
    cmd = [sys.executable, str(REPO_ROOT / "scripts/train_lora.py"), "--config", str(cfg_path), "--data", str(data), "--seed-slot", "0", "--run-id", "tiny",
           "--out-dir", str(out), "--image-root", str(root), "--skip-manifest-check"]
    env = dict(os.environ, HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="")  # CPU, like the in-process runs
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, start_new_session=True)
    try:
        deadline = time.time() + 600
        while not (out / "train_log.jsonl").exists() and p.poll() is None and time.time() < deadline:
            time.sleep(0.2)
        assert p.poll() is None, p.communicate()[0]
        os.killpg(p.pid, signal.SIGTERM)  # the workers get it too
        log, _ = p.communicate(timeout=300)
    finally:
        p.kill()
    assert p.returncode == 75, log
    assert "checkpoint step-" in log and "stop: SIGTERM" in log
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and "resumed from step-" in r.stdout, r.stdout + r.stderr
    from contam.train import train

    cfg = load_config(cfg_path)
    train(cfg, data, tmp_path / "straight", 0, "tiny", image_root=root, device="cpu", log=lambda m: None)
    _same_adapter(tmp_path / "straight", out)
    assert _losses(tmp_path / "straight") == _losses(out)


def test_run_spec_lock_check(tmp_path):
    """The run spec's eval splits must be locked with its content hash, and general_order.jsonl must be locked."""
    import shutil

    from contam.config import file_sha256, read_json
    from contam.data.freeze import DEFAULT_LOCK
    from contam.train import check_run_spec_frozen

    cfg = load_config()
    real = read_json(DEFAULT_LOCK)
    splits = tmp_path / "splits"
    splits.mkdir()
    for name in real["files"]:
        shutil.copy(cfg["paths"]["splits_dir"] / name, splits / name)
    cfg["paths"]["splits_dir"] = splits
    spec = {"eval_splits": {"A": {"path": "split_A.json", "content_hash": real["files"]["split_A.json"]["content_hash"]}}}
    write_json(real, tmp_path / "lock_without_order.json")
    with pytest.raises(RuntimeError, match="general_order.jsonl is not frozen"):
        check_run_spec_frozen(cfg, spec, tmp_path / "lock_without_order.json")
    (splits / "general_order.jsonl").write_text('{"order": 0}\n')
    lock = {**real, "files": {**real["files"], "general_order.jsonl": {"sha256": file_sha256(splits / "general_order.jsonl")}}}
    write_json(lock, tmp_path / "lock.json")
    prov = check_run_spec_frozen(cfg, spec, tmp_path / "lock.json")
    assert prov["manifest_split_sha256"] == {"A": real["files"]["split_A.json"]["sha256"]} and prov["general_order_sha256"] == lock["files"]["general_order.jsonl"]["sha256"]
    bad = {"eval_splits": {"A": {"path": "split_A.json", "content_hash": "0" * 64}}}
    with pytest.raises(RuntimeError, match="content hash"):
        check_run_spec_frozen(cfg, bad, tmp_path / "lock.json")


# ----------------------------------------------------------------------------- loss normalization / data order


def test_example_stats_match_build_example(tiny_model_dir, tmp_path):
    from PIL import Image
    from transformers import AutoProcessor

    from contam.train import IGNORE, build_example, example_stats

    data, root = make_data(tmp_path, n=12)
    examples = json.loads(data.read_text())
    examples.append({"id": "long", "conversations": [{"from": "human", "value": "Hi?"}, {"from": "gpt", "value": "word " * 50}] * 2})
    proc = AutoProcessor.from_pretrained(tiny_model_dir)
    for max_len in (2048, 590, 20):  # 590 / 20: truncation inside the answers / inside the image tokens' prompt
        lengths, sup = example_stats(proc, examples, max_len)
        for ex, n, s in zip(examples, lengths, sup):
            img = Image.open(root / ex["image"]).convert("RGB") if ex.get("image") else None
            b = build_example(proc, ex, img, proc.tokenizer.eos_token_id, max_len)
            assert (len(b["input_ids"]), sum(x != IGNORE for x in b["labels"])) == (n, s), (ex["id"], max_len)


def test_micro_batches_length_sorted_keeps_each_step():
    from contam.train import micro_batches

    lengths = list(np.random.default_rng(0).integers(10, 1000, 100))
    rnd = micro_batches(100, 7, 4, 16)
    srt = micro_batches(100, 7, 4, 16, lengths)
    assert rnd == micro_batches(100, 7, 4)  # random order: the old permutation cut
    assert len(rnd) == len(srt) == 25 and [len(b) for b in srt] == [4] * 25
    for s in range(0, 25, 4):
        assert sorted(sum(rnd[s : s + 4], [])) == sorted(sum(srt[s : s + 4], []))
        flat = sum(srt[s : s + 4], [])
        assert [lengths[i] for i in flat] == sorted((lengths[i] for i in flat), reverse=True)


def _grads(tiny_model_dir, tmp_path, norm, partition, examples, root):
    """Accumulated gradients of one optimizer step over ``partition`` (micro-batches of example indices)."""
    from contam.models import end_token_id
    from contam.train import IGNORE, MicroBatches, build_model, forward_logits, micro_loss, trainable

    cfg = tiny_cfg(tiny_model_dir, lora_dropout=0.0)
    pm, proc = build_model(str(tiny_model_dir), cfg["training"], "float32", 0, "cpu")
    ds = MicroBatches(examples, root, proc, end_token_id(proc.tokenizer), 2048, partition)
    B = sum(len(b) for b in partition)
    batches = [ds[k] for k in range(len(ds))]
    N = sum(int((b["labels"] != IGNORE).sum()) for b in batches)
    for b in batches:
        b.pop("index")
        labels = b.pop("labels")
        micro_loss(forward_logits(pm, b, "cpu", torch.float32, False), labels, norm, B, N, len(partition)).backward()
    return {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p) for n, p in trainable(pm).items()}  # text-only: no projector grad


@pytest.mark.parametrize("norm", ["per_example_mean", "token_mean_global"])
def test_step_gradient_does_not_depend_on_the_micro_batch_partition(tiny_model_dir, tmp_path, norm):
    """Same examples in a step -> same gradient, for random vs length-sorted micro-batches of 4 / 8 / 16."""
    from contam.train import micro_batches

    data, root = make_data(tmp_path, n=16)
    examples = json.loads(data.read_text())
    examples[5]["conversations"][1]["value"] = "A much longer answer than the others, " * 6  # spread the lengths
    lengths = [len(json.dumps(e)) for e in examples]
    ref = _grads(tiny_model_dir, tmp_path, norm, [[i] for i in range(16)], examples, root) if norm == "per_example_mean" else None
    for mbs in (4, 8, 16):
        for lens in (None, lengths):
            g = _grads(tiny_model_dir, tmp_path, norm, micro_batches(16, 3, mbs, 16, lens), examples, root)
            if ref is None:
                ref = g
            for n in ref:
                assert torch.allclose(g[n], ref[n], rtol=1e-4, atol=1e-7), (mbs, lens is None, n)
    assert any(float(v.abs().max()) > 0 for v in ref.values())


def test_per_example_mean_weighs_each_example_one_over_B(tiny_model_dir, tmp_path):
    """The step gradient is the mean of the single-example gradients (each example's own token mean)."""
    data, root = make_data(tmp_path, n=4)
    examples = json.loads(data.read_text())
    step = _grads(tiny_model_dir, tmp_path, "per_example_mean", [[0, 1], [2, 3]], examples, root)
    single = [_grads(tiny_model_dir, tmp_path, "per_example_mean", [[i]], examples, root) for i in range(4)]
    for n in step:
        assert torch.allclose(step[n], sum(g[n] for g in single) / 4, rtol=1e-4, atol=1e-7), n


def test_last_partial_step_and_step_totals(tiny_model_dir, tmp_path, monkeypatch):
    """12 examples, global batch 8: the second step's micro-batches are normalized by its own 4 examples / tokens."""
    import contam.train as tr

    data, root = make_data(tmp_path, n=12)
    calls, real = [], tr.micro_loss

    def spy(logits, labels, norm, step_examples, step_tokens, accum):
        calls.append((int(labels.shape[0]), int((labels != tr.IGNORE).sum()), step_examples, step_tokens, norm))
        return real(logits, labels, norm, step_examples, step_tokens, accum)

    monkeypatch.setattr(tr, "micro_loss", spy)
    res = _run(tiny_cfg(tiny_model_dir, global_batch_size=8), data, root, tmp_path / "run")
    assert res["optimizer_steps"] == 2 and len(calls) == 6 and all(c[4] == "per_example_mean" for c in calls)
    for s, micro in ((0, calls[:4]), (1, calls[4:])):
        assert {c[2] for c in micro} == {sum(c[0] for c in micro)} == {8 if s == 0 else 4}
        assert {c[3] for c in micro} == {sum(c[1] for c in micro)}


def test_micro_batch_without_supervised_tokens_contributes_zero(tiny_model_dir):
    from transformers import AutoModelForVision2Seq

    from contam.train import IGNORE, micro_loss

    model = AutoModelForVision2Seq.from_pretrained(tiny_model_dir)
    ids = torch.randint(1, 1000, (2, 7))
    labels = torch.full_like(ids, IGNORE)
    logits = model(inputs_embeds=model.get_input_embeddings()(ids), attention_mask=torch.ones_like(ids)).logits
    for norm in ("per_example_mean", "token_mean_global", "micro_batch_mean"):
        model.zero_grad()
        loss = micro_loss(logits, labels, norm, 8, 0, 4)
        loss.backward(retain_graph=True)
        assert float(loss) == 0.0
        assert all(torch.isfinite(p.grad).all() and float(p.grad.abs().max()) == 0 for p in model.parameters() if p.grad is not None)


# ----------------------------------------------------------------------------- signals, provenance, images, log


def test_stop_signal_outside_the_loop_stops_without_a_checkpoint(tiny_model_dir, tmp_path, monkeypatch):
    import contam.train as tr

    data, root = make_data(tmp_path)
    real = tr.build_model

    def build_then_signal(*a, **k):  # SIGUSR1 while the model loads
        out = real(*a, **k)
        os.kill(os.getpid(), signal.SIGUSR1)
        return out

    monkeypatch.setattr(tr, "build_model", build_then_signal)
    with pytest.raises(tr.Preempted, match="SIGUSR1"):
        _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "run")
    assert not tr.complete_checkpoints(tmp_path / "run" / "checkpoints")
    assert signal.getsignal(signal.SIGUSR1) is signal.SIG_DFL or callable(signal.getsignal(signal.SIGUSR1))


def test_death_during_export_resumes_at_the_end(tiny_model_dir, tmp_path, monkeypatch):
    import contam.train as tr

    data, root = make_data(tmp_path)
    cfg = tiny_cfg(tiny_model_dir)
    _run(cfg, data, root, tmp_path / "straight")

    def die(*a, **k):
        raise KeyboardInterrupt("killed during the export")

    monkeypatch.setattr(tr, "load_lora", die)
    with pytest.raises(KeyboardInterrupt):
        _run(cfg, data, root, tmp_path / "run")
    monkeypatch.undo()
    assert tr.complete_checkpoints(tmp_path / "run" / "checkpoints")[-1].name == "step-00000003-0000" and not (tmp_path / "run" / "DONE").exists()
    msgs = []
    res = tr.train(cfg, data, tmp_path / "run", 0, "tiny", image_root=root, device="cpu", log=msgs.append)
    assert res["status"] == "done" and res["segments"] == 2 and any("resumed from step-00000003-0000" in m for m in msgs)
    assert not any("checkpoint step-" in m for m in msgs)  # nothing left to train or to save
    _same_adapter(tmp_path / "straight", tmp_path / "run")


def test_provenance_change_is_refused(tiny_model_dir, tmp_path):
    from contam.train import Preempted

    data, root = make_data(tmp_path)
    frozen = {"manifest_lock_sha256": "l", "manifest_split_sha256": {"A": "a"}, "general_order_sha256": "g"}
    with pytest.raises(Preempted):
        _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "run", stop_after_micro=2)
    with pytest.raises(RuntimeError, match="refusing to resume"):  # started with --skip-manifest-check, resumed with a lock
        _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "run", frozen=frozen)
    _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "done", frozen=frozen)
    with pytest.raises(RuntimeError, match="is DONE with config hash"):
        _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "done", frozen={**frozen, "general_order_sha256": "other"})
    assert _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "done", frozen={**frozen, "manifest_lock_sha256": "a later --add"})["status"] == "done-noop"


def test_truncated_image_is_refused_unless_allowed(tiny_model_dir, tmp_path):
    data, root = make_data(tmp_path)
    p = root / "coco" / "train2017" / f"{1:012d}.jpg"
    p.write_bytes(p.read_bytes()[: len(p.read_bytes()) // 2])  # a half-written file from a killed unzip
    with pytest.raises(RuntimeError, match="1 of 12 examples have a missing or unreadable image"):
        _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "run")
    res = _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "run", allow_missing_images=True)
    assert res["n_missing_images_dropped"] == 1 and res["examples_seen"] == 11


def test_partial_log_line_and_save_dedupe(tiny_model_dir, tmp_path):
    """A stop right after a periodic save does not save twice; a partial last log line is dropped on resume."""
    from contam.train import Preempted, train

    data, root = make_data(tmp_path)
    cfg = tiny_cfg(tiny_model_dir, checkpoint_every_steps=1)
    _run(cfg, data, root, tmp_path / "straight")
    msgs = []
    with pytest.raises(Preempted):
        train(cfg, data, tmp_path / "run", 0, "tiny", image_root=root, device="cpu", log=msgs.append, stop_after_micro=2)  # accumulation 2
    saves = [m for m in msgs if "checkpoint step-" in m]
    assert len(saves) == 1 and "(periodic)" in saves[0] and any("already saved (stop: stop_after_micro)" in m for m in msgs)
    with open(tmp_path / "run" / "train_log.jsonl", "a") as f:
        f.write('{"step": 2, "loss": 0.')  # killed mid-write
    res = _run(cfg, data, root, tmp_path / "run")
    assert res["status"] == "done"
    assert _losses(tmp_path / "straight") == _losses(tmp_path / "run")
    _same_adapter(tmp_path / "straight", tmp_path / "run")


def test_checkpoints_kept_after_done_if_asked_and_crash_mid_delete_is_a_noop(tiny_model_dir, tmp_path, monkeypatch):
    import shutil

    data, root = make_data(tmp_path)
    keep = _run(tiny_cfg(tiny_model_dir, keep_checkpoints_after_done=True), data, root, tmp_path / "keep")
    assert [p.name for p in sorted((tmp_path / "keep" / "checkpoints").glob("step-*"))] == ["step-00000002-0000", "step-00000003-0000"]  # periodic, final
    real = shutil.rmtree

    def dying_rmtree(path, *a, **k):  # killed while deleting the checkpoints
        if Path(path).name == "checkpoints":
            real(sorted(Path(path).glob("step-*"))[0])
            raise KeyboardInterrupt("killed mid-delete")
        return real(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", dying_rmtree)
    with pytest.raises(KeyboardInterrupt):
        _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "run")
    monkeypatch.undo()
    assert (tmp_path / "run" / "DONE").exists() and (tmp_path / "run" / "checkpoints").exists()
    res = _run(tiny_cfg(tiny_model_dir), data, root, tmp_path / "run")
    assert res["status"] == "done-noop" and res["config_hash"] == keep["config_hash"]  # keep_checkpoints_after_done is not in the hash
