#!/usr/bin/env python
"""D2 timing run: GPU-hours per training run for M in {130k, 300k}, with the real trainer.

1. Sample. The first --n examples of the general-pool order (``contam.data.general_pool.order_pool``
   on the 665K mix with ``training.general_seed``, computed in memory: nothing is written to the
   splits dir and general_order.jsonl is not needed). That prefix follows the 665K source mix.
   Sources whose images are not on disk yet are handled transparently, per --missing:
     stand-in (default)  the example keeps its own conversation; its image is replaced by an
                         available image from the same prefix. Every image becomes 336 px / 576
                         image tokens, so per-example compute equals the real example's.
     reweight            missing examples are dropped and the order is read further, so the
                         available sources keep their relative shares (renormalized mix).
   The report lists the 665K share, the sample count and the number of stand-ins per source.
2. Probe (per micro-batch size, 16 then 8 then 4): one forward/backward + optimizer step on a
   worst-case micro-batch (every example an image example truncated at max_len). OOM -> next size.
3. Timing: scripts/train_lora.py on the sample for --steps optimizer steps at the first size that
   fit (an OOM there also falls back), global batch unchanged (128 = micro x accumulation).
   Steady state skips the first --warmup-steps steps of each process segment.
4. Projection: GPU-h per run = M / (examples/s) + periodic checkpoint saves + fixed start/finish
   overhead; G0 (5 runs + 6 evaluated checkpoints) and P0 (~98 runs + their checkpoints and the
   base) at EVAL_GPU_H_PER_CKPT (docs/results/w1_base_b3.md, T=0 + teacher forcing, no T=1).

Exposure (``--exposure-d``): the d>0 cell of a run with M examples and N_exp exposed images
replaces the tail of the sample with exposed examples (``contam.data.exposure``, split A by
default) at the run's share K/M = N_exp x d / M, so the timed mix is that run's mix. The exposed
rows are taken round-robin (every image once, then again), and their images must be readable
under paths.llava_images (``nocaps/`` -> the NoCaps images; ``coco/`` is the 665K COCO folder).

Repeats (``--repeat r``): an independent timing run with seed slot r (its own data order and
run dir), for the spread of the steady-state step time. ``scripts/d2_summary.py`` pools them.

Resumable: probe results and the trainer's checkpoints are reused; rerun the same command. Exits 75
if stopped by SIGTERM / SIGUSR1 / SIGHUP. Writes outputs/timing/timing_report.{json,txt} and the
sample (outputs/timing/sample_n{n}.json); trainer checkpoints go under <paths.ckpt_dir>/timing/.

Examples:
    python scripts/timing_run.py                         # n=4096, 30 steps
    python scripts/timing_run.py --sample-only --n 128   # just write a sample (GPU smoke data)
    python scripts/timing_run.py --n 128000 --steps 1000 --exposure-d 64 --n-exp 1000 --M 130000 --repeat 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from contam.config import REPO_ROOT, content_hash, file_sha256, load_config, read_json, write_json
from contam.data.general_pool import order_pool, source_of

EVAL_GPU_H_PER_CKPT = 1.7  # docs/results/w1_base_b3.md, "Eval cost per checkpoint"
M_VALUES = (130_000, 300_000)
G0_RUNS, G0_EVAL_CKPTS = 5, 6  # d=0 x 5 seeds, + the untrained base (eval only)
P0_RUNS = 98  # docs/experiment-plan.pdf, "Run budget" (P0: 98 + 5 large; the 5 large C5 runs are excluded)
MICRO_SIZES = (16, 8, 4)
EXIT_PREEMPTED = 75
FROZEN_PATTERNS = ("split_", "samedist_", "general_order")


def _readable(path: Path) -> bool:
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.convert("RGB")
        return True
    except Exception:
        return False


def build_sample(cfg: dict, n: int, missing_mode: str, log=print) -> tuple[list[dict], dict]:
    """``(examples, info)``: the first ``n`` examples of the general order (see the module docstring)."""
    mix_path = cfg["paths"]["llava_mix"]
    root = Path(cfg["paths"]["llava_images"])
    log(f"reading {mix_path}")
    mix = read_json(mix_path)
    order = order_pool(mix, int(cfg["training"]["general_seed"]))
    share = {s: c / len(mix) for s, c in Counter(source_of(ex) for ex in mix).most_common()}

    # read the order in chunks until n examples are usable (stand-in: the first n always are)
    rows, ok, pos = [], {}, 0
    with ThreadPoolExecutor(16) as pool:
        while pos < len(order) and (len(rows) < n if missing_mode == "stand-in" else sum(1 for r in rows if not r.get("image") or ok[r["image"]]) < n):
            chunk = order[pos : pos + max(n, 1024)]
            pos += len(chunk)
            imgs = [r["image"] for r in chunk if r["image"] and r["image"] not in ok]
            ok.update(zip(imgs, pool.map(lambda p: _readable(root / p), imgs)))
            rows += chunk
            if missing_mode == "stand-in":
                rows = rows[:n]
    if missing_mode == "reweight":
        rows = [r for r in rows if not r["image"] or ok[r["image"]]][:n]
    if len(rows) < n:
        raise SystemExit(f"only {len(rows)} usable examples")
    available = sorted({r["image"] for r in rows if r["image"] and ok[r["image"]]})
    if not available and any(r["image"] for r in rows):
        raise SystemExit(f"no readable image under {root}")

    examples, stand_in = [], Counter()
    for r in rows:
        ex = dict(mix[r["src_index"]])
        ex["src_index"], ex["source"] = r["src_index"], r["source"]
        if ex.get("image") and not ok[ex["image"]]:
            ex["orig_image"] = ex["image"]
            ex["image"] = available[r["src_index"] % len(available)]
            stand_in[r["source"]] += 1
        examples.append(ex)
    got = Counter(ex["source"] for ex in examples)
    by_source = {s: {"share_665k": round(share[s], 4), "share_sample": round(got[s] / n, 4), "n_sample": got[s], "n_stand_in": stand_in[s],
                     "missing_on_disk": got[s] > 0 and stand_in[s] == got[s] if missing_mode == "stand-in" else got[s] == 0}
                 for s in share}
    info = {"n": n, "missing_mode": missing_mode, "order_positions_read": pos, "general_seed": int(cfg["training"]["general_seed"]),
            "llava_mix_sha256": file_sha256(mix_path), "n_stand_in": sum(stand_in.values()), "by_source": by_source,
            "sources_missing": sorted(s for s, v in by_source.items() if v["missing_on_disk"]),
            "sources_partial": sorted(s for s, v in by_source.items() if 0 < v["n_stand_in"] < v["n_sample"])}
    return examples, info


def add_exposure(cfg: dict, examples: list[dict], info: dict, n: int, condition: str, d: int, n_exp: int, M: int, split_name: str) -> tuple[list[dict], dict]:
    """Replace the tail of a general sample with the exposed examples of one (condition, d) cell."""
    from contam.config import load_prompts
    from contam.data.exposure import build_exposure
    from contam.data.nocaps import load_annotations

    K = n_exp * d
    if not 0 < K < M:
        raise SystemExit(f"K = n_exp x d = {K} must be in 1..M-1 (M={M})")
    n_exposed = round(n * K / M)
    splits = Path(cfg["paths"]["splits_dir"])
    split = read_json(splits / f"split_{split_name}.json")
    ex = build_exposure(condition, split, load_annotations(cfg["paths"]["annotations"]), n_exp, d, load_prompts(),
                        shuffle_seed=cfg["exposure"]["shuffle_seed"],
                        samedist=read_json(splits / f"samedist_{split_name}.json") if condition == "samedist" else None)
    if condition == "image":
        raise SystemExit("the image condition needs base-model captions; time paired instead (same images, similar length)")
    rows = ex["rows"]
    picked = [rows[k % len(rows)] for k in range(min(n_exposed, len(rows) * d))]  # round-robin: every image once, then again
    root = Path(cfg["paths"]["llava_images"])
    imgs = sorted({r["image"] for r in picked if r.get("image")})
    with ThreadPoolExecutor(16) as pool:
        bad = [p for p, ok in zip(imgs, pool.map(lambda p: _readable(root / p), imgs)) if not ok]
    if bad:
        raise SystemExit(f"{len(bad)} exposed images are missing under {root}, e.g. {bad[0]}; for NoCaps: ln -s <paths.images_dir> {root / 'nocaps'}")
    exposed = [{"id": r["id"], "image": r.get("image"), "conversations": r["conversations"], "source": f"exposed_{condition}"} for r in picked]
    exposed = [{k: v for k, v in e.items() if v is not None} for e in exposed]
    out = examples[: n - len(exposed)] + exposed
    info = {**info, "n": n, "exposure": {"condition": condition, "d": d, "n_exp": n_exp, "M": M, "K": K, "share": round(K / M, 4), "split": split_name,
                                         "n_exposed_in_sample": len(exposed), "n_general_in_sample": n - len(exposed), "exposure_hash": ex["content_hash"]}}
    return out, info


# ----------------------------------------------------------------------------- probe (subprocess)


def probe(cfg: dict, sample_path: Path, micro_bs: int) -> dict:
    """Peak memory of one worst-case micro-batch step (forward, backward, AdamW step)."""
    import torch

    from contam.models import end_token_id
    from contam.train import build_example, build_model, collate, make_optimizer, micro_loss
    from PIL import Image

    tcfg = cfg["training"]
    max_len = int(tcfg["max_len"])
    pm, processor = build_model(cfg["models"]["main"], tcfg, cfg["models"]["dtype"], 0, "cuda")
    end_id = end_token_id(processor.tokenizer, cfg["models"]["end_token"])
    ex = next(e for e in read_json(sample_path) if e.get("image"))
    img = Image.open(Path(cfg["paths"]["llava_images"]) / ex["image"]).convert("RGB")
    long = {"conversations": [{"from": "human", "value": "<image>\nDescribe the image in detail."},
                              {"from": "gpt", "value": " ".join([ex["conversations"][1]["value"]] * 4000)[:40000]}]}
    item = build_example(processor, long, img, end_id, max_len)
    if len(item["input_ids"]) != max_len:
        raise RuntimeError(f"probe example has {len(item['input_ids'])} tokens, expected {max_len}")
    batch = collate([item] * micro_bs, processor.tokenizer.pad_token_id)
    labels = batch.pop("labels").cuda()
    batch = {k: v.cuda() for k, v in batch.items()}
    batch["pixel_values"] = batch["pixel_values"].to(getattr(torch, cfg["models"]["dtype"]))
    opt, _, _ = make_optimizer(pm, tcfg, 10)
    pm.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autocast("cuda", dtype=getattr(torch, cfg["models"]["dtype"])):
        logits = pm(**batch).logits
    loss = micro_loss(logits, labels, tcfg["loss_normalization"], micro_bs, int((labels != -100).sum()), 1)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    return {"micro_batch_size": micro_bs, "fit": True, "gpu": torch.cuda.get_device_name(0), "tokens_per_example": max_len, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "step_s": time.time() - t0, "loss": float(loss)}


# ----------------------------------------------------------------------------- driver


class Stop:
    sig = None


def _run(cmd: list[str], log_path: Path) -> tuple[int, str]:
    """Run ``cmd`` (stdout+stderr appended to ``log_path``); returns (returncode, its output)."""
    with open(log_path, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        start = f.tell()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        while p.poll() is None:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            if Stop.sig and p.poll() is None:
                p.send_signal(signal.SIGTERM)  # the child may not have got it (e.g. a signal to this process only)
                p.wait()
    with open(log_path) as f:
        f.seek(start)
        return p.returncode, f.read()


def _is_oom(text: str) -> bool:
    return "OutOfMemoryError" in text or "CUDA out of memory" in text


def steady_state(log_path: Path, warmup: int) -> dict:
    recs = [json.loads(line) for line in open(log_path)]
    first = {}
    for r in recs:
        first.setdefault(r["segment"], r["step"])
    keep = [r for r in recs if r["step"] - first[r["segment"]] >= warmup]
    if not keep:
        raise RuntimeError(f"no steady-state steps in {log_path} (warmup {warmup}, {len(recs)} steps)")
    t = sum(r["step_time_s"] for r in keep)
    st = [r["step_time_s"] for r in keep]
    mean = t / len(keep)
    return {"steps_measured": len(keep), "steps_total": len(recs), "examples_per_s": sum(r["examples"] for r in keep) / t,
            "tokens_per_s": sum(r["tokens"] for r in keep) / t, "tokens_per_example": sum(r["tokens"] for r in keep) / sum(r["examples"] for r in keep),
            "step_time_s_mean": mean, "step_time_s_sd": (sum((x - mean) ** 2 for x in st) / max(len(st) - 1, 1)) ** 0.5,
            "peak_mem_gib": max((r["peak_mem_gb"] or 0) for r in recs), "losses": [round(r["loss"], 4) for r in recs]}


def project(ex_per_s: float, ckpt_save_s: float, overhead_s: float, gbs: int, every: int) -> dict:
    out = {}
    for M in M_VALUES:
        steps = math.ceil(M / gbs)
        train_h = M / ex_per_s / 3600
        run_h = train_h + (steps // every) * ckpt_save_s / 3600 + overhead_s / 3600
        g0_train, p0_train = G0_RUNS * run_h, P0_RUNS * run_h
        g0_eval, p0_eval = G0_EVAL_CKPTS * EVAL_GPU_H_PER_CKPT, (P0_RUNS + 1) * EVAL_GPU_H_PER_CKPT
        out[str(M)] = {"optimizer_steps": steps, "train_gpu_h_per_run": round(train_h, 2), "gpu_h_per_run": round(run_h, 2),
                       "G0": {"runs": G0_RUNS, "train_gpu_h": round(g0_train, 1), "eval_ckpts": G0_EVAL_CKPTS, "eval_gpu_h": round(g0_eval, 1), "total_gpu_h": round(g0_train + g0_eval, 1)},
                       "P0": {"runs": P0_RUNS, "train_gpu_h": round(p0_train, 1), "eval_ckpts": P0_RUNS + 1, "eval_gpu_h": round(p0_eval, 1), "total_gpu_h": round(p0_train + p0_eval, 1)}}
    return out


def text_report(rep: dict) -> str:
    s, t = rep["sample"], rep["timing"]
    lines = [f"D2 timing run ({rep['finished']}, {rep['gpu']}; gradient checkpointing {rep['gradient_checkpointing']})", "",
             f"Sample: first {s['n']} examples of the general order (seed {s['general_seed']}), missing images: {s['missing_mode']}"]
    lines.append(f"  {'source':8s} {'665K share':>10s} {'sample':>7s} {'n':>5s} {'stand-in':>8s}")
    for src, v in s["by_source"].items():
        lines.append(f"  {src:8s} {v['share_665k']:10.3f} {v['share_sample']:7.3f} {v['n_sample']:5d} {v['n_stand_in']:8d}")
    lines.append(f"  sources missing on disk: {s['sources_missing'] or 'none'}; partly on disk: {s['sources_partial'] or 'none'}")
    if s.get("exposure"):
        e = s["exposure"]
        lines.append(f"  exposure: {e['condition']} (split {e['split']}), N_exp {e['n_exp']} x d {e['d']} = K {e['K']:,} of M {e['M']:,} "
                     f"(share {e['share']:.3f}); sample {e['n_exposed_in_sample']} exposed + {e['n_general_in_sample']} general")
    lines.append(f"  repeat {rep['args']['repeat']} (seed slot {rep['args']['repeat']})")
    lines += ["", "Micro-batch probe (worst case: every example max_len tokens with an image):"]
    for p in rep["probes"]:
        lines.append(f"  micro {p['micro_batch_size']:2d}: " + (f"fits, peak {p['peak_allocated_gib']:.1f} GiB allocated / {p['peak_reserved_gib']:.1f} reserved of {p['total_gib']:.0f}" if p["fit"] else "OOM"))
    lines += ["", f"Timing (micro {t['micro_batch_size']} x accumulation {t['gradient_accumulation']} = global {t['global_batch_size']}; "
              f"{t['steps_measured']} of {t['steps_total']} steps after {rep['args']['warmup_steps']} warmup):",
              f"  {t['examples_per_s']:.2f} examples/s, {t['tokens_per_s']:.0f} tokens/s ({t['tokens_per_example']:.0f} tokens/example incl. image tokens), "
              f"{t['step_time_s_mean']:.1f} +- {t['step_time_s_sd']:.1f} s/step, peak {t['peak_mem_gib']:.1f} GiB",
              f"  checkpoint save {t['checkpoint_save_s']:.1f} s; fixed start/finish overhead {t['overhead_s']:.0f} s", "",
              f"Projection (1 x {rep['gpu']}; eval {EVAL_GPU_H_PER_CKPT} GPU-h/checkpoint, T=0 + teacher forcing, no T=1 samples):"]
    for M, v in rep["projection"].items():
        lines.append(f"  M={int(M):,}: {v['gpu_h_per_run']:.1f} GPU-h/run ({v['optimizer_steps']} steps); "
                     f"G0 {v['G0']['train_gpu_h']:.0f} train + {v['G0']['eval_gpu_h']:.0f} eval = {v['G0']['total_gpu_h']:.0f} GPU-h; "
                     f"P0 {v['P0']['train_gpu_h']:.0f} train + {v['P0']['eval_gpu_h']:.0f} eval = {v['P0']['total_gpu_h']:.0f} GPU-h")
    lines.append("  (P0 excludes the 5 large C5 runs.)")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=4096, help="sample size (>= steps x 128)")
    ap.add_argument("--steps", type=int, default=30, help="optimizer steps to time")
    ap.add_argument("--warmup-steps", type=int, default=5, help="steps excluded from the steady state (per process segment)")
    ap.add_argument("--missing", choices=["stand-in", "reweight"], default="stand-in")
    ap.add_argument("--micro-sizes", default=",".join(map(str, MICRO_SIZES)))
    ap.add_argument("--out", default=None, help="report dir (default <outputs_dir>/timing)")
    ap.add_argument("--sample-only", action="store_true", help="write the sample and exit")
    ap.add_argument("--repeat", type=int, default=0, help="repeat index r: seed slot r, own run dir (independent repeats for the spread)")
    ap.add_argument("--exposure-d", type=int, default=0, help="time the d>0 cell: replace the sample tail with exposed examples at share N_exp x d / M")
    ap.add_argument("--exposure-condition", default="paired", choices=["paired", "caption", "shuffled", "samedist"])
    ap.add_argument("--n-exp", type=int, default=1000, help="exposed images (D2)")
    ap.add_argument("--M", type=int, default=130000, help="examples per run (D2), sets the exposure share")
    ap.add_argument("--split", default="A")
    ap.add_argument("--probe", type=int, default=None, help=argparse.SUPPRESS)  # internal: one probe in this process
    ap.add_argument("--sample", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.probe:
        print("PROBE " + json.dumps(probe(cfg, Path(args.sample), args.probe)), flush=True)
        return

    out = Path(args.out or Path(cfg["paths"]["outputs_dir"]) / "timing").resolve()
    splits = Path(cfg["paths"]["splits_dir"]).resolve()
    if out == splits or splits in out.parents:
        raise SystemExit(f"--out {out} is inside the splits dir {splits}; frozen-pattern files must never be written there")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"n{args.n}_{args.missing}"
    if args.exposure_d:
        tag += f"_{args.exposure_condition}{args.split}_nexp{args.n_exp}_d{args.exposure_d}_M{args.M}"
    sample_path, info_path = out / f"sample_{tag}.json", out / f"sample_{tag}.info.json"
    assert not any(p in sample_path.name for p in FROZEN_PATTERNS)
    if not sample_path.exists():
        examples, info = build_sample(cfg, args.n, args.missing)
        if args.exposure_d:
            examples, info = add_exposure(cfg, examples, info, args.n, args.exposure_condition, args.exposure_d, args.n_exp, args.M, args.split)
        write_json(examples, sample_path, indent=None)
        write_json(info, info_path)
    if not 0 <= args.repeat < len(cfg["training"]["seeds"]):
        raise SystemExit(f"--repeat {args.repeat}: needs a training seed slot (0..{len(cfg['training']['seeds']) - 1})")
    tag += f"_r{args.repeat}"
    info = read_json(info_path)
    print(json.dumps({k: v for k, v in info.items() if k != "by_source"}), flush=True)
    if args.sample_only:
        print(f"sample: {sample_path}")
        return
    gbs = int(cfg["training"]["global_batch_size"])
    if args.steps * gbs > args.n:
        raise SystemExit(f"--n {args.n} is too small for {args.steps} steps of {gbs}")

    for name in ("SIGTERM", "SIGUSR1", "SIGHUP"):
        signal.signal(getattr(signal, name), lambda s, f: setattr(Stop, "sig", signal.Signals(s).name))
    py, here = sys.executable, Path(__file__).resolve()
    log_path = out / f"timing_{tag}.log"
    cfg_args = ["--config", args.config] if args.config else []
    probes, timing = [], None
    run_tag = f"{tag}_{content_hash(cfg['training'])[:8]}"  # a different training config never reuses a probe or run dir
    for m in [int(x) for x in args.micro_sizes.split(",")]:
        if gbs % m:
            continue
        pf = out / f"probe_{run_tag}_mbs{m}.json"
        if pf.exists():
            pr = read_json(pf)
        else:
            rc, text = _run([py, str(here), *cfg_args, "--probe", str(m), "--sample", str(sample_path)], log_path)
            if Stop.sig:
                sys.exit(EXIT_PREEMPTED)
            if rc == 0:
                pr = json.loads(next(line for line in text.splitlines() if line.startswith("PROBE "))[6:])
            elif _is_oom(text):
                pr = {"micro_batch_size": m, "fit": False}
            else:
                raise SystemExit(f"probe at micro {m} failed (exit {rc}); see {log_path}")
            write_json(pr, pf)
        probes.append(pr)
        print(f"probe micro {m}: {pr}", flush=True)
        if not pr["fit"]:
            continue
        run_dir = Path(cfg["paths"]["ckpt_dir"]) / "timing" / f"{run_tag}_mbs{m}"
        tlog = run_dir / "train_log.jsonl"
        done = tlog.exists() and sum(1 for _ in open(tlog)) >= args.steps
        if not done:
            t0 = time.time()
            rc, text = _run([py, str(REPO_ROOT / "scripts/train_lora.py"), *cfg_args, "--data", str(sample_path), "--seed-slot", str(args.repeat), "--run-id", f"timing_{run_tag}_mbs{m}",
                             "--out-dir", str(run_dir), "--micro-batch-size", str(m), "--max-steps", str(args.steps), "--skip-manifest-check", "--allow-missing-images"], log_path)
            if rc == EXIT_PREEMPTED or Stop.sig:
                print(f"stopped ({Stop.sig or 'trainer preempted'}); rerun to resume", flush=True)
                sys.exit(EXIT_PREEMPTED)
            if rc != 0:
                if _is_oom(text):
                    probes[-1] = {**pr, "fit": False, "oom_in_timing": True}
                    write_json(probes[-1], pf)
                    continue
                raise SystemExit(f"timing run at micro {m} failed (exit {rc}); see {log_path}")
            write_json({"wall_s": time.time() - t0}, run_dir / "timing_wall.json")
        timing = {"micro_batch_size": m, "gradient_accumulation": gbs // m, "global_batch_size": gbs, "run_dir": str(run_dir), **steady_state(tlog, args.warmup_steps)}
        break
    if timing is None:
        raise SystemExit(f"no micro-batch size in {args.micro_sizes} fit")

    # checkpoint save time and fixed overhead (model load, image check, final checkpoint) from the trainer's log
    tl = (Path(timing["run_dir"]) / "train.log").read_text()
    saves = [float(x) for x in re.findall(r"checkpoint step-\S+ \(.*?\) in ([\d.]+) s", tl)]
    timing["checkpoint_save_s"] = max(saves) if saves else float("nan")
    wall = read_json(Path(timing["run_dir"]) / "timing_wall.json")["wall_s"] if (Path(timing["run_dir"]) / "timing_wall.json").exists() else float("nan")
    steps_s = sum(json.loads(line)["step_time_s"] for line in open(Path(timing["run_dir"]) / "train_log.jsonl"))
    timing["overhead_s"] = max(wall - steps_s - sum(saves[:-1]), 0.0)  # the last save is the max-steps one, a real run saves its adapter instead
    rep = {"finished": time.strftime("%F %T"), "gpu": next(p["gpu"] for p in probes if p["fit"]),
           "args": vars(args), "sample": info, "sample_path": str(sample_path), "probes": probes, "timing": timing,
           "eval_gpu_h_per_checkpoint": EVAL_GPU_H_PER_CKPT, "gradient_checkpointing": cfg["training"]["gradient_checkpointing"], "training_config_hash": run_tag.rsplit("_", 1)[1],
           "projection": project(timing["examples_per_s"], timing["checkpoint_save_s"], timing["overhead_s"], gbs, int(cfg["training"]["checkpoint_every_steps"]))}
    write_json(rep, out / f"timing_report_{run_tag}.json")
    txt = text_report(rep)
    (out / f"timing_report_{run_tag}.txt").write_text(txt)
    print(txt, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
