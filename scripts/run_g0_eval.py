#!/usr/bin/env python
"""G0 evaluation chain: the five d=0 checkpoints on one split, then the G0 margins. Does NOT train.

Per checkpoint, a GPU stage (``contam.generate``: prompt ``caption``, T=0, every split image; then
``scripts/score_refs.py``) and a CPU stage (``scripts/evaluate_checkpoint.py --logprobs``; SPICE-bound;
``--reuse-scores``: metrics that are stale only in their group statistics, e.g. made before the
current seen definition or B1 exclusion, are recomputed from their stored per-image CIDEr / SPICE
scores without a SPICE run).
Two-stage pipeline: checkpoint k's CPU stage runs while checkpoint k+1's GPU stage runs; never two
GPU stages or two CPU stages at once. When all five metrics exist, ``scripts/compute_margins.py``
(with the untrained base for the CIDEr sanity check) writes the margins and the G0 report. Without
a current base (missing, or not generated / evaluated as the d=0 runs are) the margins are still
written but are NOT gate-valid, G0 is reported incomplete, and they are recomputed on every run
until a current base exists.

The manifest lock is checked first (contam.data.freeze.check_split_frozen). Resumable: a stage
whose outputs are complete is skipped, an incomplete one is re-run (the CLIs resume themselves).
Any stage error, Ctrl-C, SIGTERM or SIGHUP stops the chain: every running stage's process group is
terminated (SIGTERM, then SIGKILL after 60 s) and the driver exits non-zero. One driver at a time
(an exclusive flock on outputs/g0/.run_g0_eval.lock).

Checkpoint ids are build_g0.py's run ids (G0_d0_s0 .. G0_d0_s4); if its run specs exist
(<outputs_dir>/runs/G0), each id must have one, made for this split. Outputs (generations and
log-probs cover every split image, which is all of NoCaps val, so they do not depend on the split):
    outputs/g0/{run_id}_caption.jsonl            T=0 captions
    outputs/logprobs/{run_id}.jsonl              teacher-forced log p of all refs
    outputs/g0/{run_id}_metrics_v_{split}.json   S / A / V / Q metrics
    outputs/g0/margins_{split}.json              G0 margins; printed report in g0_report_{split}.txt
    outputs/logs/run_g0_eval_{time}.log          chain log; per-stage logs in outputs/logs/g0/

Example:
    python scripts/run_g0_eval.py --split data/splits/split_A.json \
        --run G0_d0_s0=/scratch/.../G0_d0_s0 ... --run G0_d0_s4=/scratch/.../G0_d0_s4 [--dry-run]
"""

from __future__ import annotations

import argparse
import fcntl
import os
import queue
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

from contam.config import REPO_ROOT, load_config, load_prompts, prompt_sha, read_json, read_jsonl
from contam.data.freeze import DEFAULT_LOCK, check_split_frozen
from contam.data.nocaps import load_annotations
from contam.eval.evaluate import load_analysis_exclusion, load_seen_definition, seen_matches

PROMPT_ID = "caption"
N_G0_SEEDS = 5


class Stage(NamedTuple):
    name: str
    cmd: list[str]
    done: Callable[[], bool]  # outputs complete: skip before, required after a run
    log: Path
    log_mode: str = "a"
    check: Callable[[], bool] | None = None  # required after a run instead of ``done`` (if set)


class StageError(RuntimeError):
    pass


class Chain:
    """Runs stages as subprocesses, logs with timestamps, and stops every running stage on abort."""

    def __init__(self, log_path: Path, dry_run: bool = False, poll: float = 5.0):
        self.log_path, self.dry_run, self.poll = Path(log_path), dry_run, poll
        self.abort = threading.Event()
        self._lock = threading.Lock()
        self.running: set[subprocess.Popen] = set()  # stage processes not yet reaped

    def kill_running(self) -> None:
        """Stop every stage process group still running (whichever thread started it)."""
        for p in list(self.running):
            if p.poll() is None:  # already reaped: its pid may be reused, never signal it
                _kill_group(p)

    def log(self, msg: str) -> None:
        line = f"[{datetime.now():%F %T}] {msg}"
        with self._lock:
            print(line, flush=True)
            if not self.dry_run:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(line + "\n")

    def run(self, s: Stage) -> None:
        if s.done():
            self.log(f"{s.name}: complete, skipped")
            return
        if self.dry_run:
            self.log(f"{s.name}: would run  {shlex.join(s.cmd)}  > {s.log}")
            return
        if self.abort.is_set():
            raise StageError(f"{s.name}: not started (chain aborted)")
        s.log.parent.mkdir(parents=True, exist_ok=True)
        self.log(f"{s.name}: start  {shlex.join(s.cmd)}  > {s.log}")
        t0 = time.time()
        with open(s.log, s.log_mode) as f:
            f.write(f"\n[{datetime.now():%F %T}] $ {shlex.join(s.cmd)}\n")
            f.flush()
            # own session: a terminal Ctrl-C reaches only the driver, which then stops the whole group
            p = subprocess.Popen(s.cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO_ROOT, start_new_session=True)
            self.running.add(p)
            try:
                while True:
                    try:
                        rc = p.wait(timeout=self.poll)
                        break
                    except subprocess.TimeoutExpired:
                        if self.abort.is_set():
                            raise StageError(f"{s.name}: terminated (chain aborted)")
            except BaseException:
                self.log(f"{s.name}: stopping process group {p.pid}")
                _kill_group(p)
                raise
            finally:
                self.running.discard(p)
        mins = (time.time() - t0) / 60
        if rc != 0:
            tail = "".join(open(s.log).readlines()[-20:])
            raise StageError(f"{s.name}: exit code {rc} after {mins:.1f} min; see {s.log}\n{tail}")
        if not (s.check or s.done)():
            raise StageError(f"{s.name}: exited 0 after {mins:.1f} min but its outputs are incomplete; see {s.log}")
        self.log(f"{s.name}: done in {mins:.1f} min")


def _kill_group(p: subprocess.Popen) -> None:
    """SIGTERM the stage's process group; SIGKILL what is left after 60 s.

    A second interrupt in the main thread cuts the 60 s short only for a stage the main thread runs;
    a CPU-worker stage always gets the full grace period (bounded at ~70 s).
    """
    for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 10)):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        try:
            p.wait(timeout=wait)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            pass


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt(signal.Signals(signum).name)


def run_pipeline(chain: Chain, jobs: Sequence[tuple[Sequence[Stage], Sequence[Stage]]], final: Sequence[Stage] = ()) -> None:
    """Run ``jobs`` = [(gpu_stages, cpu_stages)] as a two-stage pipeline, then ``final`` serially.

    The main thread runs the GPU stages job by job; a single worker thread runs each job's CPU
    stages once that job's GPU stages are done. The first error (or Ctrl-C / SIGTERM / SIGHUP, raised
    as KeyboardInterrupt in the main thread) aborts both, stops their running stages, and is re-raised.
    """
    if chain.dry_run:
        for gpu, cpu in jobs:
            for s in [*gpu, *cpu]:
                chain.run(s)
        for s in final:
            chain.run(s)
        return
    todo: queue.Queue = queue.Queue()
    errors: list[BaseException] = []
    # not Thread.is_alive()/join(): on 3.11 an interrupted join() can make is_alive() wrongly False (gh-90882)
    worker_done = threading.Event()

    def cpu_worker() -> None:
        try:
            while (stages := todo.get()) is not None:
                try:
                    for s in stages:
                        chain.run(s)
                except BaseException as e:
                    if not chain.abort.is_set():
                        errors.append(e)
                        chain.abort.set()
        finally:
            worker_done.set()

    on_main = threading.current_thread() is threading.main_thread()
    prev = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGTERM, signal.SIGHUP)} if on_main else {}
    worker = threading.Thread(target=cpu_worker, name="cpu", daemon=True)
    worker.start()
    try:
        try:
            for gpu, cpu in jobs:
                if chain.abort.is_set():
                    break
                for s in gpu:
                    chain.run(s)
                todo.put(cpu)
            todo.put(None)
            while not worker_done.wait(chain.poll):
                pass
        except BaseException as e:
            if not chain.abort.is_set():
                errors.append(e)
                chain.abort.set()
            todo.put(None)
            while not worker_done.is_set():  # the worker stops its running stage; wait for that
                try:
                    worker_done.wait(chain.poll)
                except KeyboardInterrupt:
                    pass
        if errors:
            raise errors[0]
        for s in final:
            chain.run(s)
    finally:
        try:
            chain.kill_running()  # belt and braces: nothing outlives the driver
        finally:
            for sig, h in prev.items():
                signal.signal(sig, h)


def _ours(r: dict, prov: dict) -> bool:
    return r.get("prompt_id", PROMPT_ID) == PROMPT_ID and all(r.get(k) == v for k, v in prov.items())


def gens_complete(path: Path, prov: dict, ids: Sequence[int]) -> bool:
    """T=0 rows of this checkpoint / adapter / model / prompt text cover every split image."""
    if not path.exists():
        return False
    have = {int(r["image_id"]) for r in read_jsonl(path) if _ours(r, prov) and float(r["temperature"]) == 0}
    return set(ids) <= have


def logprobs_complete(path: Path, prov: dict, n_refs: dict[int, int], ids: Sequence[int]) -> bool:
    """Every reference of every split image is scored by this checkpoint / adapter / model / prompt text."""
    if not path.exists():
        return False
    have = defaultdict(set)
    for r in read_jsonl(path):
        if _ours(r, prov):
            have[int(r["image_id"])].add(int(r["ref_idx"]))
    return all(have.get(i) == set(range(n_refs[i])) for i in ids)


def json_current(path: Path, inputs: Sequence[Path], split_sha: str | None, excl_sha: str, valid: bool = True, seen: dict | None = None) -> bool:
    """Gate-valid (``valid``; False: not gate-valid) JSON output checked against the current lock entry
    of the split and made under the current B1 analysis exclusion (file sha256) and seen definition
    ``seen``, newer than every input. Outputs from before the exclusion or the seen definition (none
    recorded) are stale and recomputed (metrics: from their stored per-image scores, --reuse-scores)."""
    if split_sha is None or not path.exists() or not all(p.exists() for p in inputs):
        return False
    m = read_json(path)
    return (m.get("gate_valid") is valid and m.get("manifest_split_sha256") == split_sha and (m.get("analysis_exclusion") or {}).get("sha256") == excl_sha
            and seen_matches(m, seen) and all(path.stat().st_mtime >= p.stat().st_mtime for p in inputs))


def generated_current(path: Path, model_prov: dict) -> bool:
    """The metrics at ``path`` come from generations made with the current p_cap / model."""
    gen = read_json(path).get("provenance", {}).get("generations", {})
    return all(gen.get(k) == v for k, v in model_prov.items())


def base_problem(path: Path, model_prov: dict, split: dict, split_sha: str | None, excl_sha: str, seen: dict | None = None) -> str | None:
    """Why the untrained-base metrics cannot serve the G0 base check; None if they are current: generated
    with the current caption prompt and model, and evaluated completely on this split, its current lock
    entry, the current B1 analysis exclusion and seen definition."""
    if not path.exists():
        return f"no untrained-base metrics {path}"
    m = read_json(path)
    if not generated_current(path, model_prov):
        return f"untrained-base metrics {path} were not generated with the current p_cap / model (regenerate and re-evaluate the base)"
    if (not m.get("complete") or m.get("split", {}).get("content_hash") != split["content_hash"] or split_sha is None
            or m.get("manifest_split_sha256") != split_sha or (m.get("analysis_exclusion") or {}).get("sha256") != excl_sha or not seen_matches(m, seen)):
        return (f"untrained-base metrics {path} were not evaluated completely on split {split['name']} under its current lock entry, "
                "the current B1 analysis exclusion and seen definition (re-evaluate the base with scripts/evaluate_checkpoint.py --reuse-scores; its generations are current)")
    return None


def plan(chain: Chain, args, cfg: dict, split: dict, split_sha: str | None) -> tuple[list, list]:
    out = Path(cfg["paths"]["outputs_dir"])
    g0, lp_dir, logs = out / "g0", out / "logprobs", out / "logs" / "g0"
    ids = sorted(set(split["seen"]) | set(split["unseen"]))
    ann = load_annotations(cfg["paths"]["annotations"])
    n_refs = {int(a["image_id"]): len(a["refs"]) for a in ann}
    # the CLIs below use the default prompts file and the configured model
    model_prov = {"model": cfg["models"]["main"], "prompt_sha": prompt_sha(load_prompts()[PROMPT_ID])}
    excl_sha = load_analysis_exclusion(cfg)["sha256"]
    seen = load_seen_definition(cfg, split, ann)  # exposure.n_exp; the CLIs below load the same default
    py = [sys.executable]
    common = (["--config", args.config] if args.config else []) + ["--lock", str(args.lock)]
    sp = ["--split", str(args.split)]
    jobs, metrics = [], []
    for run_id, adapter in args.runs.items():
        gens, lp, met = g0 / f"{run_id}_caption.jsonl", lp_dir / f"{run_id}.jsonl", g0 / f"{run_id}_metrics_v_{split['name']}.json"
        ck = ["--checkpoint-id", run_id]
        prov = dict(model_prov, checkpoint_id=run_id, adapter=adapter)
        gpu = [
            Stage(f"{run_id}/generate", py + ["-m", "contam.generate", *common, *ck, "--adapter", adapter, *sp, "--prompt-id", PROMPT_ID, "--temperature", "0", "--out", str(gens)],
                  lambda gens=gens, prov=prov: gens_complete(gens, prov, ids), logs / f"{run_id}_generate.log"),
            Stage(f"{run_id}/score_refs", py + [str(REPO_ROOT / "scripts/score_refs.py"), *common, *ck, "--adapter", adapter, *sp, "--prompt-id", PROMPT_ID, "--out", str(lp)],
                  lambda lp=lp, prov=prov: logprobs_complete(lp, prov, n_refs, ids), logs / f"{run_id}_score_refs.log"),
        ]
        cpu = [
            Stage(f"{run_id}/evaluate", py + [str(REPO_ROOT / "scripts/evaluate_checkpoint.py"), *common, *ck, "--generations", str(gens), "--logprobs", str(lp), *sp,
                                              "--prompt-id", PROMPT_ID, "--reuse-scores", "--out", str(met)],
                  lambda met=met, gens=gens, lp=lp: json_current(met, [gens, lp], split_sha, excl_sha, seen=seen), logs / f"{run_id}_evaluate.log"),
        ]
        jobs.append((gpu, cpu))
        metrics.append(met)
    final = []
    if len(args.runs) == N_G0_SEEDS:
        margins = g0 / f"margins_{split['name']}.json"
        base = Path(args.base)
        cmd = py + [str(REPO_ROOT / "scripts/compute_margins.py"), *common, *sp, "--d0", *map(str, metrics), "--out", str(margins)]
        problem = base_problem(base, model_prov, split, split_sha, excl_sha, seen)
        base_gens, base_lp = g0 / "base_caption.jsonl", lp_dir / "base.jsonl"
        if problem is not None and base.exists() and base_gens.exists() and base_lp.exists() and generated_current(base, model_prov):
            # generations current, only the evaluation is stale (e.g. an older seen definition): re-evaluate it from its stored scores first
            base_id = read_json(base).get("checkpoint_id") or "base"
            final.append(Stage("G0/base_evaluate", py + [str(REPO_ROOT / "scripts/evaluate_checkpoint.py"), *common, "--checkpoint-id", base_id, "--generations", str(base_gens),
                                                           "--logprobs", str(base_lp), *sp, "--prompt-id", PROMPT_ID, "--reuse-scores", "--out", str(base)],
                               lambda: base_problem(base, model_prov, split, split_sha, excl_sha, seen) is None, logs / "base_evaluate.log"))
            chain.log(f"note: {problem}; re-evaluating it first (G0/base_evaluate, CPU, stored scores reused)")
            problem = None
        if problem is None:
            cmd[-2:-2] = ["--base", str(base)]

            def margins_done() -> bool:  # gate-valid, with the check of this base
                if not json_current(margins, [*metrics, base], split_sha, excl_sha, seen=seen):
                    return False
                m = read_json(margins)
                return "base_cider_check" in m and m.get("base_file") == os.path.realpath(base)

            final.append(Stage("G0/compute_margins", cmd, margins_done, g0 / f"g0_report_{split['name']}.txt", "w"))
        else:
            chain.log(f"WARNING: {problem}; treated as missing. G0 incomplete: margins are written without the base CIDEr check and are NOT gate-valid "
                      "(recomputed on every run until a current base exists)")

            def margins_without_base() -> bool:  # written, and carrying no (old) base check
                if not json_current(margins, metrics, split_sha, excl_sha, valid=False, seen=seen):
                    return False
                m = read_json(margins)
                return m.get("base_check") == "missing" and "base_cider_check" not in m

            # never final without a base: done() is False, so the stage always runs
            final.append(Stage("G0/compute_margins", cmd, lambda: False, g0 / f"g0_report_{split['name']}.txt", "w", margins_without_base))
    return jobs, final


def check_run_specs(chain: Chain, args, split: dict, lock_sha: str | None) -> None:
    """Each run id must have a build_g0.py spec for a d=0 run on this split (if the specs exist)."""
    spec_dir = Path(args.specs)
    if not spec_dir.is_dir():
        chain.log(f"WARNING: no build_g0.py run specs in {spec_dir}; run ids not cross-checked")
        return
    for run_id in args.runs:
        p = spec_dir / f"{run_id}.json"
        if not p.exists():
            raise SystemExit(f"{run_id}: no run spec {p} (build_g0.py run ids are G0_d0_s0 .. G0_d0_s{N_G0_SEEDS - 1})")
        spec = read_json(p)
        sp = spec.get("eval_splits", {}).get(split["name"])
        if spec.get("d") != 0 or not sp or sp["content_hash"] != split["content_hash"]:
            raise SystemExit(f"{run_id}: run spec {p} is not a d=0 run evaluated on this split {split['name']}")
        if spec.get("manifest_lock_sha256") != lock_sha:
            chain.log(f"note: {run_id} was built under manifest lock file {spec.get('manifest_lock_sha256')}, now {lock_sha} (e.g. after an --add; provenance only)")


def take_lock(path: Path):
    """Exclusive flock held for the driver's lifetime (the returned file must stay open)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another run_g0_eval.py is running (it holds {path}); not starting a second one")
    return f


def parse_runs(items: Sequence[str]) -> dict[str, str]:
    runs = {}
    for it in items:
        run_id, sep, adapter = it.partition("=")
        if not sep or not run_id or not adapter:
            raise SystemExit(f"--run expects ID=ADAPTER_DIR, got {it!r}")
        if run_id in runs:
            raise SystemExit(f"duplicate run id {run_id}")
        runs[run_id] = str(Path(adapter).resolve())
    return runs


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default=str(REPO_ROOT / "data/splits/split_A.json"))
    ap.add_argument("--run", action="append", required=True, help="RUN_ID=ADAPTER_DIR, once per d=0 checkpoint (margins need all five)")
    ap.add_argument("--specs", default=None, help="build_g0.py run specs (default: <outputs_dir>/runs/G0)")
    ap.add_argument("--base", default=None, help="untrained-base metrics for the CIDEr check (default: <outputs_dir>/g0/base_metrics_v.json)")
    ap.add_argument("--lock", default=str(DEFAULT_LOCK))
    ap.add_argument("--dry-run", action="store_true", help="print the planned stages; run nothing")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    out = Path(cfg["paths"]["outputs_dir"])
    args.runs = parse_runs(args.run)
    args.specs = args.specs or out / "runs" / "G0"
    args.base = args.base or out / "g0" / "base_metrics_v.json"
    chain = Chain(out / "logs" / f"run_g0_eval_{datetime.now():%Y%m%d-%H%M%S}.log", dry_run=args.dry_run)
    t0 = time.time()
    split = read_json(args.split)
    chain.log(f"G0 eval chain: {len(args.runs)} d=0 checkpoint(s) on split {split['name']} ({args.split}){'  [DRY RUN]' if args.dry_run else ''}")
    if not args.dry_run:
        args.run_lock = take_lock(out / "g0" / ".run_g0_eval.lock")  # held until the driver exits
    try:
        frozen = check_split_frozen(cfg, args.split, args.lock)
        chain.log(f"manifest lock ok: {args.lock} (file sha256 {frozen['manifest_lock_sha256']}); {Path(args.split).name} entry sha256 {frozen['manifest_split_sha256']}")
    except (RuntimeError, FileNotFoundError) as e:
        if not args.dry_run:
            chain.log(f"FAILED: manifest check: {e}")
            sys.exit(1)
        frozen = {"manifest_split_sha256": None, "manifest_lock_sha256": None}
        chain.log(f"WARNING (dry run continues; a real run stops here): manifest check failed: {e}")
    check_run_specs(chain, args, split, frozen["manifest_lock_sha256"])
    for run_id, adapter in args.runs.items():
        if not Path(adapter).is_dir():
            if not args.dry_run:
                chain.log(f"FAILED: {run_id}: adapter dir {adapter} does not exist")
                sys.exit(1)
            chain.log(f"WARNING: {run_id}: adapter dir {adapter} does not exist")
    jobs, final = plan(chain, args, cfg, split, frozen["manifest_split_sha256"])
    if not final:
        chain.log(f"margins skipped: {len(args.runs)}/{N_G0_SEEDS} d=0 checkpoints given")
    try:
        run_pipeline(chain, jobs, final)
    except (StageError, KeyboardInterrupt) as e:
        chain.log(f"FAILED after {(time.time() - t0) / 3600:.2f} h: {e or type(e).__name__}")
        sys.exit(1)
    if args.dry_run:
        chain.log("DRY RUN: nothing was run")
        return
    if final and not final[-1].done():
        chain.log(f"G0 incomplete: base missing/stale (see the WARNING above); margins in {final[-1].log.parent} are NOT gate-valid")
    result = f"margins and report in {final[-1].log.parent}" if final else "margins not computed"
    chain.log(f"DONE: {len(jobs)} d=0 checkpoint(s) evaluated on split {split['name']}; {result}; {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
