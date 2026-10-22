"""scripts/run_g0_eval.py: two-stage GPU/CPU pipeline, resume and failure handling with stub commands."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

import pytest
import yaml

from contam.config import REPO_ROOT, file_sha256, load_config, load_prompts, prompt_sha, write_json
from contam.data.splits import make_split
from tests.conftest import locked_split, make_annotations

_spec = importlib.util.spec_from_file_location("run_g0_eval", REPO_ROOT / "scripts" / "run_g0_eval.py")
g0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g0)

STUB = """
import json, sys, time
events, name, out, secs, rc = sys.argv[1:6]
with open(events, "a") as f:
    f.write(json.dumps(["start", name, time.time()]) + "\\n")
time.sleep(float(secs))
if out != "-":
    open(out, "w").write("ok")
with open(events, "a") as f:
    f.write(json.dumps(["end", name, time.time()]) + "\\n")
sys.exit(int(rc))
"""


def _stage(tmp_path, name, secs, rc=0, writes=True):
    out = tmp_path / "out" / name.replace("/", "_")
    out.parent.mkdir(exist_ok=True)
    cmd = [sys.executable, "-c", STUB, str(tmp_path / "events.jsonl"), name, str(out) if writes else "-", str(secs), str(rc)]
    return g0.Stage(name, cmd, out.exists, tmp_path / "logs" / f"{name.replace('/', '_')}.log")


def _jobs(tmp_path, n=3, gpu_secs=0.5, cpu_secs=0.8, **over):
    jobs = []
    for k in range(n):
        gpu = [over.get(f"s{k}/gen") or _stage(tmp_path, f"s{k}/gen", gpu_secs / 2), over.get(f"s{k}/score") or _stage(tmp_path, f"s{k}/score", gpu_secs / 2)]
        jobs.append((gpu, [over.get(f"s{k}/eval") or _stage(tmp_path, f"s{k}/eval", cpu_secs)]))
    return jobs, [_stage(tmp_path, "margins", 0.05)]


def _intervals(tmp_path):
    ev = {}
    if not (tmp_path / "events.jsonl").exists():
        return ev
    for kind, name, t in map(json.loads, open(tmp_path / "events.jsonl")):
        ev.setdefault(name, [None, None])[kind == "end"] = t
    return ev


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def test_pipeline_overlaps_cpu_with_next_gpu_and_never_doubles(tmp_path):
    chain = g0.Chain(tmp_path / "chain.log", poll=0.02)
    jobs, final = _jobs(tmp_path)
    g0.run_pipeline(chain, jobs, final)
    iv = _intervals(tmp_path)
    gpu = [iv[n] for n in sorted(iv) if n.endswith(("/gen", "/score"))]
    cpu = [iv[n] for n in sorted(iv) if n.endswith("/eval")]
    assert len(gpu) == 6 and len(cpu) == 3 and all(None not in v for v in iv.values())
    assert not any(_overlap(a, b) for i, a in enumerate(gpu) for b in gpu[i + 1 :])
    assert not any(_overlap(a, b) for i, a in enumerate(cpu) for b in cpu[i + 1 :])
    for k in range(3):
        assert iv[f"s{k}/eval"][0] >= iv[f"s{k}/score"][1] >= iv[f"s{k}/gen"][1]
    assert _overlap(iv["s0/eval"], iv["s1/gen"])  # pipelined: CPU stage k runs during GPU stage k+1
    assert iv["margins"][0] >= max(v[1] for v in cpu)
    log = (tmp_path / "chain.log").read_text()
    assert "s2/eval: done in" in log and "margins: done in" in log


def test_pipeline_resume_skips_complete_stages(tmp_path):
    chain = g0.Chain(tmp_path / "chain.log", poll=0.02)
    jobs, final = _jobs(tmp_path, gpu_secs=0.1, cpu_secs=0.1)
    g0.run_pipeline(chain, jobs, final)
    (tmp_path / "events.jsonl").unlink()
    (tmp_path / "out" / "s1_score").unlink()  # incomplete GPU output: re-run it, and only it
    (tmp_path / "out" / "s2_eval").unlink()
    g0.run_pipeline(chain, jobs, final)
    assert sorted(_intervals(tmp_path)) == ["s1/score", "s2/eval"]
    assert "s0/gen: complete, skipped" in (tmp_path / "chain.log").read_text()


def test_pipeline_gpu_failure_stops_everything(tmp_path):
    chain = g0.Chain(tmp_path / "chain.log", poll=0.02)
    jobs, final = _jobs(tmp_path, cpu_secs=0.1, **{"s0/eval": _stage(tmp_path, "s0/eval", 30), "s1/score": _stage(tmp_path, "s1/score", 0.1, rc=3)})
    t0 = time.time()
    with pytest.raises(g0.StageError, match="s1/score: exit code 3"):
        g0.run_pipeline(chain, jobs, final)
    assert time.time() - t0 < 20  # the running CPU stage was terminated, not waited for
    iv = _intervals(tmp_path)
    assert iv["s0/eval"][1] is None and not (tmp_path / "out" / "s0_eval").exists()
    assert "s2/gen" not in iv and "margins" not in iv


def test_pipeline_cpu_failure_and_incomplete_output_are_errors(tmp_path):
    chain = g0.Chain(tmp_path / "chain.log", poll=0.02)
    jobs, final = _jobs(tmp_path, gpu_secs=0.4, cpu_secs=0.1, **{"s0/eval": _stage(tmp_path, "s0/eval", 0.1, writes=False)})
    with pytest.raises(g0.StageError, match="s0/eval: exited 0 .* outputs are incomplete"):
        g0.run_pipeline(chain, jobs, final)
    assert "margins" not in _intervals(tmp_path)


def test_dry_run_runs_nothing(tmp_path, capsys):
    chain = g0.Chain(tmp_path / "chain.log", dry_run=True)
    jobs, final = _jobs(tmp_path)
    g0.run_pipeline(chain, jobs, final)
    assert not (tmp_path / "events.jsonl").exists() and not (tmp_path / "chain.log").exists()
    assert capsys.readouterr().out.count("would run") == 10


def _g0_env(tmp_path, base=True):
    ann = make_annotations(2, 4, 2, seed=5)
    split = make_split(ann, 1, 2, name="A")
    cfg_path, lock, split_path, split_sha = locked_split(tmp_path, split)
    raw = yaml.safe_load(cfg_path.read_text())
    raw["paths"].update(annotations=str(tmp_path / "ann.json"), outputs_dir=str(tmp_path / "outputs"))
    cfg_path.write_text(yaml.safe_dump(raw))
    write_json({"meta": {}, "images": ann}, tmp_path / "ann.json")
    for k in range(5):
        write_json({"run_id": f"G0_d0_s{k}", "d": 0, "manifest_lock_sha256": file_sha256(lock), "eval_splits": {"A": {"path": "split_A.json", "content_hash": split["content_hash"]}}},
                   tmp_path / "outputs" / "runs" / "G0" / f"G0_d0_s{k}.json")
        (tmp_path / f"ad{k}").mkdir()
    if base:
        write_json({"summary": {}}, tmp_path / "outputs" / "g0" / "base_metrics_v.json")
    argv = ["--config", str(cfg_path), "--lock", str(lock), "--split", str(split_path)] + [f"--run=G0_d0_s{k}={tmp_path / f'ad{k}'}" for k in range(5)]
    return ann, split, load_config(cfg_path), lock, split_sha, argv


def _prov(cfg, k, tmp_path, **over):
    return dict(checkpoint_id=f"G0_d0_s{k}", adapter=str(tmp_path / f"ad{k}"), model=cfg["models"]["main"], prompt_sha=prompt_sha(load_prompts()["caption"]), **over)


def test_main_dry_run_plans_g0_layout(tmp_path, capsys):
    ann, split, cfg, lock, split_sha, argv = _g0_env(tmp_path)
    argv = argv + ["--dry-run"]
    with open(tmp_path / "outputs" / "g0" / "G0_d0_s0_caption.jsonl", "w") as f:  # s0's generations are complete
        for a in ann:
            f.write(json.dumps(dict(_prov(cfg, 0, tmp_path), image_id=a["image_id"], prompt_id="caption", temperature=0.0)) + "\n")
    g0.main(argv)
    out = capsys.readouterr().out
    assert f"manifest lock ok: {lock} (file sha256 {file_sha256(lock)}); split_A.json entry sha256 {split_sha}" in out
    assert "G0_d0_s0/generate: complete, skipped" in out and out.count("would run") == 15  # 15 stages - 1 complete + margins
    assert "outputs/g0/G0_d0_s4_metrics_v_A.json" in out and "outputs/logprobs/G0_d0_s4.jsonl" in out and "margins_A.json" in out
    assert "--base" in out and "WARNING: no untrained-base" not in out
    assert not (tmp_path / "outputs" / "logs").exists() and not (tmp_path / "outputs" / "g0" / ".run_g0_eval.lock").exists()

    with pytest.raises(SystemExit, match="no run spec"):
        g0.main(argv[:-2] + [f"--run=d0_s4={tmp_path / 'ad4'}", "--dry-run"])


def test_main_warns_without_base_and_base_makes_margins_stale(tmp_path, capsys):
    ann, split, cfg, lock, split_sha, argv = _g0_env(tmp_path, base=False)
    g0.main(argv + ["--dry-run"])
    out = capsys.readouterr().out
    assert "WARNING: no untrained-base metrics" in out and "--base" not in out
    args = g0.argparse.Namespace(runs=g0.parse_runs([a.removeprefix("--run=") for a in argv[6:]]), config=None, lock=lock, split=argv[5], base=tmp_path / "outputs" / "g0" / "base_metrics_v.json")
    chain = g0.Chain(tmp_path / "c.log", dry_run=True)
    _, final = g0.plan(chain, args, cfg, split, split_sha)
    metrics = [tmp_path / "outputs" / "g0" / f"G0_d0_s{k}_metrics_v_A.json" for k in range(5)]
    for m in metrics:
        write_json({}, m)
    margins = tmp_path / "outputs" / "g0" / "margins_A.json"
    time.sleep(0.01)
    write_json({"gate_valid": True, "manifest_split_sha256": split_sha}, margins)
    assert final[0].done()
    time.sleep(0.01)
    write_json({"summary": {}}, args.base)  # the base appears later: margins must be recomputed
    _, final = g0.plan(chain, args, cfg, split, split_sha)
    assert not final[0].done()
    write_json({"gate_valid": True, "manifest_split_sha256": split_sha, "base_cider_check": {}}, margins)
    assert final[0].done()


def test_gens_and_logprobs_complete(tmp_path):
    cfg = load_config()
    ids = [0, 1, 2]
    prov = _prov(cfg, 0, tmp_path)
    gens, lp = tmp_path / "g.jsonl", tmp_path / "lp.jsonl"
    assert not g0.gens_complete(gens, prov, ids) and not g0.logprobs_complete(lp, prov, {i: 2 for i in ids}, ids)

    def write(path, rows):
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)

    row = lambda i, **o: dict(prov, image_id=i, prompt_id="caption", temperature=0.0, **o)  # noqa: E731
    write(gens, [row(i) for i in ids])
    assert g0.gens_complete(gens, prov, ids)
    write(gens, [row(i) for i in ids[:2]])  # partial
    assert not g0.gens_complete(gens, prov, ids)
    for other in (dict(prompt_sha="X"), dict(model="other/model"), dict(adapter="/other"), dict(checkpoint_id="G0_d0_s1"), dict(prompt_id="verbose"), dict(temperature=1.0)):
        write(gens, [row(i) for i in ids[:2]] + [dict(row(2), **other)])
        assert not g0.gens_complete(gens, prov, ids), other

    lrow = lambda i, j, **o: {**prov, "image_id": i, "ref_idx": j, "prompt_id": "caption", **o}  # noqa: E731
    write(lp, [lrow(i, j) for i in ids for j in range(2)])
    assert g0.logprobs_complete(lp, prov, {i: 2 for i in ids}, ids)
    assert not g0.logprobs_complete(lp, prov, {i: 3 for i in ids}, ids)  # a missing ref
    for other in (dict(prompt_sha="X"), dict(model="other/model"), dict(prompt_id="verbose")):
        write(lp, [lrow(i, j) for i in ids for j in range(2) if (i, j) != (2, 1)] + [lrow(2, 1, **other)])
        assert not g0.logprobs_complete(lp, prov, {i: 2 for i in ids}, ids), other


def test_json_current(tmp_path):
    inp, out = tmp_path / "in.jsonl", tmp_path / "m.json"
    inp.write_text("x")
    assert not g0.json_current(out, [inp], "S")
    time.sleep(0.01)
    write_json({"gate_valid": True, "manifest_split_sha256": "S"}, out)
    assert g0.json_current(out, [inp], "S")
    assert not g0.json_current(out, [inp], None)  # manifest check failed
    assert not g0.json_current(out, [inp], "T")  # other split entry
    assert not g0.json_current(out, [inp, tmp_path / "missing"], "S")
    write_json({"gate_valid": False, "manifest_split_sha256": "S"}, out)  # partial / unchecked
    assert not g0.json_current(out, [inp], "S")
    write_json({"gate_valid": True, "manifest_split_sha256": "S"}, out)
    time.sleep(0.01)
    inp.write_text("y")  # input regenerated after the output: stale
    assert not g0.json_current(out, [inp], "S")


def test_second_driver_refused(tmp_path):
    held = g0.take_lock(tmp_path / ".run_g0_eval.lock")
    code = f"import importlib.util as u; s = u.spec_from_file_location('m', {str(REPO_ROOT / 'scripts/run_g0_eval.py')!r}); m = u.module_from_spec(s); s.loader.exec_module(m); m.take_lock(__import__('pathlib').Path({str(tmp_path / '.run_g0_eval.lock')!r}))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode != 0 and "another run_g0_eval.py is running" in r.stderr
    held.close()
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).returncode == 0


SIGNAL_DRIVER = """
import importlib.util, signal, sys
from pathlib import Path
signal.signal(signal.SIGINT, signal.default_int_handler)
spec = importlib.util.spec_from_file_location("m", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d = Path(sys.argv[2]); tag = sys.argv[3]; cpu_only = sys.argv[4] == "cpu_only"
chain = m.Chain(d / "chain.log", poll=0.2)
cpu = m.Stage("cpu", ["sleep", tag + "1"], lambda: False, d / "cpu.log")
gpu = m.Stage("gpu", ["sh", "-c", f"sleep {tag}2 & sleep {tag}3"], lambda: False, d / "gpu.log")  # a grandchild too
try:
    # cpu_only: the main thread is waiting for the worker's last CPU stage when the signal arrives
    m.run_pipeline(chain, [([], [cpu])] if cpu_only else [([], [cpu]), ([gpu], [])])
except KeyboardInterrupt as e:
    print("interrupted", e)
    sys.exit(130)
"""


def _sleeps(tag):
    found = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if cmd[:1] == [b"sleep"] and len(cmd) > 1 and cmd[1].startswith(tag.encode()):
            found.append(int(pid))
    return found


@pytest.mark.parametrize("sig,mode", [("SIGTERM", "both"), ("SIGINT", "both"), ("SIGHUP", "both"), ("SIGTERM", "cpu_only"), ("SIGINT", "cpu_only")])
def test_signal_to_driver_leaves_no_stage_processes(tmp_path, sig, mode):
    tag = f"9{os.getpid() % 1000:03d}{signal.Signals[sig].value:02d}{int(mode == 'cpu_only')}"
    p = subprocess.Popen([sys.executable, "-c", SIGNAL_DRIVER, str(REPO_ROOT / "scripts/run_g0_eval.py"), str(tmp_path), tag, mode], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    n = 1 if mode == "cpu_only" else 3
    try:
        deadline = time.time() + 60
        while len(_sleeps(tag)) < n and time.time() < deadline:
            time.sleep(0.1)
        assert len(_sleeps(tag)) == n, "stages did not start"
        os.kill(p.pid, getattr(signal, sig))  # the driver only, not its process group
        out, _ = p.communicate(timeout=60)
        assert p.returncode == 130 and "interrupted" in out, out
        time.sleep(0.2)
        assert _sleeps(tag) == []
    finally:
        for pid in _sleeps(tag):
            os.kill(pid, signal.SIGKILL)
        p.kill()
