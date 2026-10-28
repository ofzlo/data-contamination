"""Run lock of contam.train (one trainer per out dir across nodes) and its use in scripts/run_g0_train.sh (no torch)."""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

import contam.train as tr
from contam.config import REPO_ROOT


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(tr, "LOCK_CONFIRM_S", 0.05)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)


def _held(out, job="111", age=0.0):
    out.mkdir(parents=True, exist_ok=True)
    p = out / tr.RUN_LOCK
    p.write_text(json.dumps({"host": "nodeX", "pid": 4242, "slurm_job_id": job, "slurm_restart_count": "0", "started_at": "then", "token": "theirs"}))
    _age(p, age)
    return p


def _age(p, secs):
    t = time.time() - secs
    os.utime(p, (t, t))


def _log(out, line="step 10", age=0.0, name="train.log"):
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(line + "\n")
    _age(out / name, age)


def test_lock_is_exclusive_and_released(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    lock = tr.acquire_run_lock(out, 600)
    rec = json.loads((out / tr.RUN_LOCK).read_text())
    assert rec["pid"] == os.getpid() and set(rec) >= {"host", "slurm_job_id", "slurm_restart_count", "started_at"}
    with pytest.raises(tr.RunBusy, match=f"pid {os.getpid()}"):
        tr.acquire_run_lock(out, 600)
    lock.release()
    assert not (out / tr.RUN_LOCK).exists() and (out / tr.RUN_RELEASED).exists()
    tr.acquire_run_lock(out, 600).release()


def test_fresh_foreign_lock_is_busy(tmp_path):
    out = tmp_path / "run"
    p = _held(out, age=100)
    with pytest.raises(tr.RunBusy, match="host nodeX pid 4242 SLURM job 111"):
        tr.acquire_run_lock(out, 600)
    assert json.loads(p.read_text())["token"] == "theirs"


def test_stale_lock_is_taken_over_unless_the_log_is_fresh(tmp_path):
    out = tmp_path / "run"
    _held(out, age=1000)
    _log(out, age=10)
    with pytest.raises(tr.RunBusy):  # heartbeat stale, but something still writes the log
        tr.acquire_run_lock(out, 600)
    _age(out / "train.log", 1000)
    lock = tr.acquire_run_lock(out, 600)
    assert lock.owned() and json.loads((out / tr.RUN_LOCK).read_text())["pid"] == os.getpid()
    lock.release()
    assert not (out / tr.RUN_LOCK).exists()


def test_lock_of_an_earlier_instance_of_the_same_slurm_job_is_taken_over(tmp_path, monkeypatch):
    out = tmp_path / "run"
    p = _held(out, job="222", age=5)  # held lock records restart "0"
    _log(out, age=5)
    monkeypatch.setenv("SLURM_JOB_ID", "333")
    with pytest.raises(tr.RunBusy):  # another job
        tr.acquire_run_lock(out, 600)
    monkeypatch.setenv("SLURM_JOB_ID", "222")
    with pytest.raises(tr.RunBusy):  # same job, same restart count (unset = 0), fresh lock: a live second trainer
        tr.acquire_run_lock(out, 600)
    monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
    with pytest.raises(tr.RunBusy):  # requeued, but the old instance beat less than 2 heartbeats ago
        tr.acquire_run_lock(out, 600)
    _age(p, 2 * tr.LOCK_HEARTBEAT_S + 5)
    lock = tr.acquire_run_lock(out, 600)  # requeued instance, old one silent for 2 heartbeats (log still fresh)
    assert json.loads(p.read_text())["slurm_restart_count"] == "1"
    lock.release()


def test_fresh_log_without_lock_is_busy_unless_stopped_cleanly(tmp_path):
    out = tmp_path / "run"
    _log(out, age=30)  # a trainer from before the run lock
    with pytest.raises(tr.RunBusy, match="no RUN.lock"):
        tr.acquire_run_lock(out, 600)
    _log(out, age=30, line="[x] G0_d0_s1: stopped by SIGUSR1; checkpoint saved, exiting 75 (rerun to resume)")
    tr.acquire_run_lock(out, 600).release()  # its CLI logged a preemption exit
    _log(out, age=30, name="train_log.jsonl")  # a trainer wrote after that release ...
    _age(out / tr.RUN_RELEASED, 100)
    _log(out, age=30)
    with pytest.raises(tr.RunBusy):
        tr.acquire_run_lock(out, 600)
    _age(out / tr.RUN_RELEASED, 0)  # ... or not
    tr.acquire_run_lock(out, 600).release()
    _age(out / "train.log", 1000)
    _age(out / "train_log.jsonl", 1000)
    (out / tr.RUN_RELEASED).unlink()
    tr.acquire_run_lock(out, 600).release()  # old logs


def test_heartbeat_survives_a_read_error_and_detects_a_takeover(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "LOCK_HEARTBEAT_S", 0.05)
    out = tmp_path / "run"
    out.mkdir()
    real, fails = tr._read_lock_strict, [1]

    def flaky(path):
        if fails[0]:
            fails[0] -= 1
            raise OSError("transient")
        return real(path)

    monkeypatch.setattr(tr, "_read_lock_strict", flaky)
    lock = tr.acquire_run_lock(out, 600)
    _age(out / tr.RUN_LOCK, 1000)
    time.sleep(1.5)
    assert fails == [0] and lock._thread.is_alive() and not lock.lost.is_set() and lock.owned()
    assert time.time() - (out / tr.RUN_LOCK).stat().st_mtime < 5
    _held(out)  # someone took it over
    assert lock.lost.wait(5) and lock.thief["pid"] == 4242 and not lock._thread.is_alive()
    lock.release()
    assert json.loads((out / tr.RUN_LOCK).read_text())["token"] == "theirs" and not (out / tr.RUN_RELEASED).exists()


def test_takeover_aborts_if_the_lock_changed_after_the_staleness_check(tmp_path, monkeypatch):
    out = tmp_path / "run"
    p = _held(out, age=1000)
    real = Path.write_text

    def write_then_beaten(self, *a, **k):  # another taker replaces the lock between our check and our replace
        r = real(self, *a, **k)
        if self.name.endswith(".tmp"):
            _held(out, job="999")
        return r

    monkeypatch.setattr(Path, "write_text", write_then_beaten)
    with pytest.raises(tr.RunBusy, match="changed while taking it over"):
        tr.acquire_run_lock(out, 600)
    assert json.loads(p.read_text())["slurm_job_id"] == "999" and not list(out.glob("*.tmp"))


# ----------------------------------------------------------------------------- scripts/run_g0_train.sh

FAKE_PY = """#!/bin/bash
echo "$*" >> "$FAKE_CALLS"
[ "$1" = "-m" ] && exit 0
name=$(basename "$3" .json)
rc=$(cat "$FAKE_RC/$name" 2>/dev/null || echo 0)
[ "$rc" = 0 ] && mkdir -p "$CONTAM_CKPT_DIR/$name" && touch "$CONTAM_CKPT_DIR/$name/DONE"
exit "$rc"
"""


def _runner(tmp_path, seeds=None, rcs=None, done=()):
    py, specs, ckpt, rc_dir = tmp_path / "py", tmp_path / "specs", tmp_path / "ckpt", tmp_path / "rc"
    py.write_text(FAKE_PY)
    py.chmod(0o755)
    for d in (specs, ckpt, rc_dir):
        d.mkdir(exist_ok=True)
    for k in range(5):
        (specs / f"G0_d0_s{k}.json").write_text("{}")
    for k, rc in (rcs or {}).items():
        (rc_dir / f"G0_d0_s{k}").write_text(str(rc))
    for k in done:
        (ckpt / f"G0_d0_s{k}").mkdir()
        (ckpt / f"G0_d0_s{k}" / "DONE").touch()
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(CONTAM_PY=str(py), G0_SPECS=str(specs), CONTAM_CKPT_DIR=str(ckpt), FAKE_CALLS=str(tmp_path / "calls"), FAKE_RC=str(rc_dir),
               G0_NO_REQUEUE="1")
    if seeds is not None:
        env["G0_SEEDS"] = seeds
    r = subprocess.run(["bash", str(REPO_ROOT / "scripts/run_g0_train.sh")], capture_output=True, text=True, env=env, timeout=60)
    calls = [c.split()[-1].rsplit("/", 1)[-1] for c in (tmp_path / "calls").read_text().splitlines() if "train_lora.py" in c]
    return r, calls


def test_runner_skips_busy_seeds_and_runs_only_its_seeds(tmp_path):
    r, calls = _runner(tmp_path, seeds="1 2", rcs={1: 76})
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls == ["G0_d0_s1.json", "G0_d0_s2.json"]
    assert "G0_d0_s1: busy (another job is training it), skipping" in r.stdout and "G0_d0_s2: done" in r.stdout
    assert "DONE seeds: 2; not DONE: 0 1 3 4" in r.stdout and "nothing left for this job" in r.stdout and "G0 ALL DONE" not in r.stdout


def test_runner_all_done_only_when_all_five_are(tmp_path):
    r, calls = _runner(tmp_path, seeds="2 3 4", done=(0, 1))
    assert r.returncode == 0 and calls == ["G0_d0_s2.json", "G0_d0_s3.json", "G0_d0_s4.json"]
    assert "DONE seeds: 0 1 2 3 4; not DONE: none" in r.stdout and "G0 ALL DONE" in r.stdout


def test_runner_default_seeds_preemption_and_failure(tmp_path):
    r, calls = _runner(tmp_path, rcs={1: 75})
    assert r.returncode == 75 and calls == ["G0_d0_s0.json", "G0_d0_s1.json"] and "G0_d0_s1: preempted" in r.stdout
    (tmp_path / "again").mkdir()
    r, calls = _runner(tmp_path / "again", rcs={0: 3})
    assert r.returncode == 3 and calls == ["G0_d0_s0.json"] and "G0_d0_s0: FAILED (exit 3)" in r.stdout
