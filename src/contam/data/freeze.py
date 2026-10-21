"""Freeze the data manifests before G0 (D7), and check them before every training or eval run.

The lock (``manifests/MANIFEST.lock.json``, tracked in git) records, for every frozen file in the
splits directory, its sha256 and (for split / SameDist files) its content hash, plus the seed
config that produced them. After freezing, any change to a frozen file or to those seeds makes
``check`` fail, so a run cannot silently use different data.

Frozen files: ``split_*.json``, ``samedist_*.json`` and ``general_order.jsonl``. A file can be
added to an existing lock later (``--add``, e.g. ``general_order.jsonl`` once the B1 audit
exclusions are known), but an entry already in the lock is never rewritten; unfreezing means
deleting the lock, which needs team sign-off.

    python -m contam.data.freeze            # write the lock (refuses if it exists)
    python -m contam.data.freeze --add      # add newly created manifests to the lock
    python -m contam.data.freeze --check    # verify; exit 1 on any mismatch
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from contam.config import REPO_ROOT, file_sha256, load_config, read_json, write_json

LOCK_FORMAT_VERSION = 1
DEFAULT_LOCK = REPO_ROOT / "manifests" / "MANIFEST.lock.json"
FROZEN_GLOBS = ("split_*.json", "samedist_*.json", "general_order.jsonl")


def frozen_files(splits_dir) -> list[Path]:
    d = Path(splits_dir)
    return sorted({p for g in FROZEN_GLOBS for p in d.glob(g) if not p.name.endswith(".tmp")})


def _entry(p: Path) -> dict:
    e = {"sha256": file_sha256(p)}
    if p.suffix == ".json":
        obj = read_json(p)
        e["content_hash"] = obj.get("content_hash")
        if "annotations_hash" in obj:
            e["annotations_hash"] = obj["annotations_hash"]
    return e


def _seeds(cfg: dict) -> dict:
    t = cfg.get("training", {})
    return {"splits": cfg.get("splits"), "samedist": cfg.get("samedist"), "training": {"seeds": t.get("seeds"), "general_seed": t.get("general_seed")}}


def write_lock(cfg: dict, lock_path=DEFAULT_LOCK, add: bool = False) -> dict:
    lock_path = Path(lock_path)
    files = frozen_files(cfg["paths"]["splits_dir"])
    if add:
        lock = read_json(lock_path)
        verify(cfg, lock_path, allow_unlocked=True)
        new = [p for p in files if p.name not in lock["files"]]
        if not new:
            raise SystemExit("nothing new to add")
        for p in new:
            lock["files"][p.name] = {**_entry(p), "added_at": _now()}
    else:
        if lock_path.exists():
            raise SystemExit(f"{lock_path} exists; manifests are frozen. Delete it only with team sign-off.")
        if not any(p.name.startswith("split_") for p in files):
            raise SystemExit(f"no split_*.json in {cfg['paths']['splits_dir']}")
        lock = {"format_version": LOCK_FORMAT_VERSION, "frozen_at": _now(), "seeds": _seeds(cfg), "files": {p.name: _entry(p) for p in files}}
    write_json(lock, lock_path, indent=1)
    return lock


def verify(cfg: dict, lock_path=DEFAULT_LOCK, allow_unlocked: bool = False, required: Sequence[str] = ()) -> dict:
    """Raise RuntimeError unless every locked file and the seed config match the lock.

    ``required`` names files that must be in the lock (e.g. ``general_order.jsonl`` for training).
    Unless ``allow_unlocked``, frozen-pattern files that are not in the lock are also an error.
    """
    lock = read_json(lock_path)
    d = Path(cfg["paths"]["splits_dir"])
    problems = []
    for name, e in lock["files"].items():
        p = d / name
        if not p.exists():
            problems.append(f"missing {name}")
        elif file_sha256(p) != e["sha256"]:
            problems.append(f"changed {name}")
    if lock["seeds"] != _seeds(cfg):
        problems.append("seed config (splits / samedist / training) differs from the lock")
    for name in required:
        if name not in lock["files"]:
            problems.append(f"{name} is not frozen")
    if not allow_unlocked:
        extra = sorted(p.name for p in frozen_files(d) if p.name not in lock["files"])
        if extra:
            problems.append(f"not in lock: {extra}")
    if problems:
        raise RuntimeError(f"manifest lock {lock_path} mismatch: " + "; ".join(problems))
    return lock


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Freeze or check the split / SameDist / general-order manifests.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--lock", default=str(DEFAULT_LOCK))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--add", action="store_true", help="add manifests not yet in the lock")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.check:
        try:
            lock = verify(cfg, args.lock)
        except (RuntimeError, FileNotFoundError) as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        print(f"ok: {len(lock['files'])} frozen files match {args.lock}")
    else:
        print(json.dumps(write_lock(cfg, args.lock, add=args.add), indent=1))


if __name__ == "__main__":
    main()
