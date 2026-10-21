"""Configuration loading and small I/O helpers.

Configs are YAML. String values may contain ``${VAR}`` or ``${VAR:-default}``, expanded from the
environment, so cluster paths (e.g. /scratch) never need to be hardcoded in library code.
Relative paths under ``paths`` are resolved against the repository root (the parent of ``configs/``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_PROMPTS = REPO_ROOT / "configs" / "prompts.yaml"

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: str) -> str:
    """Expand ``${VAR}`` / ``${VAR:-default}``; an unset variable without default is an error."""

    def repl(m: re.Match) -> str:
        name, default = m.group(1), m.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is None:
            raise KeyError(f"environment variable {name} is not set and has no default")
        return default

    return _ENV_RE.sub(repl, value)


def _expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return expand_env(obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def load_yaml(path: str | os.PathLike) -> dict:
    with open(path) as f:
        return _expand(yaml.safe_load(f) or {})


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load the main config; ``paths`` entries become absolute ``Path`` objects."""
    cfg = load_yaml(path or DEFAULT_CONFIG)
    cfg["paths"] = {k: resolve_path(v) for k, v in cfg.get("paths", {}).items()}
    return cfg


def load_prompts(path: str | os.PathLike | None = None) -> dict[str, str]:
    """Return ``{prompt_id: text}`` from the prompts YAML."""
    raw = load_yaml(path or DEFAULT_PROMPTS)["prompts"]
    return {k: (v["text"] if isinstance(v, dict) else v) for k, v in raw.items()}


def resolve_path(p: str | os.PathLike) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def read_json(path: str | os.PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(obj: Any, path: str | os.PathLike, indent: int | None = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def read_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(rows: Iterable[dict], path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())


def content_hash(obj: Any) -> str:
    """sha256 of the canonical (sorted-key, compact) JSON serialization of ``obj``."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def prompt_sha(prompt: str) -> str:
    """sha256 of the exact prompt text (stored in every generation / log-prob row)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
