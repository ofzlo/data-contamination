import random

import numpy as np
import pytest
import yaml

from contam.config import DEFAULT_CONFIG, file_sha256, load_config, write_json
from contam.data import freeze
from contam.data.splits import make_split


def make_annotations(n_in=41, n_near=267, n_out=141, n_refs=10, seed=0):
    rng = random.Random(seed)
    words = "dog cat man woman car tree red blue small large sitting standing on in a the with near field street".split()
    ann = []
    iid = 0
    for dom, n in (("in-domain", n_in), ("near-domain", n_near), ("out-domain", n_out)):
        for _ in range(n):
            refs = [" ".join(rng.choice(words) for _ in range(rng.randint(5, 12))) for _ in range(n_refs)]
            ann.append({"image_id": iid, "file_name": f"{iid}.jpg", "domain": dom, "refs": refs})
            iid += 1
    rng.shuffle(ann)
    return ann


def locked_split(tmp_path, split, n_boot=None):
    """Write ``split`` to <tmp>/splits/split_<name>.json, a config pointing there and a lock freezing it.

    Returns (config path, lock path, split path, split sha256) for the CLIs' --config / --lock / --split.
    """
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["paths"]["splits_dir"] = str(tmp_path / "splits")
    if n_boot:
        raw["stats"]["n_boot"] = n_boot
    cfg_path, lock = tmp_path / "config.yaml", tmp_path / "lock.json"
    cfg_path.write_text(yaml.safe_dump(raw))
    split_path = tmp_path / "splits" / f"split_{split['name']}.json"
    write_json(split, split_path)
    freeze.write_lock(load_config(cfg_path), lock)
    return cfg_path, lock, split_path, file_sha256(split_path)


@pytest.fixture
def annotations():
    return make_annotations()


def _fake_metrics(split, name, wc_by_id, complete=True):
    per = {str(i): {"word_count": float(w), "boilerplate": 0.0, "sentence_count": 1.0, "exact_match_train": 0.0} for i, w in wc_by_id.items()}
    return {"checkpoint_id": name, "split": {"name": split["name"], "content_hash": split["content_hash"]}, "complete": complete, "generations_hash": name, "manifest_split_sha256": "L", "provenance": {"generations": {"prompt_sha": "P", "model": "m", "batch_size": 16, "max_new_tokens": 300}}, "per_image": per}


@pytest.fixture
def g1_setup():
    ann = make_annotations(10, 40, 20, seed=9)
    split = make_split(ann, 1, 2)
    ids = sorted(split["seen"] + split["unseen"])
    rng = np.random.default_rng(0)
    base_wc = {i: 30 + rng.normal(0, 3) for i in ids}
    baseline = [_fake_metrics(split, f"d0_s{s}", {i: w + rng.normal(0, 0.1) for i, w in base_wc.items()}) for s in range(5)]
    margins = {"gate_valid": True, "split": {"name": split["name"], "content_hash": split["content_hash"]}, "d0_files": [f"/tmp/b{s}.json" for s in range(5)], "manifest_split_sha256": "L", "delta_S": {"word_count": 1.0, "boilerplate": 0.01}}
    cfg = load_config()
    cfg["stats"]["n_boot"] = 200

    def ckpt(name, drop):  # word count drops by `drop` (positive directional effect)
        return _fake_metrics(split, name, {i: w - drop + rng.normal(0, 0.1) for i, w in base_wc.items()})

    return split, baseline, margins, cfg, ckpt
