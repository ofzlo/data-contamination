import random

import numpy as np
import pytest

from contam.config import load_config
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


@pytest.fixture
def annotations():
    return make_annotations()


def _fake_metrics(split, name, wc_by_id, complete=True):
    per = {str(i): {"word_count": float(w), "boilerplate": 0.0, "sentence_count": 1.0, "exact_match_train": 0.0} for i, w in wc_by_id.items()}
    return {"checkpoint_id": name, "split": {"name": split["name"], "content_hash": split["content_hash"]}, "complete": complete, "generations_hash": name, "provenance": {"generations": {"prompt_sha": "P", "model": "m", "batch_size": 16, "max_new_tokens": 300}}, "per_image": per}


@pytest.fixture
def g1_setup():
    ann = make_annotations(10, 40, 20, seed=9)
    split = make_split(ann, 1, 2)
    ids = sorted(split["seen"] + split["unseen"])
    rng = np.random.default_rng(0)
    base_wc = {i: 30 + rng.normal(0, 3) for i in ids}
    baseline = [_fake_metrics(split, f"d0_s{s}", {i: w + rng.normal(0, 0.1) for i, w in base_wc.items()}) for s in range(5)]
    margins = {"gate_valid": True, "split": {"name": split["name"], "content_hash": split["content_hash"]}, "d0_files": [f"/tmp/b{s}.json" for s in range(5)], "delta_S": {"word_count": 1.0, "boilerplate": 0.01}}
    cfg = load_config()
    cfg["stats"]["n_boot"] = 200

    def ckpt(name, drop):  # word count drops by `drop` (positive directional effect)
        return _fake_metrics(split, name, {i: w - drop + rng.normal(0, 0.1) for i, w in base_wc.items()})

    return split, baseline, margins, cfg, ckpt
