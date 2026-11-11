#!/usr/bin/env python
"""Build the data bundle for the demo page (demo branch) from a main-branch checkout.

Usage: python build_demo_data.py --repo /path/to/data-contamination --out site/

Writes site/data/demo.json (sample captions + summary metrics + gate excerpt) and
site/img/<image_id>.jpg (downscaled NoCaps / Open Images photos, CC BY 2.0).
"""
import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
args = ap.parse_args()
R = args.repo
sys.path.insert(0, str(R / "src"))
from contam.data.exposure import exposure_ids  # noqa: E402

ann = json.load(open(R / "data/nocaps/annotations.json"))["images"]
A = {a["image_id"]: a for a in ann}
split = json.load(open(R / "data/splits/split_A.json"))
seen = set(exposure_ids(split, ann, 1000))
excluded = {82, 175, 418, 578, 768, 1152, 1223, 1264, 1619, 1750, 1951, 2059, 2402, 2676, 2715, 2852, 3003, 3045, 3978, 4057}
unseen = set(int(i) for i in split["unseen"]) - excluded
seen -= excluded
train_idx = {int(k): v for k, v in split["train_ref_idx"].items()}

O = R / "outputs"


def gens(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("sample_idx", 0) == 0 and r.get("temperature", 0.0) == 0.0:
            out[int(r["image_id"])] = r["caption"].strip()
    return out


def metrics(path):
    return json.load(open(path))


# Models shown per sample (seed 0 for every trained cell).
MODELS = [
    ("base", "Base LLaVA-1.5", O / "g0/base_caption.jsonl", O / "g0/base_metrics_v.json", O / "short/base_nocaps_like.jsonl"),
    ("d0", "d=0 (no exposure)", O / "g0/G0_d0_s0_caption.jsonl", O / "g0/G0_d0_s0_metrics_v_A.json", O / "short/G0_d0_s0_nocaps_like.jsonl"),
    ("P1", "Paired d=1", O / "g2/G2_Paired_d1_s0_caption.jsonl", O / "g2/G2_Paired_d1_s0_metrics_v_A.json", O / "short/G2_Paired_d1_s0_nocaps_like.jsonl"),
    ("P2", "Paired d=2", O / "ex/EX_Paired_d2_n1000_s0_caption.jsonl", O / "ex/EX_Paired_d2_n1000_s0_metrics_v_A.json", O / "short/EX_Paired_d2_n1000_s0_nocaps_like.jsonl"),
    ("P4", "Paired d=4", O / "g1/G1_Paired_d4_s0_caption.jsonl", O / "g1/G1_Paired_d4_s0_metrics_v_A.json", O / "short/G1_Paired_d4_s0_nocaps_like.jsonl"),
    ("P64", "Paired d=64", O / "g1/G1_Paired_d64_s0_caption.jsonl", O / "g1/G1_Paired_d64_s0_metrics_v_A.json", O / "short/G1_Paired_d64_s0_nocaps_like.jsonl"),
    ("S1", "SameDist d=1", O / "g2/G2_SameDist_d1_s0_caption.jsonl", O / "g2/G2_SameDist_d1_s0_metrics_v_A.json", O / "short/G2_SameDist_d1_s0_nocaps_like.jsonl"),
    ("S64", "SameDist d=64", O / "g1/G1_SameDist_d64_s0_caption.jsonl", O / "g1/G1_SameDist_d64_s0_metrics_v_A.json", O / "short/G1_SameDist_d64_s0_nocaps_like.jsonl"),
]
G = {k: gens(g) for k, _, g, _, _ in MODELS}
GS = {k: gens(s) for k, _, _, _, s in MODELS}
M = {k: metrics(m)["per_image"] for k, _, _, m, _ in MODELS}


def wc(s):
    return len(s.split())


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def em(cap, ref):
    return norm(cap) == norm(ref)


def pm(k, i, f):
    v = M[k].get(str(i), {}).get(f)
    return None if v is None else round(float(v), 3)


# --- style samples: unseen images; d0 long, Paired d1 short one-liner, no memory signal ---
def style_score(i):
    d0, p1, s1 = G["d0"].get(i), G["P1"].get(i), G["S1"].get(i)
    if not (d0 and p1 and s1):
        return None
    if wc(p1) > 16 or wc(s1) > 16 or wc(d0) < 60 or "\n" in p1:
        return None
    c = pm("P1", i, "cider_heldout") or 0
    return c - abs(wc(p1) - wc(s1))


# --- memory samples: seen images; d=4 / d=64 reproduce the training caption verbatim, d=1 does not ---
def memory_ok(i):
    ref = A[i]["refs"][train_idx[i]]
    return (
        all(k in G and i in G[k] for k in ("P1", "P2", "P4", "P64"))
        and em(G["P64"][i], ref) and em(G["P4"][i], ref) and not em(G["P1"][i], ref)
        and 6 <= wc(ref) <= 18
    )


def pick(cands, key, per_domain):
    out = []
    for dom, n in per_domain.items():
        c = sorted((i for i in cands if A[i]["domain"] == dom), key=key, reverse=True)
        out += c[:n]
    return out


style_c = [i for i in unseen if style_score(i) is not None]
style_ids = pick(style_c, style_score, {"in": 1, "near": 2, "out": 2})
mem_c = [i for i in seen if memory_ok(i)]
mem_ids = pick(mem_c, lambda i: (em(G["P2"][i], A[i]["refs"][train_idx[i]]) is False, pm("P1", i, "cider_heldout") or 0), {"in": 1, "near": 1, "out": 2})


def sample(i, group):
    a = A[i]
    t = train_idx[i]
    rec = {
        "image_id": i, "img": f"img/{i}.jpg", "domain": a["domain"], "group": group,
        "open_images_id": a["open_images_id"],
        "train_ref": a["refs"][t] if group == "seen" else None,
        "heldout_refs": [r for j, r in enumerate(a["refs"]) if j != t][:4],
        "captions": [],
    }
    for k, label, *_ in MODELS:
        c = G[k].get(i)
        if c is None:
            continue
        rec["captions"].append({
            "model": k, "label": label, "caption": c, "short_prompt": GS[k].get(i),
            "words": wc(c), "cider_heldout": pm(k, i, "cider_heldout"),
            "exact_match_train": pm(k, i, "exact_match_train"), "rmi_norm": pm(k, i, "rmi_norm"),
        })
    return rec


samples = [sample(i, "unseen") for i in style_ids] + [sample(i, "seen") for i in mem_ids]

# --- images ---
(args.out / "img").mkdir(parents=True, exist_ok=True)
for s in samples:
    im = Image.open(R / "data/nocaps/images" / A[s["image_id"]]["file_name"]).convert("RGB")
    im.thumbnail((640, 640))
    im.save(args.out / "img" / f"{s['image_id']}.jpg", quality=82, optimize=True)

# --- dose table: summary means over seeds ---
CELLS = {
    ("base", "–"): [O / "g0/base_metrics_v.json"],
    ("d=0", 0): [O / f"g0/G0_d0_s{k}_metrics_v_A.json" for k in range(5)],
}
for cond in ("Paired", "SameDist", "Caption"):
    for d, pat in ((1, "g2/G2_{c}_d1_s{s}"), (2, "ex/EX_{c}_d2_n1000_s{s}"), (4, "g1/G1_{c}_d4_s{s}"), (64, "g1/G1_{c}_d64_s{s}")):
        fs = [O / (pat.format(c=cond, s=s) + "_metrics_v_A.json") for s in (0, 1)]
        fs = [f for f in fs if f.exists()]
        if fs:
            CELLS[(cond, d)] = fs
FIELDS = ["word_count", "cider_heldout", "spice_heldout", "exact_match_train", "rmi_norm", "lsn_train"]
dose = []
for (cond, d), fs in CELLS.items():
    sums = [json.load(open(f))["summary"] for f in fs]
    row = {"condition": cond, "d": d, "seeds": len(fs)}
    for g in ("seen", "unseen"):
        row[g] = {f: round(mean(s[g][f] for s in sums), 4) for f in FIELDS}
    dose.append(row)

# --- gate excerpt ---
rep = json.load(open(O / "g1/g1_report_A_B10000.json"))
crit = rep["criteria"]


def est(x):
    return {k: round(x[k], 4) for k in ("estimate", "ci_lo", "ci_hi")}


gate = {
    "file": "outputs/g1/g1_report_A_B10000.json", "n_boot": rep["n_boot"], "gate_valid": rep["gate_valid"],
    "a": {
        "cell": crit["a"]["cell"], "verdict": crit["a"]["verdict"],
        "exact_match_change_seen": est(crit["a"]["exact_match_change_seen"]),
        "delta_rmi_seen_norm": est(crit["a"]["delta_rmi_seen"]["rmi_norm"]),
        "delta_rmi_unseen_control_norm": est(crit["a"]["delta_rmi_unseen_control"]["rmi_norm"]),
        "delta_V_norm": round(crit["a"]["delta_V"]["rmi_norm"], 4),
    },
    "b": {k: v for k, v in crit["b"].items() if k in ("dose", "primary", "verdict")},
}
gate["b"]["paired_word_count"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in crit["b"]["tests"]["Paired/word_count"].items() if k != "per_seed_estimates"}
for k in crit:
    if k not in ("a", "b") and isinstance(crit[k], dict) and "verdict" in crit[k]:
        gate[k] = {"verdict": crit[k]["verdict"]}

# --- short-prompt (format-matched) unseen CIDEr, from the formal analysis ---
short = []
for line in open(O / "explore/short_20261103/formal_nocaps_like.md"):
    m = re.match(r"\| (d=0|base \(untrained\)|Paired|Caption|SameDist) \| (\S+) \| (\S+) \| ([^|]+) \| ([\d.]+) \[[^\]]+\] \| ([^|]+) \|", line)
    if m and m.group(3) in ("–", "1000"):
        short.append({"condition": m.group(1), "d": m.group(2), "cider_heldout": float(m.group(5)), "delta": m.group(6).strip()})
    if line.startswith("## 2."):
        break

out = {
    "split": "A", "n_exp": 1000, "prompt": "Describe the image in detail.", "short_prompt": "Describe the image in one short sentence.",
    "models": [{"key": k, "label": l} for k, l, *_ in MODELS],
    "samples": samples, "dose": dose, "gate": gate, "short": short,
}
(args.out / "data").mkdir(parents=True, exist_ok=True)
json.dump(out, open(args.out / "data/demo.json", "w"), indent=1, ensure_ascii=False)
print("style", style_ids, "memory", mem_ids, "cands", len(style_c), len(mem_c))
