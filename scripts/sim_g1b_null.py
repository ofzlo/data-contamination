#!/usr/bin/env python
"""Null simulation of the G1(b) false-PASS rate, through the real gate code (validate_inputs + g1_report).

Under the null, every run (five d=0 seeds; Caption and Paired at d=4, g1_seeds_per_cell seeds each)
draws from the same distribution:

    word_count[i] = mu_i + shift_run + N(0, 4),   shift_run ~ N(0, sigma_seed)
    boilerplate[i] ~ Bernoulli(clip(p_i + shift_run / 30, 0, 1))

with per-image mu_i ~ N(60, 15) and p_i ~ U(0.2, 0.8) fixed across runs. sigma_seed is the
training-seed variance that the image bootstrap does not see. δ_S is 2 × SD of the five d=0 unseen
means (stats.style_margin). Reported false-PASS rates:

* gate:     the G1(b) verdict (Paired × word count; pooled CI_lo > δ_S AND every seed > δ_S);
* single:   the same single test with the pooled CI rule only;
* any-of-4: the earlier rule, CI_lo > δ_S for any of {Caption, Paired} × {word count, boilerplate}.

The simulation runs with no B1 analysis exclusion (exclusion=None: every simulated image analysed).

Example:
    python scripts/sim_g1b_null.py --n-images 900 --n-boot 300 --reps 300 --sigma-seed 0 0.5 1 2
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

from contam.config import load_config
from contam.data.splits import make_split
from contam.eval.gates import N_BASELINE_SEEDS, g1_report, validate_inputs
from contam.eval.stats import style_margin

PROV = {"generations": {"prompt_sha": "null-sim", "model": "null-sim", "batch_size": 16, "max_new_tokens": 300}}


def simulate(n_images: int, n_boot: int, reps: int, sigma_seed: float, seed: int) -> dict:
    rng = np.random.default_rng([seed, int(round(sigma_seed * 1000))])
    sizes = {"in-domain": n_images // 9, "near-domain": n_images * 6 // 9}
    sizes["out-domain"] = n_images - sum(sizes.values())
    ann, iid = [], 0
    for dom, n in sizes.items():
        for _ in range(n):
            ann.append({"image_id": iid, "domain": dom, "refs": [f"ref {iid} {k}" for k in range(10)]})
            iid += 1
    split = make_split(ann, 1, 2)
    ids = sorted(split["seen"] + split["unseen"])
    cfg = load_config()
    cfg["stats"]["n_boot"] = n_boot
    k_seeds = cfg["gates"]["g1_seeds_per_cell"]
    dose = cfg["gates"]["g1_b_dose"]
    mu = rng.normal(60, 15, len(ids))
    p = rng.uniform(0.2, 0.8, len(ids))

    def run(name):
        sh = rng.normal(0, sigma_seed)
        wc = mu + sh + rng.normal(0, 4, len(ids))
        bp = rng.random(len(ids)) < np.clip(p + sh / 30, 0, 1)
        per = {str(i): {"word_count": float(w), "boilerplate": float(b), "sentence_count": 1.0, "exact_match_train": 0.0} for i, w, b in zip(ids, wc, bp)}
        return {"checkpoint_id": name, "split": {"name": split["name"], "content_hash": split["content_hash"]}, "complete": True, "generations_hash": name, "manifest_split_sha256": "sim", "provenance": PROV, "per_image": per}

    counts = {"gate": 0, "single": 0, "any_of_4": 0}
    for r in range(reps):
        base_paths = [f"/null/r{r}/d0_s{s}.json" for s in range(N_BASELINE_SEEDS)]
        base = [run(f"r{r}_d0_s{s}") for s in range(N_BASELINE_SEEDS)]
        dS = {f: style_margin([np.mean([m["per_image"][str(i)][f] for i in split["unseen"]]) for m in base], cfg["margins"]["sd_multiplier"]) for f in cfg["style"]["primary_metrics"]}
        margins = {"gate_valid": True, "split": base[0]["split"], "d0_files": base_paths, "manifest_split_sha256": "sim", "base_cider_check": {"in_band": True}, "delta_S": dS}
        cks = [{"condition": c, "d": dose, "seed": s, "metrics": f"/null/r{r}/{c}_s{s}.json"} for c in ("Caption", "Paired") for s in range(k_seeds)]
        ms = [run(f"r{r}_{c['condition']}_s{c['seed']}") for c in cks]
        validate_inputs(split, margins, base_paths, base, cks, ms, k_seeds)
        b = g1_report(split, margins, base, cks, ms, cfg)["criteria"]["b"]
        prim = b["tests"][f"{cfg['gates']['g1_b_primary']['condition']}/{cfg['gates']['g1_b_primary']['metric']}"]
        counts["gate"] += b["verdict"] == "PASS"
        counts["single"] += prim["ci_lo"] > prim["delta_S"]
        counts["any_of_4"] += any(t["ci_lo"] > t["delta_S"] for t in b["tests"].values())
    return {k: v / reps for k, v in counts.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-images", type=int, default=900)
    ap.add_argument("--n-boot", type=int, default=300, help="bootstrap replicates per CI")
    ap.add_argument("--reps", type=int, default=300, help="simulated G1 experiments per sigma_seed")
    ap.add_argument("--sigma-seed", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    warnings.simplefilter("ignore")
    print(f"G1(b) null false-PASS rate: n_images={args.n_images} n_boot={args.n_boot} reps={args.reps} seed={args.seed}")
    print(f"{'sigma_seed':>10s} {'gate':>7s} {'single':>7s} {'any-of-4':>9s}")
    for s in args.sigma_seed:
        r = simulate(args.n_images, args.n_boot, args.reps, s, args.seed)
        print(f"{s:10.2f} {r['gate']:7.3f} {r['single']:7.3f} {r['any_of_4']:9.3f}", flush=True)


if __name__ == "__main__":
    main()
