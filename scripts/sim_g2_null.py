#!/usr/bin/env python
"""False-PASS rate and power of the G2 primary (equivalence) rules, through the real gate code
(validate_g2_inputs + g2_report), under training-seed variance.

Every run (five d=0 seeds; Caption and Paired at d ∈ {1, 4, 16, 64} × 3 seeds; SameDist d=16 × 3)
draws per-image values

    word_count[i] = wc_i − style(d) + N(0, 2)
    rmi_sum[i]    = r_i + memory(d) + shift_run + N(0, 2),   shift_run ~ N(0, sigma_seed)

with wc_i ~ N(30, 3), r_i ~ N(0, 3) fixed across runs. sigma_seed is the training-seed variance
that the image bootstrap does not see; it shifts every run, d=0 seeds included. Style crosses δ_S
(= 1) at d=4 (style = 0, 5, 6, 7 for Paired), so d_style = 4; the Paired V effect at d=4 is
``mult × δ_V`` (δ_V = 0.5). With mult ≥ 1 a PASS is false (the V effect is not below δ_V); with
mult < 1 the PASS rate is power. One report gives the verdict of every rule:

* pooled:                pooled CI upper bound < δ_V (image bootstrap only);
* pooled_and_every_seed: that AND every seed's estimate < δ_V (as in G1(b));
* two_level_bootstrap:   CI upper bound with seeds and then images resampled (the default rule);
* seed_t:                t upper bound over the per-seed estimates (df = 2).

Example (4 processes):
    python scripts/sim_g2_null.py --reps 150 --n-boot 300 --n-images 900 \
        --sigma-seed 0 0.1 0.2 0.4 --mult 1.0 1.2 0 0.5 --seed 0 --jobs 4
"""

from __future__ import annotations

import argparse
import itertools
import time
import warnings
from multiprocessing import Pool

import numpy as np

from contam.config import load_config
from contam.data.splits import make_split
from contam.eval.gates import G2_EQUIVALENCE_RULES, N_BASELINE_SEEDS, g2_report, validate_g2_inputs

PROV = {"generations": {"prompt_sha": "null-sim", "model": "null-sim", "batch_size": 16, "max_new_tokens": 300}}
DELTA_S, DELTA_V = 1.0, 0.5
STYLE = {1: 0.0, 4: 5.0, 16: 6.0, 64: 7.0}
MEMORY = {1: 0.0, 16: 2.0, 64: 5.0}  # d=4 is set by mult


def simulate(n_images: int, n_boot: int, reps: int, sigma_seed: float, mult: float, seed: int) -> dict:
    warnings.simplefilter("ignore")
    rng = np.random.default_rng([seed, int(round(sigma_seed * 1000)), int(round(mult * 1000))])
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
    k_seeds = cfg["gates"]["g2_seeds_per_cell"]
    memory = {**MEMORY, 4: mult * DELTA_V}

    counts = dict.fromkeys(G2_EQUIVALENCE_RULES, 0)
    counts["d_style_4"] = 0
    for r in range(reps):
        wc0, rmi0 = 30 + rng.normal(0, 3, len(ids)), rng.normal(0, 3, len(ids))

        def run(name, s_eff, m_eff):
            wc = wc0 - s_eff + rng.normal(0, 2, len(ids))
            rmi = rmi0 + m_eff + rng.normal(0, sigma_seed) + rng.normal(0, 2, len(ids))
            per = {str(i): {"word_count": float(a), "rmi_sum": float(b)} for i, a, b in zip(ids, wc, rmi)}
            return {"checkpoint_id": name, "split": {"name": split["name"], "content_hash": split["content_hash"]}, "complete": True, "generations_hash": name,
                    "manifest_split_sha256": "sim", "provenance": PROV, "per_image": per}

        base_paths = [f"/null/r{r}/d0_s{s}.json" for s in range(N_BASELINE_SEEDS)]
        base = [run(f"r{r}_d0_s{s}", 0.0, 0.0) for s in range(N_BASELINE_SEEDS)]
        cks, cms = [], []
        for cond, sf, mf in (("Caption", 0.5, 0.0), ("Paired", 1.0, 1.0)):
            for d in STYLE:
                for s in range(k_seeds):
                    cks.append({"condition": cond, "d": d, "seed": s, "metrics": f"/null/r{r}/{cond}_{d}_{s}.json"})
                    cms.append(run(f"r{r}_{cond}_{d}_{s}", sf * STYLE[d], mf * memory[d]))
        for s in range(k_seeds):
            cks.append({"condition": "SameDist", "d": 16, "seed": s, "metrics": f"/null/r{r}/SameDist_16_{s}.json"})
            cms.append(run(f"r{r}_SameDist_16_{s}", 3.0, 0.0))
        margins_path = f"/null/r{r}/margins.json"
        margins = {"gate_valid": True, "split": base[0]["split"], "d0_files": base_paths, "manifest_split_sha256": "sim", "delta_S": {"word_count": DELTA_S}}
        g1 = {"split": split["name"], "split_hash": split["content_hash"], "gate_valid": True, "manifest_split_sha256": "sim", "baseline_files": base_paths,
              "margins_file": margins_path, "criteria": {"a": {"verdict": "PASS", "delta_V": {"rmi_sum": DELTA_V}}}}
        validate_g2_inputs(split, margins, margins_path, g1, base_paths, base, cks, cms, cfg)
        prim = g2_report(split, margins, g1, base, cks, cms, cfg)["criteria"]["primary"]
        if prim.get("d_style") == 4:
            counts["d_style_4"] += 1
            for k in G2_EQUIVALENCE_RULES:
                counts[k] += prim["rules"][k]["pass"]
    return {k: v / reps for k, v in counts.items()}


def _job(a):
    n_images, n_boot, reps, sigma, mult, seed = a
    return sigma, mult, simulate(n_images, n_boot, reps, sigma, mult, seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-images", type=int, default=900)
    ap.add_argument("--n-boot", type=int, default=300, help="bootstrap replicates per CI")
    ap.add_argument("--reps", type=int, default=150, help="simulated G2 experiments per (sigma_seed, mult)")
    ap.add_argument("--sigma-seed", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4])
    ap.add_argument("--mult", type=float, nargs="+", default=[1.0, 1.2, 0.0, 0.5], help="true Paired V effect at d_style, in units of delta_V")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1, help="worker processes")
    args = ap.parse_args()

    t0 = time.time()
    grid = [(args.n_images, args.n_boot, args.reps, s, m, args.seed) for m, s in itertools.product(args.mult, args.sigma_seed)]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            res = pool.map(_job, grid)
    else:
        res = [_job(a) for a in grid]
    print(f"G2 primary PASS rate (mult >= 1: false PASS; mult < 1: power): n_images={args.n_images} n_boot={args.n_boot} reps={args.reps} seed={args.seed}")
    print(f"{'mult':>5s} {'sigma_seed':>10s} " + " ".join(f"{k:>22s}" for k in G2_EQUIVALENCE_RULES) + f" {'d_style=4':>9s}")
    for sigma, mult, r in res:
        print(f"{mult:5.2f} {sigma:10.2f} " + " ".join(f"{r[k]:22.3f}" for k in G2_EQUIVALENCE_RULES) + f" {r['d_style_4']:9.3f}")
    print(f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
