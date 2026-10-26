#!/usr/bin/env python
"""Pool the D2 timing repeats (scripts/timing_run.py --repeat r) into one table per timed cell.

Reads every ``timing_report_*.json`` under --dir, groups them by cell (general only, or the exposure
condition / N_exp / d / M of the sample) and GPU, and reports per cell: the repeats, steady-state
examples/s and s/step (mean and SD across repeats), and GPU-h per run at M (from the pooled
examples/s, with the checkpoint and fixed overheads of timing_run's projection). Writes a Markdown
table (--out) to paste into docs/results/ and the D2 row of prereg.md.

Example:
    python scripts/d2_summary.py --dir outputs/timing --M 130000 --out docs/results/d2_timing.md
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from pathlib import Path

from contam.config import load_config, read_json


def cell_of(rep: dict) -> str:
    e = rep["sample"].get("exposure")
    return "general only (d=0)" if not e else f"{e['condition']} d={e['d']} (N_exp {e['n_exp']}, share {e['share']:.3f} of M {e['M']:,})"


def summarize(reports: list[dict], M: int, gbs: int, every: int) -> list[dict]:
    groups = defaultdict(list)
    for r in reports:
        groups[(cell_of(r), r["gpu"])].append(r)
    rows = []
    for (cell, gpu), reps in sorted(groups.items()):
        reps.sort(key=lambda r: r["args"].get("repeat", 0))
        eps = [r["timing"]["examples_per_s"] for r in reps]
        sps = [r["timing"]["step_time_s_mean"] for r in reps]
        mean_eps = statistics.mean(eps)
        save = max(r["timing"]["checkpoint_save_s"] for r in reps)
        over = statistics.mean(r["timing"]["overhead_s"] for r in reps)
        steps = math.ceil(M / gbs)
        run_h = M / mean_eps / 3600 + (steps // every) * (0 if math.isnan(save) else save) / 3600 + over / 3600
        rows.append({
            "cell": cell, "gpu": gpu, "repeats": [r["args"].get("repeat", 0) for r in reps],
            "steps_measured": [r["timing"]["steps_measured"] for r in reps],
            "micro_batch": sorted({r["timing"]["micro_batch_size"] for r in reps}),
            "examples_per_s": (round(mean_eps, 3), round(statistics.stdev(eps), 3) if len(eps) > 1 else None),
            "step_s": (round(statistics.mean(sps), 2), round(statistics.stdev(sps), 2) if len(sps) > 1 else None),
            "tokens_per_example": round(statistics.mean(r["timing"]["tokens_per_example"] for r in reps)),
            "peak_mem_gib": round(max(r["timing"]["peak_mem_gib"] for r in reps), 1),
            "gpu_h_per_run": round(run_h, 2), "M": M, "optimizer_steps": steps,
        })
    return rows


def markdown(rows: list[dict]) -> str:
    out = ["| Cell | GPU | Repeats (steady steps) | Micro | Examples/s (mean ± SD) | s/step (mean ± SD) | Tokens/example | Peak GiB | GPU-h/run |",
           "|---|---|---|---|---|---|---|---|---|"]
    pm = lambda v: f"{v[0]}" + (f" ± {v[1]}" if v[1] is not None else "")
    for r in rows:
        reps = ", ".join(f"r{k} ({n})" for k, n in zip(r["repeats"], r["steps_measured"]))
        out.append(f"| {r['cell']} | {r['gpu']} | {reps} | {'/'.join(map(str, r['micro_batch']))} | {pm(r['examples_per_s'])} | {pm(r['step_s'])} | "
                   f"{r['tokens_per_example']} | {r['peak_mem_gib']} | {r['gpu_h_per_run']} (M={r['M']:,}, {r['optimizer_steps']} steps) |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--dir", default=None, help="timing_run output dir (default <outputs_dir>/timing)")
    ap.add_argument("--M", type=int, default=130000)
    ap.add_argument("--out", default=None, help="write the Markdown table here")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = Path(args.dir or Path(cfg["paths"]["outputs_dir"]) / "timing")
    reports = [read_json(p) for p in sorted(d.glob("timing_report_*.json"))]
    if not reports:
        raise SystemExit(f"no timing_report_*.json in {d}")
    rows = summarize(reports, args.M, int(cfg["training"]["global_batch_size"]), int(cfg["training"]["checkpoint_every_steps"]))
    md = markdown(rows)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)


if __name__ == "__main__":
    main()
