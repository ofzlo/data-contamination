#!/usr/bin/env python
"""B6 detectors (CDD, DCQ, KDS) for one checkpoint, and the across-checkpoint report.

Definitions and deviations from the papers: ``contam.eval.detectors``. Settings: ``detectors`` in
configs/default.yaml. Every phase writes one JSON with per-image (or per-set) scores, the seen and
unseen means, their difference and the seen-vs-unseen AUROC with bootstrap CIs, plus provenance.
The B1 analysis exclusion (``analysis.exclude_image_ids_file``) is applied unless
``--keep-excluded`` is set.

Phases:

* ``cdd`` (CPU): from the checkpoint's T=0 generations and its T=1 samples (``contam.generate
  --temperature 1``, 20 per image). All split images.
* ``dcq`` (GPU, ~10 min): needs perturbations from ``scripts/dcq_perturb.py`` for the DCQ image subset.
* ``kds`` (GPU, ~1 h): embeds the KDS subsets, fine-tunes a fresh LoRA on each group and embeds again.
* ``report``: B6 table over several result files (e.g. base and the five d=0 seeds): each
  detector's seen-vs-unseen AUROC and difference per checkpoint, and the mean and SD across the
  d=0 seeds (the false-positive spread C9 compares exposed checkpoints against).

Examples:
    python scripts/run_detectors.py cdd --split data/splits/split_A.json --checkpoint-id base \\
        --greedy outputs/g0/base_caption.jsonl --samples outputs/gens/base_t1.jsonl --out outputs/detectors/cdd_base.json
    python scripts/run_detectors.py dcq --split data/splits/split_A.json --checkpoint-id G0_d0_s0 --adapter CKPT \\
        --perturbations outputs/detectors/dcq_perturbations_A.jsonl --out outputs/detectors/dcq_G0_d0_s0.json
    python scripts/run_detectors.py kds --split data/splits/split_A.json --checkpoint-id base --out outputs/detectors/kds_base.json
    python scripts/run_detectors.py report --inputs outputs/detectors/*.json --d0-prefix G0_d0 --out outputs/detectors/b6_report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from contam.config import file_sha256, load_config, load_prompts, prompt_sha, read_json, read_jsonl, write_json
from contam.data.freeze import add_check_args, cli_check
from contam.eval import detectors as det


def load_vlm(args, cfg):
    from contam.models import HFVLM

    return HFVLM(args.model or cfg["models"]["main"], adapter=args.adapter, dtype=cfg["models"]["dtype"], end_token=cfg["models"]["end_token"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["cdd", "dcq", "kds", "report"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--checkpoint-id", default=None)
    ap.add_argument("--model", default=None, help="HF base model id (default: config models.main)")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (none = untrained base)")
    ap.add_argument("--greedy", default=None, help="cdd: T=0 generations JSONL")
    ap.add_argument("--samples", default=None, help="cdd: T>0 samples JSONL")
    ap.add_argument("--prompt-id", default="caption")
    ap.add_argument("--perturbations", default=None, help="dcq: scripts/dcq_perturb.py output")
    ap.add_argument("--max-images", type=int, default=None, help="smoke tests: N images per group; output is not gate-valid")
    ap.add_argument("--keep-excluded", action="store_true", help="robustness check: keep the B1 exclusion images")
    ap.add_argument("--inputs", nargs="*", default=[], help="report: detector result files")
    ap.add_argument("--d0-prefix", default="G0_d0", help="report: checkpoint ids of the d=0 seeds start with this")
    ap.add_argument("--out", required=True)
    add_check_args(ap)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.phase == "report":
        write_json(report(args.inputs, args.d0_prefix), args.out)
        print(f"-> {args.out}")
        return
    if not args.split or not args.checkpoint_id:
        ap.error(f"{args.phase} needs --split and --checkpoint-id")
    frozen = cli_check(args, cfg, args.split, f"run_detectors {args.phase}")

    from contam.data.nocaps import ImageLoader, load_annotations, refs_by_id
    from contam.data.splits import check_annotations

    dcfg = cfg["detectors"][args.phase]
    st = cfg["stats"]
    ann = load_annotations(cfg["paths"]["annotations"])
    split = read_json(args.split)
    check_annotations(split, ann)
    groups, excluded = det.detector_groups(cfg, split, args.phase, args.keep_excluded)
    if args.max_images:
        groups = {g: v[: args.max_images] for g, v in groups.items()}
    prompts = load_prompts(args.prompts)
    out = {
        "detector": args.phase, "checkpoint_id": args.checkpoint_id, "model": args.model or cfg["models"]["main"], "adapter": args.adapter,
        "split": split.get("name"), "settings": dcfg, "excluded_image_ids": excluded, **frozen,
        "gate_valid": frozen["manifest_split_sha256"] is not None and not args.max_images and not args.keep_excluded,
        "groups": groups,
    }
    kw = dict(n_boot=st["n_boot"], seed=st["boot_seed"], ci=st["ci"])

    if args.phase == "cdd":
        if not args.greedy or not args.samples:
            ap.error("cdd needs --greedy and --samples")
        ids = groups["seen"] + groups["unseen"]
        want = set(ids)
        psha = prompt_sha(prompts[args.prompt_id])
        rows = [r for r in read_jsonl(args.greedy) if r["prompt_id"] == args.prompt_id and int(r["image_id"]) in want]
        samp = [r for r in read_jsonl(args.samples) if r["prompt_id"] == args.prompt_id and int(r["image_id"]) in want]
        for r in rows + samp:
            if r.get("checkpoint_id") != args.checkpoint_id or r.get("prompt_sha") != psha:
                raise SystemExit(f"cdd: row for image {r['image_id']} is from {r.get('checkpoint_id')!r} / prompt_sha {str(r.get('prompt_sha'))[:12]}, "
                                 f"expected {args.checkpoint_id!r} / {psha[:12]}")
        peaks = det.cdd_scores(rows, samp, ids, alpha=dcfg["alpha"], max_len=dcfg["max_len"])
        out["n_samples_per_image"] = int(np.median([sum(1 for r in samp if int(r["image_id"]) == i) for i in ids[:50]]))
        out["per_image"] = peaks
        out["summary"] = det.group_summary(peaks, groups, **kw)
        out["summary"]["leaked_rate"] = {g: float(np.mean([peaks[i] > dcfg["xi"] for i in groups[g]])) for g in ("seen", "unseen")}

    elif args.phase == "dcq":
        if not args.perturbations:
            ap.error("dcq needs --perturbations")
        refs = refs_by_id(ann)
        pert = list(read_jsonl(args.perturbations))
        template = prompts["dcq_quiz"]
        vlm = load_vlm(args, cfg)
        loader = ImageLoader(ann, cfg["paths"]["images_dir"])
        out["prompt_sha"] = prompt_sha(template)
        out["perturbations_file"] = str(args.perturbations)
        out["perturbations_sha256"] = file_sha256(args.perturbations)
        out["per_group"], per_image = {}, {}
        for g in ("seen", "unseen"):
            items = det.dcq_items(pert, split, refs, groups[g])
            answers = det.run_dcq(vlm, loader, items, template, max_new_tokens=dcfg["max_new_tokens"])
            bdq = {i: a["letter"] for i, a in answers["bdq"].items()}
            bcq = {p: {i: a["letter"] for i, a in answers[f"bcq_{det.LETTERS[p]}"].items()} for p in range(4)}
            est = det.dcq_estimate(bdq, bcq)
            per_image.update(est.pop("per_image"))
            out["per_group"][g] = {**est, "answers": answers}
        out["per_image"] = per_image
        out["summary"] = det.group_summary(per_image, groups, **kw)

    else:  # kds
        refs = refs_by_id(ann)
        loader = ImageLoader(ann, cfg["paths"]["images_dir"])
        prompt = prompts[args.prompt_id]
        out["prompt_sha"] = prompt_sha(prompt)
        vlm = load_vlm(args, cfg)
        out["per_group"] = {}
        for g in ("seen", "unseen"):
            ids = groups[g]
            train = [refs[i][int(split["train_ref_idx"][str(i)])] for i in ids]
            Z = embed_all(vlm, loader, ids, prompt, train, dcfg["embed_batch_size"])
            examples = [(i, prompt, r) for i, r in zip(ids, train)]
            pm = det.kds_finetune(vlm, loader, examples, epochs=dcfg["epochs"], lr=dcfg["lr"], batch_size=dcfg["batch_size"], lora_r=dcfg["lora_r"],
                                  lora_alpha=dcfg["lora_alpha"], lora_dropout=dcfg["lora_dropout"], target_modules=dcfg["target_modules"],
                                  seed=dcfg["seed"], max_len=cfg["training"]["max_len"])
            vlm.model = pm
            try:
                Z_ft = embed_all(vlm, loader, ids, prompt, train, dcfg["embed_batch_size"])
            finally:
                vlm.model = pm.unload()  # back to the checkpoint for the next group
            out["per_group"][g] = det.kernel_divergence_score(Z, Z_ft)
            # KDS of the group is one number; the per-sample share of the divergence is kept for C9
            out["per_group"][g]["per_image_shift"] = {int(i): float(v) for i, v in zip(ids, np.linalg.norm(det.l2_normalize(Z_ft) - det.l2_normalize(Z), axis=1))}
        out["summary"] = {"kds_seen": out["per_group"]["seen"]["kds"], "kds_unseen": out["per_group"]["unseen"]["kds"],
                          "seen_minus_unseen": out["per_group"]["seen"]["kds"] - out["per_group"]["unseen"]["kds"]}
        shifts = {**out["per_group"]["seen"]["per_image_shift"], **out["per_group"]["unseen"]["per_image_shift"]}
        # smaller shift = more "seen-like"; negate so AUROC > 0.5 means seen images move less
        out["summary"]["shift"] = det.group_summary({i: -v for i, v in shifts.items()}, groups, **kw)

    write_json(out, args.out)
    s = out["summary"]
    if "auroc" in s:
        a = s["auroc"]
        print(f"{args.phase} {args.checkpoint_id}: seen {s['seen']['mean']:.4f} unseen {s['unseen']['mean']:.4f} AUROC {a['estimate']:.3f} [{a['ci_lo']:.3f}, {a['ci_hi']:.3f}]")
    else:
        print(f"{args.phase} {args.checkpoint_id}: KDS seen {s['kds_seen']:.4f} unseen {s['kds_unseen']:.4f}")
    print(f"-> {args.out}")


def embed_all(vlm, loader, ids, prompt, refs, batch_size: int) -> np.ndarray:
    out = []
    for s in range(0, len(ids), batch_size):
        out.append(vlm.embed([loader(i) for i in ids[s : s + batch_size]], prompt, refs[s : s + batch_size]))
    return np.concatenate(out)


def _headline(res: dict) -> dict:
    s = res["summary"]
    if res["detector"] == "kds":
        return {"seen": s["kds_seen"], "unseen": s["kds_unseen"], "seen_minus_unseen": s["seen_minus_unseen"], "auroc": s["shift"]["auroc"]["estimate"]}
    return {"seen": s["seen"]["mean"], "unseen": s["unseen"]["mean"], "seen_minus_unseen": s["seen_minus_unseen"]["estimate"], "auroc": s["auroc"]["estimate"],
            "auroc_ci": [s["auroc"]["ci_lo"], s["auroc"]["ci_hi"]]}


def report(paths, d0_prefix: str) -> dict:
    """Per detector: every checkpoint's headline numbers, and mean / SD (ddof 1) across the d=0 seeds."""
    rows: dict[str, dict[str, dict]] = {}
    valid = True
    shas = set()
    for p in paths:
        r = read_json(p)
        if "detector" not in r:
            continue
        rows.setdefault(r["detector"], {})[r["checkpoint_id"]] = _headline(r)
        valid &= bool(r.get("gate_valid"))
        shas.add(r.get("manifest_split_sha256"))
    if len(shas) > 1:
        print(f"WARNING: inputs come from {len(shas)} different split lock entries", file=sys.stderr)
        valid = False
    out = {"gate_valid": valid, "manifest_split_sha256": sorted(shas, key=str), "detectors": {}}
    for d, by_ck in rows.items():
        d0 = [v for k, v in by_ck.items() if k.startswith(d0_prefix)]
        spread = {}
        if len(d0) >= 2:
            for f in ("auroc", "seen_minus_unseen"):
                x = np.array([v[f] for v in d0], float)
                spread[f] = {"n_seeds": len(x), "mean": float(x.mean()), "sd": float(x.std(ddof=1))}
        out["detectors"][d] = {"checkpoints": by_ck, "d0_spread": spread}
        print(f"== {d}")
        for k, v in sorted(by_ck.items()):
            print(f"  {k:24s} seen {v['seen']:9.4f} unseen {v['unseen']:9.4f} diff {v['seen_minus_unseen']:+9.4f} AUROC {v['auroc']:.3f}")
        for f, v in spread.items():
            print(f"  d=0 spread {f}: mean {v['mean']:+.4f} SD {v['sd']:.4f} (n={v['n_seeds']})")
    return out


if __name__ == "__main__":
    main()
