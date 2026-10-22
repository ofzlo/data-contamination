# Style Before Memory

**When Benchmark Exposure Starts to Inflate VLM Scores**

A controlled study of benchmark contamination in vision-language models (VLMs). We fine-tune VLMs on NoCaps evaluation items at controlled exposure levels and ask whether the resulting score gains come from *style adaptation* (writing in the benchmark's caption style) or *reference memorization* (reproducing the specific training caption), and whether existing contamination detectors can tell the two apart.

> Status: planning / pilot (W1, week of Oct 19, 2026). Target venue: ICML 2027. Started as a CSE 576 course project.

## Hypothesis

Benchmark captions share a short, single-sentence register that differs from the long paragraphs instruction-tuned VLMs write by default. Every exposed caption reinforces that style, while a given image–caption link is reinforced only when that item repeats. We therefore expect:

- **Low exposure** → style adaptation: benchmark-style captions on *every* image, including unseen ones. CIDEr rises without any reference being memorized.
- **High exposure** → reference memorization: the training caption reproduced verbatim on seen images.

Headline claim to test: **style follows the total number of exposed captions (K), memory follows per-item repetition (d).**

## Research questions

| RQ | Question |
|----|----------|
| RQ1 | Does style adaptation appear at a lower exposure level than reference memorization? |
| RQ2 | Which part of exposure (image, caption, pairing, breadth vs. depth, reference variety) causes which effect? |
| RQ3 | Is the score gain real quality, or inflation from style? |
| RQ4 | Do detectors (DCQ, CDD, KDS) catch exposure, memorization, or inflation? |

## Setup

- **Benchmark:** NoCaps validation (4,500 images, 10 human references each). Half of the images are exposed (stratified by in-/near-/out-of-domain); for each exposed image, 1 reference is used for training and 9 are held out for scoring. A class-disjoint split tests vocabulary transfer.
- **Model:** LLaVA-1.5-7B with LoRA on the language model + projector (vision encoder frozen). Fallback/replication: SmolVLM-2.2B or Molmo-7B-O, chosen after a training-data audit.
- **Exposure grid:** repetition d ∈ {0, 1, 4, 16, 64}, under four conditions:
  - **Image:** benchmark image with a non-reference target (the base model's own caption)
  - **Caption:** reference caption in a text-only example
  - **Shuffled:** images and captions both exposed, captions permuted across images
  - **Paired:** image with its training reference under the captioning prompt
- **Controls:** a shared d=0 baseline, plus a same-distribution control (COCO captions matched in count and length). Exposed examples replace general instruction examples, so the total example budget M stays fixed.

## Metrics

Every checkpoint is scored on four components:

| Part | What it measures | Key metrics |
|------|------------------|-------------|
| **S** — style | Style change on unseen images | Caption length, boilerplate rate, style classifier |
| **A** — item association | Gain specific to seen items | Δ<sub>d,c</sub> (seen − unseen gain vs. d=0), Γ<sub>d</sub> (paired gain beyond unimodal gains) |
| **V** — verbatim memory | Reproduction of the training reference | ΔRMI, rank of the training ref among 10, exact match, longest shared n-gram |
| **Q** — real quality | Whether captions actually improved | CIDEr, SPICE, CIDEr on style-counterfactual refs, CLIPScore, LLM judge, human pairwise |

Detectors: DCQ and CDD (AUROC for separating seen from unseen images), KDS (dataset-level). False positives are measured on the d=0 and same-distribution controls.

## Experiment plan

The plan has three tracks:

- **Track A — gates (G0–G4):** run in order, each with a preregistered pass rule and fallback.
  - G0: pipeline and noise check
  - G1: pilot (W4, go/no-go for ICML)
  - G2: locate τ<sub>style</sub> and τ<sub>memory</sub> (W6)
  - G3: replication on a fresh split
  - G4: quality vs. inflation
- **Track B — independent work (B1–B8):** data audit, style-counterfactual references, prompt interventions, style metrics, metric sensitivity, detectors, replication-model port, related work.
- **Track C — extensions (C1–C11):** modality/pairing, breadth vs. depth at fixed K (core RQ1 evidence), wording vs. meaning, dilution, realistic contamination inside LLaVA-665K, update scope, class transfer, a second model, a detector analysis, a second task, a third model.

Pass rules and decision margins are fixed in `prereg.md` before results come in. Thresholds are chosen on split A and confirmed on split B.

See [`docs/`](docs/) for the full proposal and experiment plan.

## Team

| Member | Role |
|--------|------|
| Minho Chung | Training runs, go/no-go calls |
| Yoorhim Cho | Evaluation, metrics, gate verdicts |
| Kaustav Chanda | Model choice, condition formats, detectors |
| Injoh Tanwani | Reference sets, metric sensitivity, paper support |

## Repository layout

```
docs/
  CSE576_Project.pdf      # project proposal
  experiment-plan.pdf     # detailed experiment plan (gates, tracks, timeline)
prereg.md                 # preregistration draft (D5 margins, pass rules)
configs/                  # default.yaml (paths, seeds, margins, boilerplate), prompts.yaml
src/contam/               # data/ (prep, splits), eval/ (S, A, V, Q, stats), models.py, generate.py
scripts/                  # CLIs below
tests/                    # pytest, no GPU needed
```

## Evaluation pipeline

Setup (cluster):

```bash
conda activate /scratch/$USER/envs/contam && pip install -e .
export CONTAM_DATA_ROOT=/scratch/$USER/data-contamination/data   # or symlink ./data there
pytest -q
```

Data, outputs and checkpoints stay out of git (`data/`, `outputs/` are gitignored).

```bash
# 1. NoCaps val parquet -> annotations.json + images (domain labels from a separate table)
python scripts/prep_nocaps.py --domains $CONTAM_DATA_ROOT/nocaps/domain_source/nocaps_val_domains.parquet
# 2. Splits A (explore) and B (confirm)
python scripts/make_splits.py
# 2b. SameDist control (D4), general-pool order, then freeze before G0 (D7)
python scripts/make_samedist.py                 # COCO train2017 captions -> samedist_{A,B}.json
python scripts/general_pool.py                  # LLaVA-665K -> general_order.jsonl (add B1 exclusions first)
python scripts/freeze_manifests.py              # manifests/MANIFEST.lock.json (tracked); --check before every run
python scripts/build_g0.py --M <M>              # G0 run specs (d=0 x 5 seeds + base); needs M from D2
# 3. Per checkpoint: T=0 captions, teacher-forced ref log-probs, metrics
python -m contam.generate --checkpoint-id d0_s0 --adapter CKPT --split data/splits/split_A.json --out outputs/gens/d0_s0.jsonl
python scripts/score_refs.py --checkpoint-id d0_s0 --adapter CKPT --split data/splits/split_A.json --out outputs/logprobs/d0_s0.jsonl
python scripts/evaluate_checkpoint.py --checkpoint-id d0_s0 --generations outputs/gens/d0_s0.jsonl \
    --logprobs outputs/logprobs/d0_s0.jsonl --split data/splits/split_A.json --out outputs/metrics/d0_s0.json
#    (add --baseline outputs/metrics/d0_s*.json for paired S / Δ / ΔRMI vs. d=0)
#    Every eval CLI refuses a split that is not byte-identical to its manifests/MANIFEST.lock.json entry
#    and records that entry's sha256 (--skip-manifest-check: smoke tests only, output not gate-valid).
#    G0 chain for the five d=0 checkpoints (steps 3 and 4, pipelined, resumable; --dry-run to preview):
python scripts/run_g0_eval.py --split data/splits/split_A.json --run G0_d0_s0=CKPT0 ... --run G0_d0_s4=CKPT4
# 4. G0 margins, G1 report, B3 prompt intervention
python scripts/compute_margins.py --d0 outputs/metrics/d0_s*.json --base outputs/metrics/base.json --out outputs/margins_A.json
python scripts/gate_report.py --manifest g1_manifest.yaml --out outputs/g1_report.json
python scripts/sim_g1b_null.py   # G1(b) false-PASS rate under no effect (prereg.md, Section 6)
python scripts/b3_prompt_intervention.py {generate,evaluate} --split data/splits/split_A.json --model-tag base --model-tag d0_s0=CKPT
```

Use `python -m contam.generate --temperature 1` to get T=1 samples. Definitions and pass rules are in `prereg.md`.
