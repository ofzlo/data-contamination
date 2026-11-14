# Style Before Memory

**When does benchmark exposure start to inflate VLM scores?**

**[Interactive demo →](https://ofzlo.github.io/data-contamination/)** Sample captions from every
exposure level, showing style adaptation and verbatim memorization side by side.

A controlled study of benchmark contamination in vision-language models (VLMs). We fine-tune
LLaVA-1.5-7B (LoRA) on NoCaps validation items at controlled exposure levels and separate two
effects that both raise caption scores:

- **Style adaptation:** the model starts writing in the benchmark's short, one-sentence caption
  register, on *every* image, including images it never saw.
- **Reference memorization:** the model reproduces the specific training caption of an exposed
  image.

The hypothesis is that style follows the total number of exposed captions (K), while memory
follows per-item repetition (d), so style should appear at a lower exposure level than memory.

## Main findings (split A)

| Question | Result |
|---|---|
| Does style appear before memory? | **Yes.** At d=1 caption length drops by ~73 words (saturated) while the latent-memory signal (ΔRMI) stays below its margin. |
| When does latent memory appear? | **d=2**: ΔRMI above the preregistered margin δ_V. |
| When does verbatim memory appear? | **d=4**: exact match on seen images goes from 3.5% (d=2) to 83% (d=4) and 99.95% (d=64). |
| Can style appear without any memory? | **Yes.** The same-distribution control (COCO captions) gives the same style shift as Paired exposure, with no memory up to d=64. |
| With format-matched (short) prompts, is there a real unseen gain? | **Small.** Unseen CIDEr rises by +6.0 at Paired d=1 (vs. +114 under the long default prompt, which is mostly the length penalty). |

Only the G1 gate verdict follows the preregistered rule; the dose / K-sweep and short-prompt
analyses are exploratory.

## Design

- **Benchmark:** NoCaps validation (4,500 images, 10 references each). Half of the images form
  the exposure pool (stratified by in- / near- / out-of-domain); each exposed image has 1
  training reference and 9 held-out references. Two splits: A (explore) and B (confirm).
- **Model:** `llava-hf/llava-1.5-7b-hf` with LoRA (r=128, α=256, lr 2e-4) on the language model's
  attention and MLP projections; the projector is trained in full (PEFT `modules_to_save`) at
  lr 2e-5 and the vision encoder is frozen. Replication model: SmolVLM.
- **Exposure:** N_exp = 1,000 exposed images, each repeated d ∈ {0, 1, 2, 4, 16, 64} times. The
  K = N_exp × d exposed examples *replace* general LLaVA-665K examples, so every run trains on the
  same total M = 130,000 examples.
- **Conditions:** Paired (image + its reference caption), Caption (caption only, text-only),
  Image (image + the base model's own caption), Shuffled, and SameDist (COCO images / captions
  matched in count and length). The d=0 baseline uses 5 training seeds.
- **Metrics:**

  | Part | Measures | Metrics |
  |---|---|---|
  | S: style | style change on unseen images | caption length, boilerplate rate |
  | A: association | gain specific to seen images | Δ (seen − unseen gain vs. d=0), Γ (paired beyond unimodal) |
  | V: verbatim memory | reproduction of the training reference | ΔRMI, rank among 10 refs, exact match, longest shared n-gram |
  | Q: quality | whether captions really improved | CIDEr-D, SPICE (held-out and all-10 references) |

  Contamination detectors (DCQ, CDD, KDS) are evaluated as seen-vs-unseen classifiers.
- **Data audit (B1):** 20 NoCaps val images are also in LLaVA's training data (COCO / Visual
  Genome). They are excluded from group statistics and from the SameDist and general pools; the
  ID lists are in [`resources/b1/`](resources/b1/).

Decision margins, gate pass rules and every protocol detail are fixed in
[`prereg.md`](prereg.md).

## Repository layout

```
configs/      default.yaml (paths, seeds, margins, training), prompts.yaml (captioning prompts)
manifests/    MANIFEST.lock.json: sha256 of every frozen split / pool file (checked before each run)
resources/    B1 overlap and exclusion ID lists
src/contam/   data/ (NoCaps prep, splits, exposure), eval/ (S/A/V/Q, stats, gates, detectors),
              models.py, generate.py, train.py
scripts/      CLIs for every step below; analysis/ holds the exploratory analyses
tests/        pytest suite (no GPU needed; GPU / Java tests are marked)
prereg.md     preregistration: definitions, margins, pass rules
```

## Reproducing

### Requirements

- Python ≥ 3.10, a CUDA GPU for training and generation (runs were done on one A100 80GB;
  ~11.6 GPU-h per d=0 run at M = 130k), and Java (for the PTB tokenizer and SPICE).
- Disk: ~4 GB for NoCaps, the LLaVA-665K json and COCO captions, plus ~100 GB for the
  LLaVA-665K images used in training.

### 1. Environment

```bash
git clone https://github.com/ofzlo/data-contamination.git && cd data-contamination
pip install -e ".[models,data,test]"
pytest -q                         # CPU-only unit tests
```

All large files live under `$CONTAM_DATA_ROOT` (default `./data`); outputs go to `./outputs`
and checkpoints to `$CONTAM_CKPT_DIR` (default `data/ckpt`). None of these are tracked by git,
except the frozen splits in `data/splits/` (`split_{A,B}.json`, `samedist_{A,B}.json`,
`general_order.jsonl`), whose sha256 are pinned in `manifests/MANIFEST.lock.json`. If you set
`CONTAM_DATA_ROOT` elsewhere, copy `data/splits/` there.

```bash
export CONTAM_DATA_ROOT=/path/to/storage/data   # optional
export CONTAM_CKPT_DIR=/path/to/storage/ckpt    # optional
export CONTAM_PY=$(which python)                # interpreter used by the SLURM job scripts
```

### 2. Data and frozen manifests

`scripts/setup.sh` creates a conda env (`$CONTAM_ENV`, default `./env`), downloads NoCaps val
(`lmms-lab/NoCaps`), the NoCaps domain labels, `llava_v1_5_mix665k.json` and the COCO 2017
captions, then rebuilds every manifest and checks it against `manifests/MANIFEST.lock.json`
byte for byte:

```bash
bash scripts/setup.sh
```

If you manage the environment yourself, the data steps are:

```bash
python scripts/prep_nocaps.py --domains $CONTAM_DATA_ROOT/nocaps/domain_source/nocaps_val_domains.parquet
python scripts/make_splits.py                                                       # splits A and B
python scripts/make_samedist.py --exclude-coco-ids resources/b1/b1_exclude_coco_ids.txt
python scripts/general_pool.py --exclude-images resources/b1/b1_exclude_665k_images.txt
python scripts/freeze_manifests.py --check                                          # must pass
```

Training also needs the LLaVA-665K images under `$CONTAM_DATA_ROOT/llava/{coco,gqa,ocr_vqa,textvqa,vg}`
(see the [LLaVA repository](https://github.com/haotian-liu/LLaVA) for download instructions).

### 3. Training

Run specs are built first, then trained with a resumable runner (works interactively or under
SLURM; adjust the `#SBATCH` lines to your cluster).

```bash
# d=0 baseline: 5 seeds + the untrained base (eval only)
python scripts/build_g0.py --M 130000
bash scripts/run_g0_train.sh                 # or: sbatch scripts/run_g0_train.sh

# exposed cells, e.g. d=4 for three conditions, seed slot 0
python scripts/build_g1.py --M 130000 --n-exp 1000 --conditions Caption Paired SameDist \
    --doses 4 --slots 0 --link-nocaps
RUN_SPECS="outputs/runs/G1/G1_Paired_d4_s0.json" bash scripts/run_train.sh
```

`--exploratory` in `build_g1.py` allows doses outside the preregistered grid (e.g. d=2) and other
N_exp values (K sweep).

### 4. Evaluation

Per checkpoint: T=0 captions under the captioning prompt, teacher-forced log-probabilities of
all 10 references, then S/A/V/Q metrics.

```bash
CKPT=$CONTAM_CKPT_DIR/G1_Paired_d4_s0
SPLIT=data/splits/split_A.json
python -m contam.generate --checkpoint-id G1_Paired_d4_s0 --adapter $CKPT --split $SPLIT --out outputs/gens/G1_Paired_d4_s0.jsonl
python scripts/score_refs.py --checkpoint-id G1_Paired_d4_s0 --adapter $CKPT --split $SPLIT --out outputs/logprobs/G1_Paired_d4_s0.jsonl
python scripts/evaluate_checkpoint.py --checkpoint-id G1_Paired_d4_s0 --split $SPLIT \
    --generations outputs/gens/G1_Paired_d4_s0.jsonl --logprobs outputs/logprobs/G1_Paired_d4_s0.jsonl \
    --baseline outputs/metrics/G0_d0_s*.json --out outputs/metrics/G1_Paired_d4_s0.json
```

Batch versions: `scripts/run_g0_eval.py` (whole d=0 chain plus margins), `scripts/run_gen_job.sh`
(GPU stages for a list of `RUN_IDS`), and `scripts/run_short_eval_job.sh` (format-matched
short-prompt evaluation). Every evaluation CLI refuses a split that does not match its entry in
`manifests/MANIFEST.lock.json`.

### 5. Gates and analyses

```bash
python scripts/compute_margins.py --d0 outputs/metrics/G0_d0_s*.json --base outputs/metrics/G0_base.json --out outputs/margins_A.json
python scripts/gate_report.py --manifest g1_manifest.yaml --out outputs/g1_report.json
python scripts/g2_report.py --split $SPLIT --margins outputs/margins_A.json --g1-report outputs/g1_report.json \
    --baseline outputs/metrics/G0_d0_s*.json --checkpoints outputs/metrics/G*_d*_s*.json --out outputs/g2_report_A.json

# exploratory dose / K-sweep analysis and figures
python scripts/analysis/explore_dose.py --split $SPLIT --tag my_run --checkpoints outputs/metrics/G*_d*_s*.json

# contamination detectors (CDD, DCQ, KDS) and their report
python scripts/run_detectors.py {cdd,dcq,kds,report} --help
```

Null simulations for the gate rules: `scripts/sim_g1b_null.py`, `scripts/sim_g2_null.py`.

## Configuration

`configs/default.yaml` holds paths (expanded from environment variables such as
`CONTAM_DATA_ROOT`), model IDs, generation and training settings, split seeds, and the
preregistered margins. `configs/prompts.yaml` holds the captioning prompt `p_cap` and the
alternative prompts used in the prompt-intervention and short-prompt analyses.
