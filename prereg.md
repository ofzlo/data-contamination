# Preregistration: Style Before Memory

> **STATUS: DRAFT, NOT LOCKED.** Drafted by Yoorhim (W1, 2026-10-20). It is locked by team sign-off
> before any exposed (d > 0) run starts. After locking, changes only go in a dated amendment
> section at the end. `TODO(owner)` marks decisions owned by someone else.

Implementation: `src/contam/` (see README, "Evaluation pipeline"). Equation numbers follow the
proposal (`docs/CSE576_Project.pdf`): eq. 1 generation, eq. 2 group score, eq. 3 Δ, eq. 4 Γ.

## 1. Locked decisions (D1 to D7)

| ID | Decision | Status |
|----|----------|--------|
| D1 | Main model: LLaVA-1.5-7B if the B1 audit passes, otherwise SmolVLM-2.2B. | TODO(Kaustav): B1 audit result |
| D2 | Fixed total examples M per run; exposed K = N_exp × d replace general examples. M, N_exp from the W1 timing run (300k / 2,250 or 130k / 1,000). | TODO(Minho) |
| D3 | Caption condition: fixed text-only template. Image condition: the base model's own caption as target. Captioning prompt p_cap. | TODO(Kaustav): templates and p_cap wording (`configs/prompts.yaml`, currently PLACEHOLDER) |
| D4 | SameDist: COCO train captions (no eval-image overlap), matched to NoCaps in count, length and total K. | Built and frozen per Section 2 (`samedist_{A,B}.json`); TODO(Kaustav): sign off, and D3 decides the example format |
| D5 | Decision margins, defined exactly in Section 4. | Drafted here (Yoorhim) |
| D6 | Replication model: SmolVLM-2.2B or Molmo-7B-O, whichever is not the main model. | TODO(Kaustav) |
| D7 | Separate training, split and reference-assignment seeds. Split A = explore, split B = confirm. | Seeds set in Section 2 (Minho); manifests frozen in `manifests/MANIFEST.lock.json` |

## 2. Data, splits and seeds (D7)

- **Benchmark.** NoCaps validation: 4,500 images (417 in-domain, 2,670 near-domain, 1,413 out-domain),
  10 human references each. Images and references come from the lmms-lab parquet mirror. Domain
  labels come from the `image_domain` column of the laicsiifes/nocaps-pt-br mirror, joined on Open
  Images id (the official S3 JSON returns 403). The counts match the official split.
- **Reference normalization** (recorded in `annotations.json: meta.ref_normalization`):
  - Surrounding whitespace is stripped.
  - A trailing "¬" (U+00AC, an annotation-tool artifact on 458 references) is removed. The three
    references with "¬" mid-caption are left unchanged.
  - TODO(Minho): training targets must use these same normalized references.
  - Normalization changes the annotations hash, and so the split content hashes.
    Current hashes: split A `7b1f1ad80182…`, split B `c29202cabb4d…`.
- **Caption normalization before PTB.** Every hypothesis and reference has each Unicode whitespace
  run collapsed to one space before tokenization. This includes \r, \x0b, \x0c, \x85, U+2028 and
  U+2029, which the Java PTBTokenizer would otherwise treat as line breaks. The tokenizer must
  return exactly one line per input caption, or scoring aborts.
- **Seeds.** `split_seed` picks the exposed images. `ref_seed` picks the training reference of each
  image. Training seeds belong to the training config, not the split. All three are independent.
  - Split A (explore): `split_seed = 20261019`, `ref_seed = 20261020`.
  - Split B (confirm): `split_seed = 20261119`, `ref_seed = 20261120`.
  - Training seeds (`training.seeds`): slot k = 0..4 is `20261201 + k`, used for LoRA init and
    data order. d=0 (G0) uses all five slots, G1 cells slots 0 and 1, G2 cells slots 0 to 2. The
    same slot is the same seed in every condition and d, so cells differ only in their data.
  - General-pool order (`training.general_seed = 20261022`): LLaVA-665K is put in one fixed order
    with its sources interleaved, so every prefix keeps the 665K source mix. A run with K exposed
    examples trains on the first M − K general examples plus the K exposed ones; d=0 trains on the
    first M. B1 exclusions are applied before this order is frozen.
- **SameDist (D4).** `contam.data.samedist`: one COCO train2017 caption per seen image, with the
  same word count as that image's training reference (`contam.eval.style.words`). Each COCO image
  is used once; ties inside a length bucket are broken by sha256 of (`samedist.seed = 20261021`,
  annotation id). Rows are assigned smallest exposure set first, so SameDist sets nest like the
  exposure sets, and the run builder repeats each row d times, so K matches. All 2,250 rows of both
  splits match length exactly, and the total word count equals that of the training references.
  Current hashes: SameDist A `3c8c96c1e9bc…`, SameDist B `5b386c7cfbf7…`.
- **Freezing.** `manifests/MANIFEST.lock.json` (tracked) holds the sha256 of every split, SameDist
  and general-order file and the seed config above. Every training and eval run checks it
  (`scripts/freeze_manifests.py --check`); a file can be added to the lock but never changed. Every
  eval CLI also requires the split it is given to be a locked file (same name and sha256) and
  records that lock entry's sha256 (`manifest_split_sha256`; the whole lock's sha256 is recorded as
  provenance only). G0 margins and the gate require all their inputs to share one
  `manifest_split_sha256`, equal to the current lock entry for the split being gated. So an `--add`
  never invalidates existing metrics, and a changed split entry is impossible by construction.
- **Exposure.** 50% of images, stratified by domain. Each stratum gets one permutation (a separate
  RNG stream per stratum). Per-stratum counts use largest-remainder rounding, so the total is
  exactly round(f · N): 2,250 at 50% (209 / 1,335 / 706). The exposure set at fraction f is the
  per-stratum prefix of that permutation.
- **Nested breadth sets (C2).** 50% ⊃ 12.5% ⊃ 3.125%, with 2,250 ⊃ 563 ⊃ 141 images. They are nested
  because each is a prefix of the same permutation. Seen metrics for C2 use the shared 3.125% set.
  The unseen group is the complement of the 50% set in every arm.
- **References.** Every image, seen or unseen, gets a `train_ref_idx`. Seen images train on that
  reference. On all images, the other 9 are held out. So the reference configuration is the same
  in both groups:
  - `heldout`: 9 references
  - `train`: 1 reference
  - `all10`: 10 references
- **Split files** record seeds, counts per domain, a hash of the annotations and a content hash.
  Gate reports record the split hash. Generation (with `--split`), log-prob scoring and evaluation
  recompute the annotations hash and refuse annotations that differ from the split's.
- **Comparability rules** (enforced in `evaluate.check_comparable` and `gates`):
  - Model and d=0 metrics must carry the split's content hash and cover the complete image set.
    Otherwise the comparison raises an error.
  - `--allow-partial` exists for smoke tests only. Its output is marked `gate_valid: false`.
  - The baseline must be five distinct d=0 runs, with distinct files, checkpoint ids and
    generations. No metrics file or checkpoint id may appear in both the baseline and the
    checkpoints.
  - The gate accepts only margins marked `gate_valid: true` and computed from exactly the gate's
    baseline files (compared by resolved path).
  - Every G1 cell must have `gates.g1_seeds_per_cell` (2) training seeds, so the every-seed rule
    is never decided by one seed.
  - All baseline and checkpoint generations must record the same, non-missing `prompt_sha`,
    model, `batch_size` and `max_new_tokens`.
  - Only `--allow-partial` relaxes the margins, file-identity and seed-count rules.
- **Shared d=0.** The d=0 model sees no NoCaps items, so one set of d=0 checkpoints serves both splits.
  Its metrics are recomputed for each split's grouping and reference assignment.
- **C7 class-disjoint split.** Not built yet (stub). It needs NoCaps class labels.

## 3. Metrics (S, A, V, Q)

All metrics are per image. Each checkpoint is scored on all 4,500 images. Generation is T=0 greedy
with p_cap (eq. 1). T=1 samples (20 per image, seed `t1_base_seed + image_id`) are stored for CDD
and robustness.

- **Provenance.** Every generation and log-prob row records `checkpoint_id`, model, adapter and
  `prompt_sha` (sha256 of the exact prompt text). Generation rows also record `batch_size` and
  `max_new_tokens`. Resume refuses a file written under a different prompt, model, adapter,
  `batch_size` or `max_new_tokens`. The gate requires `prompt_sha`, model, `batch_size` and `max_new_tokens`
  to be recorded and identical across the baseline and all checkpoints. Evaluation refuses generations and log-probs whose `prompt_sha` or
  `checkpoint_id` disagree.
- **Batch composition.** Batched greedy decoding in bf16 can depend on which images share a batch.
  So T=0 batches are fixed: sorted image ids are cut into chunks of `batch_size`, and `batch_size`
  is stored in every row. On resume, a partly finished chunk (for example the last chunk of an
  earlier `--max-images` run) is regenerated whole: its old rows are first removed from the file,
  so every chunk in the final file comes from one batch.
- **Adapters.** Loading a LoRA adapter fails unless LoRA modules match the model, the adapter
  file's weight keys equal the injected LoRA parameters (nothing missing or unexpected), and at
  least one `lora_B` is non-zero. On failure the LoRA layers are removed again. This prevents a
  fully or partly mismatched adapter from silently scoring as (part of) the base model. Modules
  whose `lora_B` is still zero only raise a warning that lists them: a trained adapter can have
  modules that never got a gradient (LLaVA's last CLIP layer, with `vision_feature_layer = -2`).
- **Log-prob completeness.** Every image must have log-probs for all 10 references from one
  checkpoint. Duplicate or conflicting rows are an error. On resume, an image counts as done only
  when all 10 are present.

### Q: quality

- **Metrics.** CIDEr-D and SPICE from pycocoevalcap, with PTB tokenization, scaled ×100.
- **Single-corpus rule.** CIDEr IDF depends on the reference corpus scored together. So seen and
  unseen images are always scored in one call per reference configuration. Per-image scores are
  split by group only after that call. The code enforces this: a partial image set raises an error
  unless `--allow-partial` is set, which is for smoke tests only.
- **Group score** (eq. 2): S^g_{d,c} = mean over images i in I_g of m(ŷ_i, R_i).
- **Sanity check.** The untrained base model's CIDEr (all10, all images) is compared with a warning
  band, currently [95, 115] around the ≈105 that lmms-eval reports for LLaVA-1.5-7B on NoCaps val.
  This is a warning band, not a hard pass/fail.
  - W1 check (2026-10-21, split A, all 4,500 images, T=0). The ≈105 holds only for a one-sentence
    prompt. "Describe the image in one short sentence." gives CIDEr 105.6, matching lmms-eval.
    The current p_cap placeholder, "Describe the image.", gives CIDEr 0.003: base captions average
    72.6 words, and CIDEr-D's Gaussian length penalty (σ = 6) zeroes them. SPICE (held-out) is
    16.29 for both prompts. Details: `docs/results/w1_base_b3.md`.
  - TODO(team): the band must be set for the locked p_cap. With a long-form p_cap the band is ≈0.
- **G4 metrics** (counterfactual references, CLIPScore, LLM judge, human pairwise) are out of scope
  for this draft.

### S: style

Primary metrics, measured on **unseen** images:

1. Change in mean word count.
2. Change in boilerplate rate.

Each change is relative to the seed-pooled d=0 model. Seen images are also logged. Secondary
metrics: sentence count, function-word ratio, POS mix (optional, uses spaCy), and the B4 style
classifier (not built yet).

- **Boilerplate.** A caption is boilerplate if it matches any regex in
  `configs/default.yaml: style.boilerplate_patterns`. The list is locked together with this document.
  - The "(a|an|the) (image|picture|photo|photograph) of" and "the (image|…) is a(n)" patterns only
    match at the start of the caption. So content such as "has a picture of a rooster on it" is
    not flagged.
  - Rate on the NoCaps val references: 63 / 45,000 = 0.14%.
  - A labelled set of positive and negative examples lives in `tests/test_review_fixes.py`.
- **Expected direction.** NoCaps captions are short and free of boilerplate. So the directional
  effect is −1 × (model − d0) for both primary metrics; a positive effect means adaptation toward
  NoCaps style.

### A: item association

- Δ_{d,c} = (S^seen_{d,c} − S^seen_0) − (S^unseen_{d,c} − S^unseen_0) (**eq. 3**)
  - Primary metric: CIDEr against the 9 held-out references.
  - Also reported: SPICE, and the `train` and `all10` configurations.
- Γ_d = Δ_{d,pair} − (Δ_{d,image} + Δ_{d,caption}) (**eq. 4**)
- Both are computed from aligned per-image arrays. Because the image sets are identical, a group
  difference equals the mean of the per-image differences.

### V: verbatim memory

**Text metrics** (on the T=0 caption):

- Exact match with the training reference. Before comparing, both strings are lowercased,
  apostrophes are removed, other punctuation becomes a space, and whitespace is collapsed.
- Longest shared contiguous word n-gram with the training reference.
- Control: the same n-gram measure against the 9 held-out references, reported as both max and mean.
  The max over 9 references is biased upward compared with a single reference, so the mean is the
  fairer control.

**Likelihood metrics** (teacher forced, using the same prompt template as generation):

- **Scored sequence.** `prefix(p_cap) + " " + ref.strip()`, followed by one end token: EOS, or
  `<end_of_utterance>` for SmolVLM.
- **Which tokens count.** Log-probabilities are summed over the caption tokens plus the end token.
  Prompt, BOS and image tokens are masked. Including the end token means a reference counts as
  likely only if the model would also stop after it.
  - TODO(Minho): the training target format must use the same separator and end token.
- **Rank** of the training reference among all 10 by summed log p. Rank 1 is the most likely. Ties
  do not count against the training reference.
- **RMI** (Section "Shared metrics" of the plan):
  - RMI_i = log p(r^train_i | x_i) − (1/9) Σ_j log p(r^held_ij | x_i), using summed log p (primary).
  - Also reported: a per-token variant, using log p / n_tokens.
- **ΔRMI_i** = RMI_i(checkpoint) − RMI_i(d=0), on the same image.
- **Primary V effect**: mean ΔRMI over seen images, summed variant. Unseen images serve as a control:
  their "training" reference was never trained on.

## 4. Decision margins (D5)

All margins are computed on split A and reused for split B. Here SD means the sample SD (ddof = 1)
across the five d=0 training seeds.

- **δ_S** is computed for each primary S metric: 2 × SD across seeds of the d=0 model's
  **unseen-group mean** of that metric. Examples: the unseen mean word count of seed s, and the
  unseen boilerplate rate of seed s.
- **δ_A** is computed for each A metric (primary: CIDEr held-out): 2 × SD across seeds of the d=0
  model's **seen-minus-unseen gap** of group means, S^seen_0 − S^unseen_0.
- **δ_V** = max(0.10 × E, floor).
  - E is the pooled point estimate of the Paired d=64 V effect (mean seen ΔRMI, summed), taken at G1.
  - `floor` is a named config value: `margins.delta_v_floor.rmi_sum`, a PLACEHOLDER of 0.5 nats.
    The per-token variant uses a placeholder of 0.02 nats per token.
  - TODO(team): fix the floor values before G1 results are seen.
- **δ_Q** = 5 points of human win rate. This is for G4 and is not computed by this code.

The script `scripts/compute_margins.py` produces δ_S and δ_A. The script `scripts/gate_report.py`
produces δ_V at G1.

## 5. Statistics

- **Paired image-level bootstrap.**
  - Every replicate draws one index vector per group (seen and unseen resampled separately) and
    applies it to every checkpoint being compared.
  - Percentile CIs, 95%, two-sided. B = 2,000 by default (`stats.n_boot`, fixed bootstrap seed).
  - TODO(team): use B = 10,000 for the final gate reports.
- **Seeds.** Per-seed values are reported. Pooled values average each image's score across training
  seeds before bootstrapping. The d=0 reference is always the pool of all five d=0 seeds.
- **Threshold τ.** τ = the smallest d whose effect CI lower bound exceeds δ, from a monotone fit
  over log₂ d. Procedure:
  1. In each bootstrap replicate, compute the effect at every grid dose d > 0. All doses share the
     replicate's image indices. d = 0 is the reference point, not a fit point.
  2. Fit an increasing isotonic regression (PAVA) over x = log₂ d.
  3. Pointwise lower and upper bounds are the 2.5 and 97.5 percentiles of the replicate fits. The
     bounds are monotone in d because every fit is.
  4. τ̂ = the smallest grid d with lower bound > δ. If no d qualifies, τ̂ = +∞ ("never crosses",
     censored) and is reported as such.
- **Components.**
  - τ_style uses the primary S metric `gates.tau_style_metric` (word count), as a directional effect
    on unseen images, with δ_S.
  - τ_memory uses mean seen ΔRMI (summed), with δ_V.
  - TODO(team): should τ_style also be computed on boilerplate rate and reported?
- **τ_memory − τ_style on log₂ d.**
  1. In each replicate, compute both components from the same resampled images.
  2. Each replicate's crossing τ*_b is the first dose whose fitted effect exceeds δ, linearly
     interpolated on log₂ d. It is left-censored at the smallest dose.
  3. The difference is log₂ τ*_mem − log₂ τ*_style, with these censoring rules:
     - Only memory never crosses: +∞.
     - Only style never crosses: −∞.
     - Neither crosses: 0. This means no evidence of separation, the conservative choice for a
       "CI above 0" rule.
  4. Percentiles are order statistics, so ±∞ values are handled exactly. The fraction of censored
     replicates is reported for each case.
  5. Caveat: τ̂ itself uses the grid lower-bound rule (step 4 of the τ procedure). The difference
     CI uses interpolated crossings of the per-replicate fits. These are different estimators, so
     the reported τ̂ difference can fall outside its own CI. Gate decisions use the CI. τ̂ is
     reported for interpretation only.
- **Equivalence (G2 primary).** The V effect at d_style (Paired, mean seen ΔRMI summed, pooled
  over seeds) passes if the upper bound of its two-sided 95% two-level bootstrap CI is below δ_V
  (`gates.g2_equivalence_rule: two_level_bootstrap`). Each replicate resamples the 3 seeds of the
  cell and the 5 d=0 seeds (with replacement), pools each image over the drawn seeds, then
  resamples images, so the CI includes training-seed variance.
  - Proposed by Yoorhim on the basis of the Section 6 G2 table. TODO(team): sign-off; revisit after
    σ_seed for ΔRMI is estimated from the G0 seeds and the G1 seed pairs.
  - Known limitation: resampling k = 3 seeds with replacement understates the seed variance by
    (k − 1)/k, and 3 seeds allow only 10 distinct resamples. Boundary false PASS is therefore still
    above nominal at large σ_seed: 0.046 at σ_seed 0.4 in Section 6. An independent reviewer
    simulation, with seed shifts on trained runs only, gives 0.067 (SE ≈ 0.02).
    Candidate fixes to decide at sign-off:
    1. Inflate the drawn seed deviations by √(k/(k − 1)).
    2. Gate on the larger of the two-level and seed-t upper bounds.
  - The image bootstrap alone does not see training-seed variance, so the CI-only rule falsely
    passes far more often than 2.5% when σ_seed > 0. Three alternatives are computed and reported,
    not gating: the image-bootstrap CI only; that CI AND every seed's point estimate below δ_V
    (mirroring G1(b)); a seed-level t interval (df = 2) over the per-seed estimates.
  - TODO(team): the plan says "upper 95% CI". A one-sided 95% bound (the 90% two-sided interval)
    would be the standard TOST convention. This draft uses the stricter 97.5th percentile.

## 6. Gates and pass rules

The rules below are copied from the experiment plan. Notes in *italics* give the operational
reading implemented in `scripts/gate_report.py`.

**G0. Is the pipeline right, and how noisy is it?**
- Conditions: split A, d=0 × 5 seeds, plus the untrained base model.
- Pass rule: base CIDEr matches the reported range. The seed SD sets δ_S and δ_A. GPU hours per run
  are confirmed.
- If it fails: hold G1 until the eval pipeline is fixed.
- *Base CIDEr is checked against the warning band in Section 3. GPU hours are a manual check (Minho).*

**G1. Can we see style and memory effects separately? (pilot)**
- Conditions: split A, {Caption, Paired, SameDist} × d ∈ {4, 64} × 2 seeds.
- Pass rule:
  - (a) Paired d=64 ΔRMI CI above 0, and exact match clearly above d=0. This sets δ_V.
  - (b) Unseen S change at d=4 exceeds δ_S.
  - (c) The P0 grid fits our compute.
- If it fails:
  - (a) Stop and fix exposure or LoRA.
  - (b) Try d=16. If there is still nothing, drop "style first" and pivot to "decomposing
    contamination gains + detectors".
  - (c) Apply the cut list. If still short, move to NeurIPS 2027.
- *(a) is read as: the 95% CI lower bound of the pooled mean seen ΔRMI (summed) is > 0, AND the 95%
  CI lower bound of the change in seen exact-match rate vs. d=0 is > 0.*
- *(b) is read as one preregistered primary test: Paired × word count at d=4
  (`gates.g1_b_primary`). It passes only if both hold:*
  - *the pooled CI lower bound of the directional unseen change is > δ_S;*
  - *the per-seed point estimate exceeds δ_S in every training seed.*
- *Caption, and the boilerplate rate, are reported as secondary results and do not decide (b).
  SameDist is also reported without deciding (b). The plan says: "If SameDist shows the same S
  change as Caption in G1, that is a finding, not a failure."*
- *Why one test: under no true effect, the earlier "any of {Caption, Paired} × any of 2 metrics"
  rule falsely passes more often than one test, and more so as training-seed variance σ_seed
  grows, because the image bootstrap CI does not include it. The every-seed rule bounds this.
  Null false-PASS rates from `python scripts/sim_g1b_null.py --n-images 900 --n-boot 300
  --reps 1000 --sigma-seed {0,0.5,1,2} --seed 0` (real gate code; per-image word-count SD 4,
  σ_seed in words; model in the script docstring):*

  | σ_seed | G1(b) rule | single test, CI only | any-of-4, CI only |
  |-------:|-----------:|---------------------:|------------------:|
  | 0      | 0.000      | 0.000                | 0.001             |
  | 0.5    | 0.008      | 0.009                | 0.020             |
  | 1      | 0.014      | 0.018                | 0.039             |
  | 2      | 0.010      | 0.025                | 0.053             |
- *Proposed by Yoorhim. TODO(team): sign-off.*
- *(c) is manual.*

**G2. Where are τ_style and τ_memory? (RQ1 main)**
- Conditions: Caption and Paired at d ∈ {1, 16} × 3 seeds, a third seed for d ∈ {4, 64}, and
  SameDist d=16 × 3. Add d=8 if the gap is wide.
- Pass rule:
  - Primary: at d_style, the upper 95% CI of the V effect is below δ_V (an equivalence test).
  - Secondary: the CI of τ_memory − τ_style is above 0 on log₂ d.
  - d_style and d_memory are fixed here.
- If it fails: if both effects start at the same d, reframe as "dose-response per component" and
  skip C3.
- *Implemented in `scripts/g2_report.py` (`contam.eval.gates.g2_report`). Inputs: the five G0 d=0
  seeds as baseline, δ_S from the G0 margins, δ_V from a gate-valid G1 report whose (a) passed,
  and every metrics file under the split's current lock entry (Section 2). Caption and Paired
  share one grid with every d in {1, 4, 16, 64} (plus d=8 if supplied), 3 seed slots per cell;
  SameDist d=16 × 3 is reported only.*
- *d_style = τ̂_style and d_memory = τ̂_memory, both within the Paired condition
  (`gates.g2_tau_condition`). τ_style is the directional unseen word-count effect vs δ_S, τ_memory
  the mean seen ΔRMI (summed) vs δ_V (Section 5). Caption's τ_style and τ_memory are reported, not
  gating.*
- *Primary: the equivalence rule of Section 5 (two-level bootstrap CI upper bound < δ_V). If τ̂_style = +∞ there is no d_style on the grid: the primary is not evaluable
  and fails. The G2 verdict is the primary one. The G1 report must have been made from the same
  baseline and margins files.*
- *PASS rates of the G2 primary rules from `python scripts/sim_g2_null.py --reps 500 --n-boot 300
  --n-images 900 --sigma-seed 0 0.1 0.2 0.4 --mult 1.0 1.2 0 0.5 --seed 0 --jobs 4` (real gate
  code, 11.5 min on 4 CPUs; model in the script docstring: per-image ΔRMI SD 2, δ_V = 0.5, every
  run shifted by N(0, σ_seed), d=0 seeds included; true Paired V effect at d_style = mult × δ_V).
  mult ≥ 1: false PASS (nominal ≤ 0.025); mult < 1: power. The two-level bootstrap is the gate rule:*

  | mult | σ_seed | pooled CI only | pooled + every seed | two-level bootstrap | seed t (df 2) |
  |-----:|-------:|---------------:|--------------------:|--------------------:|--------------:|
  | 1.0  | 0      | 0.032          | 0.032               | 0.000               | 0.036         |
  | 1.0  | 0.1    | 0.088          | 0.062               | 0.010               | 0.034         |
  | 1.0  | 0.2    | 0.184          | 0.118               | 0.030               | 0.044         |
  | 1.0  | 0.4    | 0.322          | 0.160               | 0.046               | 0.032         |
  | 1.2  | 0      | 0.000          | 0.000               | 0.000               | 0.002         |
  | 1.2  | 0.1    | 0.004          | 0.002               | 0.002               | 0.002         |
  | 1.2  | 0.2    | 0.078          | 0.044               | 0.006               | 0.014         |
  | 1.2  | 0.4    | 0.236          | 0.102               | 0.034               | 0.014         |
  | 0    | 0      | 1.000          | 1.000               | 1.000               | 0.982         |
  | 0    | 0.1    | 1.000          | 1.000               | 0.992               | 0.858         |
  | 0    | 0.2    | 0.992          | 0.960               | 0.850               | 0.548         |
  | 0    | 0.4    | 0.876          | 0.680               | 0.498               | 0.236         |
  | 0.5  | 0      | 0.950          | 0.944               | 0.704               | 0.680         |
  | 0.5  | 0.1    | 0.882          | 0.822               | 0.556               | 0.436         |
  | 0.5  | 0.2    | 0.746          | 0.610               | 0.362               | 0.196         |
  | 0.5  | 0.4    | 0.624          | 0.392               | 0.198               | 0.124         |
- *Plainly: the every-seed rule roughly halves the CI-only rule's false PASS, but at the boundary
  (V effect = δ_V) it still falsely passes 6% to 16% for σ_seed 0.1 to 0.4 (vs. 2.5% nominal), and
  10% at 1.2 × δ_V with σ_seed 0.4. Only the two-level bootstrap (≤ 4.6%) and the seed t interval
  (≤ 4.4%) stay near nominal, at a clear cost in power; the two-level bootstrap has more power
  than the t interval, so it is the default. It is still above 2.5% at σ_seed 0.4 (0.046): three
  seeds per cell cannot pin down seed variance. TODO(team): sign off, and revisit after estimating
  σ_seed for ΔRMI from the five G0 seeds and the G1 seed pairs.*
- *Secondary: the bootstrap CI of log₂ τ*_memory − log₂ τ*_style (Section 5 censoring rules) has
  lower bound > 0; censoring fractions are reported.*
- *The fallback above is reported whenever τ̂_style = τ̂_memory (finite).*
- *TODO(team): should τ_style come from Paired, or from Caption? Should a failed secondary also
  fail G2?*

**G3. Does the order replicate on a new split?**
- Conditions: split B, {Caption, Paired} × {d_style, d_memory} × 2 seeds, d=0 shared.
- Pass rule: same order on split B, with effect sizes inside split A's CI.
- If it fails: report the failure, add seeds, and limit the claim to split A.

**G4. Is the low-exposure gain real quality or style inflation? (RQ3 main)**
- Conditions: G2 checkpoints at d ∈ {0, d_style, d_memory} × {Caption, Paired, SameDist}.
  Interventions: counterfactual references, output neutralizing, reference-free metrics, length
  control, an LLM judge, and blind human pairwise on 2 key contrasts.
- Pass rule: call it "inflation" only if all three hold:
  1. The original-reference score goes up.
  2. The human and grounded quality gain is below δ_Q.
  3. The gain shrinks under neutral references and neutralized outputs.
- If it fails: if quality rises too, reframe as "harmless style adaptation that CIDEr cannot tell
  apart from contamination". RQ4 still stands.

## 7. Analysis-plan items not yet decided

- TODO(Kaustav): B3 prompt wording for the neutral, nocaps_like and verbose prompts. Only
  nocaps_like is final ("Describe the image in one short sentence.").
- TODO(Injoh): B2 counterfactual reference protocol, used for G4.
- TODO(Kaustav): detector protocol (DCQ, CDD, KDS) and the false-positive controls.
- TODO(team): sign off on the single-primary G1(b) rule in Section 6, which settles the
  multiple-comparison question at G1. The same question is still open for G2's τ_style metric.
- TODO(team, raised by Yoorhim after W1): CIDEr-D and a long-form p_cap. With the current p_cap,
  base CIDEr-D is ≈0 (Section 3). Any exposure that shortens captions toward NoCaps length then
  moves CIDEr-D from ≈0 to ≈100 through the length penalty alone. That inflates the primary A
  metric (Δ on held-out CIDEr, eq. 3) with an S effect. Options:
  1. Keep the neutral p_cap. Add a length-controlled A metric, either SPICE held-out or CIDEr
     without the length penalty, as co-primary. In B3, the CIDEr variants still separate the
     prompts: without the length penalty by +78, and on the first sentence only by +51. SPICE
     held-out does not (Δ −0.00 [−0.21, +0.21]). So SPICE held-out is the candidate
     (`docs/results/w1_base_b3.md`).
  2. Use a one-sentence p_cap. This removes most of the length-driven S signal that RQ1 relies on.
  Decide before G1.

## Amendments

None yet. The document is not locked.
