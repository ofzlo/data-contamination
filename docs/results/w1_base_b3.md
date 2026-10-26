# W1 results: G0 base check and B3 prompt intervention

Yoorhim, 2026-10-21. The wording was locked on 2026-10-26 (Minho, prereg Section 7) as run
here: neutral "Describe the image.", nocaps_like, verbose "Describe the image in detail." (= p_cap
since D3). So the B3 base results below are final; only the G0 base row under the old p_cap
placeholder ("Describe the image.") must be redone (prereg Section 3).

## Setup

- **Model:** untrained `llava-hf/llava-1.5-7b-hf`, bf16. No LoRA, no training.
- **Data:** NoCaps val, split A (`7b1f1ad80182…`), all 4,500 images.
- **Decoding:** T=0 greedy, batch_size 16, max_new_tokens 300. No caption hit the cap.
- **Metrics:** CIDEr-D and SPICE (pycocoevalcap, PTB tokenizer, ×100), plus S metrics, from
  `scripts/evaluate_checkpoint.py` and `scripts/b3_prompt_intervention.py`.
- **CIs:** image-level paired bootstrap, B = 2,000, 95%.
- **Hardware:** one A100 80GB. Generation takes ~32 min per prompt for long-form captions and
  ~17 min for one-sentence captions. SPICE takes ~1 h per reference configuration for long-form
  captions (see Notes).

## B3: prompt alone

| Prompt | CIDEr all10 | CIDEr held-out | SPICE all10 | SPICE held-out | Words | Sentences | Boilerplate |
|---|---|---|---|---|---|---|---|
| neutral "Describe the image." (= current p_cap) | 0.003 | 0.003 | 16.11 | 16.29 | 72.6 | 4.29 | 100% |
| nocaps_like "Describe the image in one short sentence." | 105.6 | 105.3 | 15.57 | 16.29 | 11.1 | 1.00 | 0% |
| verbose "Describe the image in detail." | 0.001 | 0.001 | 16.01 | 16.18 | 76.5 | 4.48 | 100% |

Paired change vs. neutral, estimate [95% CI]:

| | CIDEr held-out | SPICE held-out | SPICE all10 | Words | Boilerplate |
|---|---|---|---|---|---|
| nocaps_like | +105.3 [+103.4, +107.3] | −0.00 [−0.21, +0.21] | −0.54 [−0.74, −0.34] | −61.5 [−62.1, −60.9] | −1.00 |
| verbose | −0.00 [−0.01, +0.00] | −0.11 [−0.19, −0.02] | −0.10 [−0.18, −0.01] | +3.9 [+3.5, +4.3] | 0.00 |

The equal SPICE held-out means (16.2903) for neutral and nocaps_like are a coincidence, not a
cache artifact. Per image, only 0.3% of values are equal (r = 0.39). The group means differ in
opposite directions and cancel: seen 16.35 vs. 16.32, unseen 16.23 vs. 16.26.

## Does a length-controlled CIDEr remove the style effect?

`scripts/analysis/b3_length_control.py` scores the same captions against the 9 held-out
references in three ways. Each variant scores the 4,500 images as one corpus.

- **CIDEr-D:** standard, σ = 6.
- **CIDEr-D without the length penalty:** σ → ∞; clipping is kept.
- **First sentence only:** CIDEr-D on each caption's first sentence.

Word counts here are whitespace counts.

| Prompt | Words | First-sentence words | CIDEr-D | No length penalty | First sentence |
|---|---|---|---|---|---|
| neutral | 72.4 | 15.5 | 0.00 | 42.5 | 53.9 |
| nocaps_like | 11.1 | 11.1 | 105.3 | 120.6 | 105.3 |
| verbose | 76.3 | 15.5 | 0.00 | 41.1 | 53.4 |

nocaps_like − neutral, estimate [95% CI]:

| Metric | Difference |
|---|---|
| CIDEr-D | +105.3 [+103.4, +107.3] |
| No length penalty | +78.1 [+76.4, +80.0] |
| First sentence | +51.4 [+49.9, +53.0] |
| SPICE held-out (above) | −0.00 [−0.21, +0.21] |

- Removing the length penalty, or comparing only first sentences, still leaves a +51 to +78
  point gap. CIDEr's TF-IDF cosine dilutes long captions and penalizes their register (for
  example "The image features …"), so no CIDEr variant isolates content.
- SPICE held-out shows no gap.
- **Recommendation for the prereg Section 7 decision:** if the long-form p_cap is kept, make
  SPICE held-out a co-primary A metric. A CIDEr variant would still carry the S effect.

## G0: base sanity check

- With the p_cap placeholder, base CIDEr all10 is **0.003**. The prereg warning band is
  [95, 115], built around the ≈105 that lmms-eval reports.
- With the one-sentence prompt, base CIDEr is **105.6**. So the pipeline reproduces the
  reported number. The band only applies to a one-sentence prompt.
- By domain, CIDEr all10 is ≤ 0.007 everywhere. SPICE held-out ranges from 15.0 (out) to 17.1 (in).
- Average words: 64 (out) to 80 (in).

## V on the base model (teacher-forced, p_cap placeholder)

`scripts/score_refs.py` scored all 4,500 images × 10 references, 45,000 sequences (caption + EOS).

| Group | log p(train ref), summed | Train-ref tokens | RMI summed | RMI per token | Train-ref rank of 10 |
|---|---|---|---|---|---|
| seen | −45.42 | 15.9 | −0.17 | −0.010 | 5.53 |
| unseen | −45.33 | 15.9 | −0.04 | +0.004 | 5.54 |

- These are the values expected with no exposure: the training reference ranks at chance (5.5)
  and RMI is ≈0.
- The seen-vs-unseen gap in base RMI (0.13 nats) comes only from which reference was assigned
  as the training reference. ΔRMI subtracts the d=0 value per image, so this gap cancels.
- The gap is small next to the δ_V floor placeholder (0.5 nats).

## Eval cost per checkpoint (A100 80GB, 4 CPUs)

| Step | Time |
|---|---|
| T=0 generation, 4,500 images | ~32 min GPU (long-form), ~17 min (one sentence) |
| Teacher-forced scoring, 45,000 refs | 67 min GPU |
| CIDEr + SPICE, 3 reference configs | ~3 h CPU for new long-form captions; minutes for short ones or cache hits |

- This is roughly **1.7 GPU-h per checkpoint**, before the T=1 samples for CDD (20 per image).
- At ~100 P0 runs, evaluation alone is ~170 GPU-h. Please include this in D2 (Minho).
- SPICE runs on CPU and can overlap the next checkpoint's GPU work.

## What this means

1. **Style alone moves CIDEr by ~105 points with no training,** while held-out SPICE does not
   move. This is a training-free RQ3 result: the score gain comes from caption length and
   register, not from content.
2. **CIDEr-D is close to a length gate.** Its Gaussian length penalty (σ = 6 words) multiplies
   the n-gram similarity by ≈e^−51 at 72 vs. ~11 words.
   - Under the current p_cap, any exposure that shortens captions will raise CIDEr-D from ≈0.
   - That puts an S effect inside the primary A metric (Δ on held-out CIDEr, eq. 3).
   - Raised as a team decision in prereg Section 7: keep the long-form p_cap with a
     length-controlled co-primary A metric, or use a one-sentence p_cap.
3. **The G0 base-CIDEr band must be set for the locked p_cap** (prereg Section 3).

## Notes

- SPICE runs once per reference configuration (held-out, train, all10). Its own parse cache
  cannot store long captions ("Caption may be too long"), so long-form outputs are re-parsed in
  each run, ~1 h each on 4 CPUs. Our per-image SPICE cache (`outputs/cache/spice_cache.json`)
  makes reruns of identical captions instant.
- Outputs (gitignored, on the cluster):
  - `outputs/g0/base_caption.jsonl` and `outputs/g0/base_metrics.json`
  - `outputs/b3/base.jsonl`, `outputs/b3/b3_report.json` and `outputs/b3/b3_report_cider.json`
  - `outputs/logprobs/base.jsonl` and `outputs/g0/base_metrics_v.json` (G0 metrics with V)
  - `outputs/b3/length_control.json`
