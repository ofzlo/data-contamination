# Deviations and exploratory additions

Log of analyses and runs outside the locked preregistration (`prereg.md`, locked 2026-10-28). Everything
here is **exploratory**: no entry is a gate input, and the outputs record `exploratory: true` /
`gate_valid: false`.

**Unchanged by every entry below:** `prereg.md`, the margins (δ_S, δ_A, δ_V rule and floors), the gate
rules (`contam.eval.gates`), the G0 d=0 checkpoints (not retrained) and the locked exposure set
(split A, N_exp = 1,000, ids sha256 `fccfe788952a…`).

## 2026-11-01 (Yoorhim)

Context: the split A seed-0 dose results (`docs/results/dose_explore_d0_d1_d4.md`) show style
saturated at d=1 while verbatim memory appears only at d=4. The additions below locate the onsets
more finely and check the result against a prompt-matched baseline.

| Addition | What | Why |
|---|---|---|
| d=2 runs | Paired / Caption / SameDist at d=2, N_exp = 1,000, seed slot 0. `scripts/build_g1.py --exploratory`: gate `EX`, run ids `EX_<Cond>_d2_n1000_s<k>` (rejected by `gates.parse_run_id`), spec field `"exploratory": true`. | d=2 is not on the prereg grid {1, 4, 16, 64}; it falls between the d=1 (no verbatim memory) and d=4 (85% exact match) points. |
| K sweep | Paired and SameDist at d=1 with N_exp ∈ {30, 100, 300}, same builder flag (`EX_<Cond>_d1_n<N>_s0`). The sets are prefixes of the locked exposure order: verified on split A, 30 ⊂ 100 ⊂ 300 ⊂ 1,000 (ids sha256 `ced64c7371e4…`, `b32f72d48bed…`, `3c688ae05d1b…`, `fccfe788952a…`; none holds a B1 overlap image). Seen for each run = its own N_exp set; unseen unchanged. | Tests "style follows K" below K = 1,000, where style was already saturated. |
| Short-prompt evaluation | T=0 captions under the B3 prompt `nocaps_like` ("Describe the image in one short sentence.") for base, the five d=0 seeds and the trained checkpoints; Q, S, exact match and diversity only (no log-probs). `scripts/run_short_eval_job.sh`, outputs under `outputs/short/`. | Under the long-form p_cap, d=0 has CIDEr ≈ 0, so the CIDEr gain mixes style and content. A prompt-matched baseline asks whether exposure still raises scores. |
| τ_verbatim / τ_latent split | τ procedure of prereg Section 5 applied separately to seen exact match and to seen ΔRMI (summed and per token). τ_verbatim: smallest d whose 95% CI lower bound of the seen exact-match change vs d=0 exceeds δ_verbatim = 0.05 absolute (see the δ_verbatim entry below); the δ = 0 onset (lower bound > 0) is reported as a secondary value. τ_latent: δ_V as set at G1, read from `outputs/g1/g1_report_A.json` `criteria.a.delta_V` (rmi_sum 6.921, rmi_norm 0.4459 = 10% of the Paired d=64 seen effect, above the floors 0.5 / 0.02). | At d=1, ΔRMI already moves (+4.2 nats) while exact match does not (0.5%). τ_memory as preregistered (ΔRMI) and the verbatim onset can differ. |
| Per-token ΔRMI | `rmi_norm` reported next to the summed primary, against the per-token δ_V = 0.4459 from the G1 report (floor 0.02). | Summed ΔRMI grows with caption length; the conditions' references differ in length. |
| Diversity metrics | distinct-1/2 (corpus level), duplicate-caption rate, nouns per caption (spaCy `en_core_web_sm`) on unseen and seen; `contam.eval.diversity`. The duplicate rate depends on group size, so it is not comparable across groups or K. | Checks whether the CIDEr gain comes with collapsed, templated captions (RQ3). |

Analysis: `scripts/analysis/explore_dose.py --tag <tag>` → a new directory `outputs/explore/<tag>/` (every target is checked
before anything is written; existing files are never overwritten), figures `outputs/figures/explore_<split>_<tag>_*`.

### δ_verbatim fixed (2026-11-01, decision by the lead)

Primary δ_verbatim = **0.05 absolute** (5 percentage points of seen exact-match change vs d=0); τ_verbatim = smallest
d whose 95% CI lower bound (prereg τ procedure) is > 0.05. The δ = 0 onset is kept as a secondary value
(`verbatim_exact_match_delta0`).

Reason: the first version used δ = 0, so a handful of exact matches fires the onset; Paired d=1 got τ_verbatim = 1
from 0.5% exact match (5 of ~1,000 seen captions), which is not verbatim memorization in any useful sense.
5 pp is far above the d=0 level (0%) and the d=1 level, and far below the d=4 level (85%).

Timing: fixed after the d=0, d=1 and d=4 results had been seen (the d=1 0.5% value is what showed the problem) and
**before the d=2 results were inspected**. δ_V is not chosen here: it is the value G1 set by the prereg rule.

## 2026-11-02 (Yoorhim): short-prompt evaluation as an RQ3 main analysis

**What.** The short-prompt (B3 `nocaps_like`, "Describe the image in one short sentence.") evaluation of the trained
checkpoints is reported **alongside** the long-prompt results as an RQ3 main analysis. It stays EXPLORATORY: it is not a
gate input, and the prereg primary analyses (G1/G2 gates, long-form p_cap, prereg Sections 2/4/5) are unchanged.

| Item | Choice |
|---|---|
| Baseline | the five short-prompt d=0 seeds (G0_d0_s0..4); the untrained base is reported as a row. |
| Statistic | two-level bootstrap as in the prereg G2 primary (`contam.eval.stats.bootstrap_two_level`: resample training seeds of each side, pool per image, then resample images), B = 10,000, 95% percentile CIs; generalised to several groups and fields in `explore_dose.two_level_group_means` (identical draws to `bootstrap_two_level` in the one-group case, tested). Cells with 2 seeds: pooled and per-seed values. |
| Groups | seen = the run's own exposure set (K-sweep runs regrouped to their N_exp), unseen, seen_unexposed; B1 overlap images excluded; unseen additionally per NoCaps domain (annotation field `domain`: in / near / out). |
| New analyses | (1) seen **all10** CIDEr / SPICE (all 10 refs incl. the training caption = how official NoCaps scoring sees a contaminated image); (2) per-domain unseen CIDEr / SPICE; (3) Paired − SameDist unseen CIDEr / SPICE held-out, **seed-matched** (same seed slots, resampled together; SameDist d=1 has only s0, so d=1 is s0 vs s0), also per domain; (4) decomposition of the seen all10 rise vs d=0: style share = seen_unexposed rise, memorisation share = seen rise − seen_unexposed rise (renamed "seen-specific share" below). |
| CIDEr on subsets | CIDEr-D per image is computed once per checkpoint on the full split (IDF from every split image's references, single-corpus rule); group and domain values are means of those per-image scores (IDF not recomputed on the subset). |

**Why.** Under the long p_cap, d=0 has CIDEr ≈ 0 (length penalty on long descriptions; base with the short prompt gets
104, close to the lmms-eval value), so every trained condition's CIDEr "gain" (+80–114) is mostly the switch to one-sentence
captions; it exaggerates style effects and hides content effects. With a format-matched prompt and baseline, the
remaining changes in unseen quality, the official-scoring (all10) inflation on seen images, and its style vs memorisation
split can be measured. The image-only bootstrap used so far for the exploratory tables ignores training-seed variance; the
two-level bootstrap is the prereg's own way to include it.

Tool: `scripts/analysis/explore_dose.py --formal [--formal-only] --formal-long-from outputs/explore/full_20261101` →
`outputs/explore/<tag>/formal_nocaps_like.{json,md}`, figures `outputs/figures/explore_<split>_<tag>_short_formal_*`
(F1: unseen CIDEr held-out vs d, long vs short prompt, the long panel recomputed with the same bootstrap; F2: the
decomposition). Timing: defined after the seed-0 short-prompt point estimates (d=1, d=4) had been seen; the remaining
18 short-prompt runs (second seeds, d=2, d=64, K sweep) were not inspected before.

### 2026-11-02 (Yoorhim): format-matched style vs. vocabulary transfer (exploratory, stands in for the B4 style classifier)

**What.** `src/contam/eval/style_transfer.py` + `scripts/analysis/style_transfer.py` →
`outputs/explore/style_20261102/style_transfer_A*.{json,md}`. EXPLORATORY, not a gate input; the B4 style classifier
(docs/task-allocation.md, W2–W4) does not exist yet. On the short-prompt (`nocaps_like`) generations, **unseen** images only
(seen = N_exp = 1,000 exposure set, B1 exclusion applied):

| Item | Choice |
|---|---|
| Style distance | Jensen–Shannon divergence (base 2) of the outputs' first-two-word, function-word-type and coarse POS-bigram (spaCy en_core_web_sm UPOS, sentence markers) distributions to (1) the NoCaps references of the seen_unexposed images only (disjoint from the evaluated images, never trained on) and (2) a fixed random sample of 30,000 COCO train2017 captions (seed `stats.boot_seed`). |
| Vocabulary | Open Images V5 boxable class names (601; `data/openimages/class-descriptions-boxable.csv`, downloaded 2026-11-02 from storage.googleapis.com/openimages/v5/, sha256 in `data/openimages/SOURCE.txt`; plain CSV parse) named as nouns (plural-tolerant, longest match) in an output; *matched* = also named in that image's NoCaps references. Class sets: all, non-COCO (name/alias overlap with the 80 COCO categories removed; approximate), rare non-COCO (< 1 per 1,000 COCO sample captions). Overall and per NoCaps domain. |
| Statistic | one image bootstrap (unseen images resampled, B = 2,000, `stats.boot_seed`) shared by every corpus, so all Δ (vs the five d=0 seeds; Paired − SameDist seed-matched) are paired; levels = mean over training seeds of per-seed corpus statistics (keeps the plug-in JS small-sample bias equal for 1-seed cells and the 5-seed d=0); bias-shifted percentile CIs as in `contam.eval.diversity`. References are fixed. Training-seed variance is not in these CIs (d=0 seed SD reported). |
| Interpretation rule | Paired − SameDist unseen vocabulary gain concentrated in out-domain → vocabulary transfer; spread evenly → style; in-domain uses COCO classes and may favour SameDist (trained on COCO captions) — flagged separately. Written into the output md. |

Timing: metrics and the interpretation rule were fixed before any of these numbers were computed. A partial run on the 12
complete short-prompt generations (`style_transfer_A_partial12`) was inspected before the remaining 18 runs arrived; the
final file (`style_transfer_A`) uses the same code and settings.

Also 2026-11-02 (tooling, no rule change): `scripts/gate_report.py --n-boot` sets B explicitly and records it (`n_boot`) in
the report; `outputs/g1/g1_report_A_B10000.json` is the G1 report at the prereg's locked B = 10,000 (Section 5). δ_V
(6.921 summed / 0.446 per token, a point estimate) and the (a)/(b) verdicts are unchanged from the B = 2,000 report.

### 2026-11-02 (Yoorhim): definitions corrected after code review (short-prompt formal analysis, style / vocabulary transfer)

Rerun into new tags (`outputs/explore/short_20261102c/`, `outputs/explore/style_20261102b/`, figures
`explore_A_short_20261102c_*`; `outputs/explore/INVALID_short_20261102b_1checkpoint_scripting_error/` is a discarded run with
a shell quoting error, one checkpoint only, not a result); the earlier `short_20261102` / `style_20261102` outputs are kept as they are, superseded.

| Item | Change |
|---|---|
| Seen-rise decomposition (renamed) | The "memorisation share" is renamed and defined as the **seen-specific share = seen rise − seen_unexposed rise** (JSON key `seen_specific`). It includes memorisation and any other exposed-image-specific gain (e.g. the exposed images not suffering the general drop the unexposed ones show); when the style share (seen_unexposed rise) is negative, the seen-specific share exceeds the total. |
| SameDist / Caption decomposition | SameDist never saw the seen images or their captions, so its seen-specific share is reported as a **negative control** (placebo, expected 0); Caption's is "captions seen without images". Numbers unchanged; labels in md / F2 / JSON (`seen_specific_label`). |
| Paired − SameDist at d=1 | SameDist d=1 has only s0, so the seed-matched contrast is s0 vs s0 with an **image-only CI** (no seed variance); marked in the md. Added a labelled **sensitivity** contrast: Paired s0+s1 pooled vs SameDist s0, two-level bootstrap with each side's seeds resampled on their own (seed variance on the Paired side only), images paired (`contrasts_sensitivity`). The seed-matched contrast stays the primary one. |
| Non-COCO Open Images classes | The first set was too wide. Now also counted as COCO-overlapping: the 13 human body parts (`Human arm/beard/body/ear/eye/face/foot/hair/hand/head/leg/mouth/nose` → person) and 26 near-aliases / kinds with a COCO head noun (Kitchen knife→knife, Coffee table→dining table, Computer monitor→tv, Loveseat→couch, Wall/Alarm/Digital clock→clock, Infant bed→bed, Brown/Polar bear→bear, Mixing bowl→bowl, Submarine sandwich / Hamburger→sandwich, Flowerpot→potted plant, Racket→tennis racket, Taxi/Limousine/Van→car, Ambulance→truck, Stationary bicycle / Bicycle wheel→bicycle, Aircraft→airplane, Watercraft/Canoe/Gondola/Barge→boat). List: `contam.eval.style_transfer.OI_COCO_NEAR_ALIASES`, also recorded in the output JSON. Not added (new object names, not COCO names): subtypes named by a non-COCO noun (duck, owl → bird; mule → horse), dog bed, billiard table, table tennis racket, telephone, tablet computer. Corrected set (`oi_noncoco`, `oi_rare`) is primary; the first set (`oi_noncoco_old`, `oi_rare_old`) is reported as a sensitivity. |
| Other | Domain labels via `contam.data.splits.normalize_domain` in both scripts; style-transfer CIs record `n_boot_used` (finite replicates; precision / recall are NaN in replicates without mentions); F2 draws total, style and seen-specific shares as side-by-side bars with CI whiskers. Caveats added to the md: the COCO reference sample may contain SameDist training captions (overlap counted); the first-two-word feature carries object content; at d=1 seed counts differ between conditions (F1). |

Timing: these changes follow a code review of the earlier outputs, whose numbers had been seen; no metric, statistic or
interpretation rule was changed in response to a result.
