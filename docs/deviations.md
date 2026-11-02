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
