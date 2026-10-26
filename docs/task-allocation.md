# Task allocation

Source: `docs/experiment-plan.pdf` ("What each member does", Oct 19). Updated 2026-10-26: **all of
Kaustav's items move to Minho.** Yoorhim's and Injoh's items are unchanged. Items that came from
Kaustav are marked (K). Status as of 2026-10-26; ✓ done, ◐ in progress, ○ not started.

## Minho: training, running experiments, model choice, detectors

Owns every training run, the go or no-go calls, the model choice, the condition formats and
everything about detectors.

| Week | Item | Status |
|---|---|---|
| W1 | Timing run (~1k steps) for GPU-h per run, then fix M and N_exp (D2) | ○ runbook in README; needs the cluster |
| W1 | Splits and seed scheme (D7); lock SameDist (D4); start G0 training (d=0 × 5 seeds) | ✓ D7, D4; ○ G0 (after D2) |
| W1 | (K) B1 training-data audit; choose the main and replication models (D1, D6) | ✓ B1, D1 (LLaVA-1.5-7B); ✓ D6 audit (`docs/results/d6_audit.md`), SmolVLM-2.2B proposed, needs team sign-off |
| W1 | (K) Write the B3 prompts | ✓ locked 2026-10-26 (`configs/prompts.yaml`, prereg Section 7) |
| W2–W4 | (K) Implement DCQ, CDD, KDS (B6); run on base, d=0 and G1 | ◐ code and tests done (`contam.eval.detectors`, `scripts/run_detectors.py`); runs need T=1 samples and the d=0 checkpoints |
| W3 | Train G1 pilot (12 runs) | ○ |
| W3–W4 | (K) Spot-check Injoh's B2 rewrites | ○ |
| W4 | ICML go or no-go call with the team (G1) | ○ |
| W5–W6 | Train G2 (21 runs) and C2 (12 runs); start C1 (24 runs) | ○ |
| W5–W6 | (K) Port the replication model (B7) and reproduce its d=0 baseline | ○ |
| W6–W7 | (K) Design the second task (C10) | ○ |
| W7–W8 | Train G3, C3, C4, C5 (665K runs), C8 (replication model), C2b, C10 | ○ |
| W8–W10 | (K) Evaluate C8 on the replication model | ○ |
| W9 | Train C6, C7 if compute allows | ○ |
| W9–W11 | (K) C9 detector analysis, incl. the style-corrected score | ○ |
| W12–W14 | Paper intro, method and setup sections; merge the full draft | ○ |
| W12–W14 | (K) Detector and qualitative analysis sections | ○ |

The W7–W10 load is heavy (C8 training and C8 evaluation in the same weeks). If it slips, the plan's
cut list applies first; C8 itself stays to the end.

## Yoorhim: evaluation and analysis

| Week | Item | Status |
|---|---|---|
| W1 | Decision margins (D5), Caption and Image formats (D3), prereg.md with the team | ✓ drafted; prereg not locked (D2, D6 open) |
| W1–W2 | Eval pipeline for S, A, V (incl. RMI), Q; run B3 with the B3 prompts | ✓ pipeline, B3 on base; ○ B3 on d=0 |
| W2 | Evaluate G0, fix δ_S and δ_A | ○ (after G0 training) |
| W2–W4 | B4 style metrics and the style classifier | ○ |
| W4 | Evaluate G1, report pass or fail per criterion | ○ |
| W6 | Evaluate G2, fix d_style and d_memory | ○ |
| W7 | G4 automatic metrics | ○ |
| W5–W10 | Evaluate every new checkpoint; bootstrap CIs and monotone fits | ○ |
| W9 | Evaluate G3 | ○ |
| W12–W14 | Results section and figures, appendix | ○ |

## Injoh: references, metric checks, paper support

| Week | Item | Status |
|---|---|---|
| W1–W3 | Related work (B8) | ? not in the repository |
| W2–W3 | Metric sensitivity analysis (B5) | ○ |
| W2–W4 | B2 style-counterfactual references (600 images) | ○ |
| W3 | ICML template, impact statement, reciprocal-reviewer check | ○ |
| W8 | G4 blind human pairwise eval | ○ |
| W12–W14 | Related Work, references, anonymization check | ○ |
