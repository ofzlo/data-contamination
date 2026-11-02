# Style Before Memory: dose / K-sweep 탐색 결과 (split A)

작성 2026-11-02 · Yoorhim · **EXPLORATORY** (G1 판정 제외; 탐색 항목과 그 이유는 `docs/deviations.md` 2026-11-01)

## 결론 (먼저)

**가설("캡션 스타일 적응이 참조 캡션 기억보다 낮은 노출에서 나타난다")을 지지한다.**

- Paired 노출에서 효과가 나타나는 순서: **스타일 d=1 → 잠재 기억(ΔRMI > δ_V) d=2 → 원문 그대로 기억(exact match > 5%p) d=4.**
- d=1에서는 스타일이 이미 포화(unseen 단어 수 −73)됐지만, 기억은 등가 마진 안이다.
  - seen ΔRMI = 4.33 (seed 2개 pooled)
  - 두 단계 bootstrap 95% 상한 **4.74 < δ_V = 6.92**, seed별 4.21 / 4.45
- **SameDist 대조군**(같은 분포의 다른 이미지·캡션)은 스타일 변화가 Paired와 같지만(−73 단어), d=64까지도 기억이 없다(ΔRMI CI가 0을 포함, exact match 0). 스타일 변화는 기억 없이 생긴다.
- 기억은 노출 이미지에만 생긴다. unseen ΔRMI는 모든 조건과 dose에서 약 0이다.

**단서**

1. 스타일은 실험한 최소 노출(N_exp=30, d=1: 13만 개 중 30개)에서 이미 포화됐다. 그래서 τ_style은 아래쪽에서 잘려 있고, 스타일–기억 간격은 하한(≥ log2 1단계)만 말할 수 있다.
   - 여기서 측정한 "스타일"은 사실상 긴 설명 → 한 문장 캡션으로의 **형식 전환**이다.
   - 짧은 프롬프트로 평가하면 d=0 대비 단어 수 차이는 −0.8 이하로 줄어든다(§5).
2. N_exp가 작을수록 이미지당 기억이 강하다(d=1에서 ΔRMI: N=30 6.8 > N=1000 4.3).
   - N_exp=30에서는 CI 상한(9.0)이 δ_V를 넘는다. 즉 d=1 등가는 N_exp=1,000에서의 결론이다.
3. d=2는 seed가 하나뿐이다. G2 판정은 그리드(d=16, SameDist d=16, seed 3개)를 채우지 않아 `--allow-partial`로 돌렸고, gate-valid가 아니다.
4. 품질 지표는 노출과 같은 방향으로 움직이지 않는다.
   - unseen CIDEr(heldout)는 d=1에서 +114, d=64에서 +88로 올랐다(d=0의 CIDEr ≈ 0은 긴 p_cap에 붙는 길이 페널티 때문).
   - unseen SPICE(heldout)는 −0.4에서 −2.2로 내려갔다.
   - 그러니 성능 향상이 아니라 스타일 효과로 읽어야 한다.

## 1. 설정

- **학습 설정:** LLaVA-1.5-7B + LoRA(r 128, lr 2e-4), M = 130,000(LLaVA-665K 일반 데이터 + K개 노출 행), p_cap 프롬프트, T=0. split A.
- **노출:**
  - 고정 노출 집합(N_exp = 1,000, ids sha256 `fccfe788…`)
  - K-sweep 부분집합 30 ⊂ 100 ⊂ 300 ⊂ 1,000(sha `ced64c73…`, `b32f72d4…`, `3c688ae0…`)
- **그룹:** seen = 그 run의 노출 집합; unseen = 2,250장. B1 제외 20장은 모든 결과에서 뺀다.
- **기준선:** d=0 5 seed(재학습 없음). Δ = 모델 − d=0이고, 95% CI는 paired image bootstrap B = 10,000이다. seed가 2개인 칸은 pooled.
- **마진:**
  - δ_S(word count) = 0.8109 (`outputs/g0/margins_A.json`)
  - δ_A(SPICE heldout) = 0.3199
  - δ_V = 6.921 nats summed / 0.4459 per token (G1에서 설정, 아래)
  - δ_verbatim = 0.05 (탐색용, deviations 로그)
- **조건:**
  - Paired = 노출 이미지 + 그 이미지의 NoCaps 참조 캡션
  - Caption = 참조 캡션 텍스트만(이미지 없음)
  - SameDist = 같은 분포의 다른 이미지·캡션

### 학습한 run

| 그룹 | run | 비고 |
|---|---|---|
| P1 | G1 {Paired, Caption, SameDist} d64 s0 | δ_V 결정 |
| P2 | G2 Paired d1 s1 | d=1 seed 2 |
| P3 | EX {Paired, Caption} d2 n1000 s0 | exploratory dose |
| P4 | EX {Paired, SameDist} d1 n{30,100,300} s0 | K sweep |
| P5 | G1 {Paired, Caption, SameDist} d{4,64} s1 | G1 seed 2 |

기존 run(d=0 × 5, d=1 s0 × 3, d=4 s0 × 3)과 합쳐 분석했다.

## 2. G1 정식 판정 (프리레그 규칙, gate-valid)

`outputs/g1/g1_report_A.json` (bootstrap B = 2,000; 프리레그는 최종 보고에 B = 10,000을 요구하므로 재실행 필요)

- **(a) PASS:** Paired d=64 seen ΔRMI = 69.21 [68.33, 70.13], exact match 변화 = +99.95%p.
  - → **δ_V = max(0.10 × 69.21, 0.5) = 6.921 nats**, per token 0.4459
- **(b) PASS:** Paired d=4 unseen word count 효과 = 71.97 [71.24, 72.70] > δ_S 0.81, seed별 71.85 / 72.09.
- (c) 계산 자원 확인은 수동 항목.

## 3. Dose 결과: Δ vs d=0 (95% CI)

**굵게** = CI 전체가 마진을 넘음. 단어 수와 boilerplate는 unseen, 기억 지표는 seen.

| 조건 | d | seeds | 단어 수 (unseen) | exact match %p (seen) | ΔRMI sum (seen) | ΔRMI/token (seen) | ΔRMI sum (unseen) |
|---|---|---|---|---|---|---|---|
| Paired | 1 | 2 | **−73.08 [−73.79, −72.38]** | 0.45 [0.10, 0.85] | 4.33 [3.98, 4.67] | 0.292 [0.260, 0.325] | 0.07 [−0.12, 0.27] |
| Paired | 2 | 1 | **−72.60 [−73.32, −71.89]** | 3.41 [2.31, 4.62] | **13.68 [13.13, 14.22]** | **0.898 [0.854, 0.941]** | −0.03 [−0.24, 0.19] |
| Paired | 4 | 2 | **−71.97 [−72.70, −71.25]** | **83.28 [81.48, 85.04]** | **37.26 [36.49, 38.03]** | **2.401 [2.348, 2.455]** | −0.18 [−0.51, 0.14] |
| Paired | 64 | 2 | **−72.35 [−73.07, −71.63]** | **99.95 [99.85, 100.00]** | **69.21 [68.30, 70.13]** | **4.459 [4.401, 4.518]** | 0.09 [−0.54, 0.73] |
| Caption | 1 | 1 | **−2.93 [−3.45, −2.41]** | 0.00 | 2.20 [2.00, 2.41] | 0.146 [0.131, 0.161] | 0.03 [−0.06, 0.14] |
| Caption | 2 | 1 | **−2.34 [−2.88, −1.80]** | 0.00 | 5.23 [4.92, 5.54] | 0.343 [0.321, 0.365] | 0.01 [−0.10, 0.11] |
| Caption | 4 | 2 | **−2.41 [−2.82, −1.98]** | 0.00 | **8.76 [8.37, 9.16]** | **0.572 [0.544, 0.600]** | 0.02 [−0.11, 0.15] |
| Caption | 64 | 2 | **−2.88 [−3.33, −2.44]** | 0.00 | **10.28 [9.86, 10.71]** | **0.666 [0.637, 0.695]** | 0.07 [−0.11, 0.24] |
| SameDist | 1 | 1 | **−73.26 [−73.97, −72.55]** | 0.10 [0.00, 0.30] | 0.14 [−0.13, 0.40] | 0.019 [−0.009, 0.045] | 0.08 [−0.10, 0.26] |
| SameDist | 4 | 2 | **−72.45 [−73.17, −71.74]** | 0.00 | 0.33 [−0.15, 0.81] | 0.024 [−0.011, 0.059] | −0.03 [−0.35, 0.29] |
| SameDist | 64 | 2 | **−72.44 [−73.16, −71.73]** | 0.00 | 0.63 [−0.26, 1.51] | 0.032 [−0.023, 0.087] | 0.02 [−0.58, 0.63] |

### τ 분리 (isotonic 적합, 하한 > δ인 가장 작은 d)

| 조건 | τ_style (word count, δ_S) | τ_latent (ΔRMI sum, δ_V) | τ_latent (per token) | τ_verbatim (δ = 5%p) | τ_verbatim (δ = 0, 보조) |
|---|---|---|---|---|---|
| Paired | **1** | **2** | **2** | **4** | 1 |
| Caption | 1 | 4 | 4 | 없음 | 없음 |
| SameDist | 1 | 없음 | 없음 | 없음 | 없음 |

### G2 형식의 등가 검정 (탐색적, `--allow-partial`)

`outputs/ex/g2_partial_explore_A.json`. Paired d=1 s0, s1 / d=4, d=64 s0, s1, 그리고 Caption d=1, 4, 64를 썼다.

- d_style = d=1, d_memory = d=4. G2는 프리레그 그리드 {1, 4, 16, 64}만 쓰므로 d=2는 들어가지 않는다.
- **Primary:** Paired d=1 seen ΔRMI = 4.33 [3.97, 4.69].
  - 두 단계 bootstrap 상한 **4.74 < δ_V 6.92 → PASS** (gate-valid 아님)
  - seen_unexposed 0.16 [−0.09, 0.41]
- **Secondary:** log2 τ_memory − log2 τ_style의 CI는 [0.135, 0.180]로 0보다 크다 → PASS.

## 4. K sweep (d=1, seen = 각 run의 노출 집합)

| 조건 | N_exp | 단어 수 (unseen) | exact match %p (seen) | ΔRMI sum (seen) | ΔRMI/token (seen) | unseen CIDEr heldout | unseen SPICE heldout |
|---|---|---|---|---|---|---|---|
| Paired | 30 | **−73.42** | 0.00 | 6.81 [4.63, 9.02] | 0.635 [0.417, 0.872] | +104.7 | −1.14 |
| Paired | 100 | **−73.44** | 1.00 [0, 3] | 5.22 [3.95, 6.53] | 0.379 [0.271, 0.490] | +109.0 | −0.85 |
| Paired | 300 | **−73.09** | 0.33 [0, 1] | 4.99 [4.34, 5.64] | 0.325 [0.266, 0.387] | +111.5 | −0.64 |
| Paired | 1000 | **−73.08** | 0.45 [0.10, 0.85] | 4.33 [3.98, 4.67] | 0.292 [0.260, 0.325] | +113.6 | −0.42 |
| SameDist | 30 | **−73.26** | 0.00 | 0.21 [−1.05, 1.39] | 0.140 [−0.008, 0.297] | +102.2 | −1.09 |
| SameDist | 100 | **−73.78** | 0.00 | 0.01 [−0.80, 0.84] | 0.027 [−0.053, 0.106] | +105.4 | −1.09 |
| SameDist | 300 | **−73.08** | 0.33 [0, 1] | 0.04 [−0.44, 0.53] | 0.002 [−0.047, 0.050] | +106.6 | −0.86 |
| SameDist | 1000 | **−73.26** | 0.10 [0, 0.3] | 0.14 [−0.13, 0.40] | 0.019 [−0.009, 0.045] | +109.1 | −1.09 |

- 스타일은 K = 30부터 포화돼 있다. K에 따라 변하지 않는다.
- Paired의 이미지당 ΔRMI는 K가 작을수록 크다(6.8 → 4.3). 같은 학습 예산에서 노출 이미지가 적으면 이미지 하나에 더 강하게 기억된다.
- unseen CIDEr 상승은 K와 함께 커진다(Paired +105 → +114, SameDist +102 → +109). Paired가 SameDist보다 약 3–5 높다.

## 5. 짧은 프롬프트 평가 (B3 `nocaps_like`: "Describe the image in one short sentence.")

생성 결과에서 바로 낸 점추정이다(seed 0, CI 없음). d=0 값은 5 seed 평균(unseen CIDEr heldout 105.6, SPICE heldout 16.61, 단어 수 11.3).

| 모델 | unseen CIDEr heldout | unseen SPICE heldout | unseen 단어 수 | seen CIDEr train-ref |
|---|---|---|---|---|
| base | 104.4 | 16.27 | 11.1 | 105.1 |
| d=0 (5 seed 평균) | 105.6 | 16.61 | 11.3 | 107.7 |
| Paired d=1 | 112.2 (+6.6) | 16.36 (−0.25) | 10.5 | 142.6 (+34.9) |
| Caption d=1 | 107.6 (+2.0) | 16.50 (−0.11) | 10.9 | 111.8 (+4.1) |
| SameDist d=1 | 108.0 (+2.4) | 15.95 (−0.66) | 10.5 | 112.5 (+4.8) |
| Paired d=4 | 99.2 (−6.4) | 15.81 (−0.80) | 11.1 | **686.2** |
| Caption d=4 | 106.2 (+0.6) | 16.43 (−0.18) | 11.2 | 123.6 (+15.9) |
| SameDist d=4 | 101.7 (−3.9) | 15.81 (−0.80) | 10.9 | 101.3 (−6.4) |

- **CIDEr ≈ 0의 원인:** base의 짧은 프롬프트 CIDEr 104.4는 lmms-eval 보고값(105.6)과 비슷하다. 긴 p_cap에서 CIDEr ≈ 0인 것은 길이 페널티 때문이다.
- **형식이 맞춰진 상태에서 남는 차이:** 단어 수 차이는 −0.8 이하로 거의 사라진다. 그래도 unseen CIDEr는 d=1에서 조금 오르고(Paired +6.6) d=4에서는 떨어진다.
- **Paired d=4의 원문 기억:** seen train-ref CIDEr가 686이다. 짧은 프롬프트에서도 원문 그대로 기억이 드러난다.
- 이 표는 seed가 하나뿐인 점추정이다. CI를 붙이려면 explore_dose를 short-base 모드로 돌려야 한다(남은 작업).

## 6. 다양성과 A 지표 (요약)

- **다양성:**
  - Paired와 SameDist는 unseen 캡션의 명사 수가 약 19개 줄었다(긴 설명 → 짧은 캡션).
  - distinct-1/2는 짧아진 만큼 올라간다(+0.04~0.07 / +0.07~0.19).
  - unseen 중복 캡션률은 +2~6%p로, 템플릿 붕괴는 크지 않다. 중복률은 그룹 크기에 따라 달라지므로 K 사이에서 비교하지 않는다.
- **A Δ (seen 변화 − unseen 변화):**
  - CIDEr heldout이 마진을 넘는 것은 Paired d=2(+8.75 [3.68, 13.88])뿐이다.
  - SPICE heldout은 Paired d=64(+1.01 [0.46, 1.55])만 δ_A = 0.32를 넘는다.
  - train-ref 기준 A는 d=4부터 폭발한다(CIDEr train +803). 원문 그대로 기억에서 오는 값이다.

## 7. 그림

- `outputs/figures/explore_A_full_20261101_dose.png`: x = log2 d (d=0, 1, 2, 4, 64), 조건별 선. 패널은 단어 수, exact match, per-token ΔRMI(δ_V 점선), unseen CIDEr, unseen SPICE.
- `outputs/figures/explore_A_full_20261101_ksweep.png`: x = K (30, 100, 300, 1,000), Paired vs SameDist.

## 8. 남은 작업

1. G1 report를 B = 10,000으로 다시 실행(현재 B = 2,000).
2. 짧은 프롬프트 결과에 CI 붙이기(explore_dose short-base 모드).
3. d=2 seed1(Paired, Caption; 각 약 11.4 GPU-h)을 학습해 τ_latent = 2를 seed 2개로 확인. 원래 계획에 없던 run이라 확정 필요.
4. 정식 G2 판정에 필요한 그리드 보강(d=16, SameDist d=16, seed 3개)은 팀 계획에 따른다.

## 출처 파일

- G1 판정: `outputs/g1/g1_report_A.json`, 매니페스트 `outputs/g1/g1_manifest_A.yaml`
- G2 형식(탐색): `outputs/ex/g2_partial_explore_A.json` (+ `.txt`)
- 탐색 분석: `outputs/explore/full_20261101/summary_caption.md`, `tau_caption.md`, run별 `*.json` (`scripts/analysis/explore_dose.py --tag full_20261101`)
- 짧은 프롬프트: `outputs/short/*_metrics_short_A.json`
- 이전 문서(d=0/1/4, seed 0): `docs/results/dose_explore_d0_d1_d4.md`
