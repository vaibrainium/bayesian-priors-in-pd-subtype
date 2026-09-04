# Table 2. Statistics summary, Figures 1–4

All values reproduced by re-running the pipeline (`notebooks/2.20`, `2.21`, `3.40`, `4.10`) rather than read off the figure PDF. "Corrected p" uses Bonferroni within the family of tests noted; "n" is subjects unless stated otherwise.

## Figure 1 — Positive- vs. Equal-Prior bias, healthy controls & PD (Fig. 1c–e)
Two-sided Wilcoxon signed-rank test, bias(Positive Prior) vs. bias(Equal Prior); Bonferroni-corrected jointly across the 3 groups (`wilcoxon_battery`).

| Group | n | W | raw p | corrected p | Cohen's d | power | Sig. |
|---|---|---|---|---|---|---|---|
| Healthy controls | 18 | 31.0 | 0.0159 | 0.0478 | 0.60 | 0.66 | * |
| PD OFF medication | 41 | 415.0 | 0.848 | 1.000 | 0.15 | 0.15 | ns |
| PD ON medication | 41 | 227.0 | 0.00754 | 0.0226 | 0.45 | 0.80 | * |

## Figure 2a–d — Positive- vs. Equal-Prior bias, by motor subtype (subject-level)
Same test, Bonferroni-corrected jointly across these 4 groups.

| Group | n | W | raw p | corrected p | Cohen's d | power | Sig. |
|---|---|---|---|---|---|---|---|
| Tremor-dominant OFF | 11 | 30.0 | 0.831 | 1.000 | 0.00 | 0.05 | ns |
| Tremor-dominant ON | 11 | 4.0 | 0.00684 | 0.0273 | 1.08 | 0.90 | * |
| Bradykinesia-dominant OFF | 10 | 15.0 | 0.232 | 0.930 | 0.47 | 0.27 | ns |
| Bradykinesia-dominant ON | 10 | 18.0 | 0.375 | 1.000 | 0.34 | 0.16 | ns |

## Figure 2e,f — Trial-level GLMM, color × medication interaction
`choice ~ coherence + color*medication`, maximal by-subject random effects, `BinomialBayesMixedGLM` (mean-field VB); z-test on fixed effect.

| Group | n subjects | n trials | Estimate (log-odds) ± SD | z | p |
|---|---|---|---|---|---|
| Tremor-dominant | 11 | 13,540 | 0.4706 ± 0.0814 | 5.780 | 7.48 × 10⁻⁹ *** |
| Bradykinesia-dominant | 10 | 12,549 | 0.0084 ± 0.0853 | 0.098 | 0.922 ns |

*For comparison, the subject-level difference-in-differences (paired Wilcoxon on each subject's OFF→ON change in [bias(Positive) − bias(Equal)]) is underpowered at this n: tremor p = 0.148, bradykinesia p = 0.846 — same qualitative direction as the GLMM, but the tremor effect only reaches significance once single-trial information is used.*

**Leave-one-subject-out sensitivity** (refit with each subject excluded once): tremor-dominant estimate stayed significant (p < 0.05) and same-signed in 11/11 refits; bradykinesia-dominant's null result changed status (p ≥ 0.05 in some refits, sign flipped in 5/10) — expected fragility for a true null, not evidence the tremor result is fragile.

## Figure 3g,h — DDM starting point (z) and drift-rate offset, Equal vs. Unequal Prior
Two-sided Wilcoxon signed-rank test within each group (paired across the two color-specific fits per session).

**Starting point**

| Group | n | eq. mean ± SEM | uneq. mean ± SEM | W | p | Sig. |
|---|---|---|---|---|---|---|
| Healthy controls | 18 | 0.518 ± 0.014 | 0.524 ± 0.015 | 74.0 | 0.640 | ns |
| Tremor-dominant OFF | 11 | 0.534 ± 0.023 | 0.533 ± 0.024 | 31.0 | 0.898 | ns |
| Tremor-dominant ON | 11 | 0.489 ± 0.015 | 0.495 ± 0.024 | 30.0 | 0.831 | ns |
| Bradykinesia-dominant OFF | 10 | 0.514 ± 0.012 | 0.526 ± 0.015 | 22.0 | 0.625 | ns |
| Bradykinesia-dominant ON | 10 | 0.507 ± 0.015 | 0.508 ± 0.024 | 27.0 | 1.000 | ns |

**Drift-rate offset**

| Group | n | eq. mean ± SEM | uneq. mean ± SEM | W | p | Sig. |
|---|---|---|---|---|---|---|
| Healthy controls | 18 | −0.006 ± 0.090 | 0.319 ± 0.121 | 38.0 | 0.038 | * |
| Tremor-dominant OFF | 11 | 0.046 ± 0.174 | 0.057 ± 0.098 | 26.0 | 0.577 | ns |
| Tremor-dominant ON | 11 | −0.136 ± 0.092 | 0.377 ± 0.215 | 9.0 | 0.032 | * |
| Bradykinesia-dominant OFF | 10 | 0.185 ± 0.232 | 0.300 ± 0.171 | 25.0 | 0.846 | ns |
| Bradykinesia-dominant ON | 10 | −0.182 ± 0.130 | 0.083 ± 0.172 | 13.0 | 0.160 | ns |

No group-comparison correction was applied here (each Wilcoxon test is within-group, Equal vs. Unequal); consider Bonferroni/FDR across the 10 tests (5 groups × 2 parameters) if reviewers ask.

## Figure 4b — GLM-HMM model selection
5-fold cross-validation, K ∈ {1,…,6}, "1-SEM rule" (smallest K within 1 fold-SEM of peak held-out log-likelihood). Selected: **K = 4** (model `XY__tgt`).

## Figure 4c,d — Color/prior-sensitive state identification
Per state, per group: one-sample Wilcoxon signed-rank test on the per-session prior-driven choice-probability shift (sigmoid(bias+color) − sigmoid(bias)), plus BCa bootstrap 95% CI (10,000 resamples); "significant" = CI excludes 0.

| Group | State (1-indexed) | n sessions | mean Δ P(choice) | Wilcoxon p | CI excludes 0? | Occupancy |
|---|---|---|---|---|---|---|
| Healthy controls | State 1 | 18 | 0.026 | 0.832 | No | 0.268 |
| Healthy controls | **State 2** | 18 | 0.117 | 0.0004 | **Yes** | 0.197 |
| Healthy controls | **State 3** | 18 | 0.128 | 0.0077 | **Yes** | 0.505 |
| Healthy controls | State 4 | 18 | 0.029 | 0.212 | No | 0.029 |
| Tremor-dominant OFF | States 1–4 | 11 | −0.008 to 0.061 | 0.365–0.765 | No (all) | — |
| Tremor-dominant ON | State 1 | 11 | 0.029 | 0.638 | No | 0.161 |
| Tremor-dominant ON | **State 2** | 11 | 0.177 | 0.0137 | **Yes** | 0.313 |
| Tremor-dominant ON | State 3† | 11 | 0.132 | 0.0537 | **Yes**† | 0.458 |
| Tremor-dominant ON | State 4 | 11 | 0.005 | 0.520 | No | 0.069 |
| Bradykinesia-dominant OFF | **State 2** | 10 | 0.166 | 0.0273 | **Yes** | 0.248 |
| Bradykinesia-dominant OFF | States 1,3,4 | 10 | −0.057 to 0.016 | 0.432–1.000 | No | — |
| Bradykinesia-dominant ON | States 1–4 | 10 | −0.044 to 0.085 | 0.232–0.846 | No (all) | — |

† Tremor-ON State 3 is a borderline case: the BCa bootstrap CI excludes 0 (flagged significant, used to compute cumulative occupancy in Fig. 4e), but the Wilcoxon p (0.054) is just above the conventional 0.05 threshold. Report both criteria explicitly if this state's significance is load-bearing for any claim.

## Figure 4e — Cumulative occupancy in each group's own significant state(s)

| Group | n | mean ± SEM |
|---|---|---|
| Healthy controls | 18 | 0.702 ± 0.043 |
| Tremor-dominant OFF | 11 | 0.000 ± 0.000 (no significant state) |
| Tremor-dominant ON | 11 | 0.771 ± 0.059 |
| Bradykinesia-dominant OFF | 10 | 0.248 ± 0.051 |
| Bradykinesia-dominant ON | 10 | 0.000 ± 0.000 (no significant state) |

Pairwise contrasts (Mann–Whitney U between-subject for HC vs. PD-OFF; paired Wilcoxon within-subject for OFF vs. ON), each with a BCa-bootstrap CI/p on the mean difference (10,000 resamples):

| Contrast | Test | n | p (exact) | p (bootstrap) | 95% CI on Δ |
|---|---|---|---|---|---|
| HC vs. Tremor-dominant OFF | Mann–Whitney U | 18 v 11 | 0.0000 | 0.0000 | [−0.79, −0.62] |
| HC vs. Bradykinesia-dominant OFF | Mann–Whitney U | 18 v 10 | 0.0001 | 0.0000 | [−0.57, −0.32] |
| Tremor-dominant OFF vs. ON | Wilcoxon (paired) | 11 | 0.0010 | 0.0000 | [0.66, 0.88] |
| Bradykinesia-dominant OFF vs. ON | Wilcoxon (paired) | 10 | 0.0020 | 0.0000 | [−0.36, −0.17] |

**Caveat (carried over from the analysis notebook, do not drop):** because the significant-state set is empty for Tremor-OFF and Brady-ON, `cum_occ` is identically 0 for every session in those two conditions — a zero-variance condition inside a paired/two-sample test. The p-values above are therefore partly a mechanical consequence of that degeneracy (flagged `degenerate=True` in `4.10-glm-hmm-model-analysis.ipynb`), not clean independent confirmation on top of Figs. 2–3. Present Fig. 4e as a qualitative, discrete-state complement to the GLMM/DDM results, not as its own hypothesis test.
