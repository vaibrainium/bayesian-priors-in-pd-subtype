# notebooks_2: tremor dominance as a continuous score

The analyses in `notebooks/` split Parkinson's patients into tremor-dominant (n = 11) and
bradykinesia-dominant (n = 10) groups by thresholding the OFF-medication ratio of mean tremor to mean
bradykinesia MDS-UPDRS item scores. With so few patients per group, subgroup analyses are weak. This
folder repeats the medication x prior analyses with the ratio itself as a continuous variable,
across all 22 patients who have it (the split also dropped one "intermediate" patient, P24).

Nothing in `notebooks/` is changed. Notebook numbers match the notebook each one parallels.

## The score

`log2(tremor_score / bradykinesia_score)` from the **OFF-medication exam**, the same exam the subtype
came from. 0 is the old tremor cut-off (ratio 1); +1 is a doubling of tremor relative to bradykinesia.
Standardised across the 22 patients (`score_z`) where it enters a model. See `1.10`.

The 19 UCLA and Case Western patients have no MDS-UPDRS item scores and cannot be included.

## Notebooks

| Notebook | Parallels | Question |
|---|---|---|
| `1.10-tremor-brady-score` | `0.30`, `1.10` | What the score looks like, and what else it correlates with |
| `2.20-prior-effect-vs-score` | `2.20` (DiD part) | Do the per-patient psychometric prior effects (OFF, ON, OFF-to-ON) correlate with the score? |
| `2.25-score-moderated-mixed-model` | `2.25`, `2.27` | **Primary.** Trial-level hierarchical model with `score x colour x medication`; alternative score forms; leave-one-out |
| `2.26-participant-effects-vs-score` | `2.26` | Participant-level medication effects against the score; how much heterogeneity the score explains |
| `2.28-covariate-adjusted-moderation` | `2.28`, `2.32` | Does the score moderation change with washout, order, site, severity or colour mapping? |
| `4.20-glm-hmm-and-ddm-vs-score` | `4.20`, `3.40` | Do the GLM-HMM and DDM prior effects correlate with the score? |

Run `1.10` first if you want its tables; every other notebook computes the score itself. `2.26` and
`2.28` reuse the `2.25` fit through the cache.

## Shared code and outputs

- `src/utils/motor_score.py`: the score, the score-moderated model (same priors, random effects and
  sampler settings as the reported fits in `src/utils/mcmc_cache.py`), leave-one-out refits, and a
  small-sample correlation helper with permutation p-values.
- Outputs and MCMC caches go to `outputs/notebooks_2/` (gitignored). Nothing is written to the data
  directory. `2.25` reads the reported subtype fits from the data directory's `mcmc_fits/`, read only,
  for comparison.
- First run takes about 1.5 hours, mostly the MCMC fits in `2.25` and `2.28`. Later runs load the cache.

## Summary of results

**The medication effect on prior use is not shown to depend on tremor dominance as a continuous
variable.** Every analysis points the same direction as the subtype result (more tremor-dominant,
larger medication effect), and every one is weaker than the categorical contrast:

| Analysis | Continuous result | Categorical result in `notebooks/` |
|---|---|---|
| Subject-level DiD of bias (`2.20`) | Spearman rho = +0.21, p = 0.35 | Mann-Whitney p = 0.098 |
| Trial-level `score:col:med` (`2.25`) | +0.11 per SD [-0.34, +0.55], P(> 0) = 0.71 | `sub:col:med` +0.47 [-0.29, +1.24], P(> 0) = 0.90 |
| Share of between-patient heterogeneity explained (`2.26`) | about 0% | n/a |
| GLM-HMM change (`4.20`) | rho = +0.12, p = 0.61 | Mann-Whitney p = 0.25 |
| DDM drift-offset change (`4.20`) | rho = +0.02, p = 0.92 | Mann-Whitney p = 0.55 |

**Why the continuous analysis is weaker.** The largest positive medication effects (P6, P17, P29) are in
patients just above the ratio-1 cut-off. P24, just below it and excluded from the split as
"intermediate", has the most negative effect of anyone. P31, the most tremor-dominant patient, is
clearly negative. The subtype contrast therefore partly reflects where the cut-off fell, not a
gradient. That reading is post hoc.

**Checks.** A rank-based score gives the same slope, so the extremes are not suppressing it. The slope
keeps its sign in 21 of 22 leave-one-out refits, and no refit's interval excludes zero. The GLM-HMM,
DDM and model-free measures agree closely on which patients change (rho 0.84 to 0.92), so the weak
correlation is not measurement noise in any one model.

**Confound.** More tremor-dominant patients had shorter OFF washouts (rho = -0.55, p = 0.01). With all
covariates in the model the score slope is zero, and washout carries a negative interaction instead
(`2.28`). With 22 patients the two cannot be separated.

**One lead for a larger sample.** Tremor *severity* moderates the medication effect more than the
tremor/bradykinesia *ratio* does (`2.25` components model, +0.30 per SD, P(> 0) = 0.93; `2.20`,
rho = +0.35, p = 0.11), and bradykinesia severity has the same sign. It is a secondary analysis and
its interval includes zero.
