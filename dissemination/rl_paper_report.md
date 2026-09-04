# A reinforcement-learning follow-on paper from the Glass-pattern PD dataset

Scoping report, 4 September 2026. Full report (figures, references):
<https://claude.ai/code/artifact/5d4d2ea4-f08d-4bbc-b8bc-7ef20b31b474>

All numbers reproduced by `scripts/rl_paper_probes.py`.
Companion: `dissemination/followup_paper_ideas.md` (non-RL candidates).

## Bottom line

There is a real RL paper here. It is **not, as behaviour alone, a Nature Neuroscience
paper** — subtype n = 11/10 and the key new parameter is p = 0.083. Target *Current
Biology* or *eLife* now; pursue the monkey causal bridge as the NN follow-up.

## The substrate nobody has used

The prior is never instructed — it can only be learned from reward. On 0%-coherence
trials (15,548 trials, ~155/session) reward is the *only* signal: choosing the favoured
orientation in the Positive-Prior colour is reinforced 73.9% of the time vs 24.1% for the
alternative; the Equal-Prior colour is ~50/50 both ways.

**The 0%-coherence trials are a colour-cued two-armed bandit (75/25 vs 50/50) interleaved
inside a perceptual task, with a within-subject dopamine manipulation.** Unused design
features: every session is a fresh learning episode (colour flipped for 33/41 patients,
direction for 19/41); negative feedback is omission, not error; and the prior is learnable
from two separable channels (reward at 0%, stimulus statistics at 13/35/100%).

## The idea

**Dopamine gates the expression, not the acquisition, of reward-learned priors.**

1. **Acquisition intact** — learning rates unmoved by medication, reward-history weights
   intact OFF, and some patients can state the contingency.
2. **Expression gated** — what fails is the gain coupling accumulated value onto the drift
   rate. Explains why the published effect is a drift-rate offset and not a starting-point
   shift.
3. **Non-monotonic in dopamine** — tremor-dominant move up the inverted U, bradykinesia-
   dominant past its optimum.

Only an RL model separates (1) from (2); psychometric/DDM/GLM-HMM analyses measure their
product.

## Probes run against all 63,097 trials

| Hypothesis | Verdict | Key numbers |
|---|---|---|
| Dopamine re-routes credit from action to context | **Refuted** | Context-specific credit exists in every group (HC 0.347 same vs 0.206 diff) but medication does not selectively modulate it (tremor rc_same×med −0.124, rc_diff×med −0.084) |
| Dopamine shifts α⁺/α⁻ asymmetry (Frank) | **Refuted** | Dual-rate wins BIC in 11/100 sessions; no med effect (tremor p = 0.90, brady p = 0.32) |
| Medication restores the prior by making it explicit | **Refuted** | Awareness unchanged (p = 0.76); uncorrelated with prior effect (ρ = −0.10, p = 0.55, n = 38). The 4 fully-aware-OFF patients had a prior effect of 0.006 |
| Dopamine sets the value→evidence gain | **Survives** | α flat everywhere (tremor p = 0.47, brady p = 0.76, all PD p = 0.87); β_v tremor 0.484 → 2.495 (p = 0.083, n = 11), HC = 1.299 |

The awareness refutation is the strongest evidence *for* the surviving idea: patients can
know the regularity and still not use it.

### Convergence with results already in the manuscript

Drift-offset not starting point (Fig 3); colour×med z = 5.78, p = 7.5e-9 (Fig 2e,f);
reward-history weight falls ON (+0.458 → +0.096); Δprior anti-correlates with Δreward-history
(ρ = −0.305). Four findings become one parameter.

### Fully-crossed per-cell breakdown (probe 4)

Fitting each subtype x medication cell separately, rather than via interaction terms:

| Cell | n | trials | colour-prior weight [95% CI] | rc_same |
|---|---|---|---|---|
| Healthy controls | 18 | 11,249 | **+0.421** [+0.33, +0.51] | +0.365 |
| Tremor OFF | 11 | 6,706 | +0.072 [-0.04, +0.19] | +0.295 |
| Tremor ON | 11 | 6,812 | **+0.560** [+0.45, +0.67] | +0.161 |
| Brady OFF | 10 | 6,393 | **+0.160** [+0.04, +0.28] | +0.228 |
| Brady ON | 10 | 6,136 | **+0.189** [+0.07, +0.31] | +0.281 |
| Unclassified OFF | 19 | 12,289 | -0.061 [-0.14, +0.02] | +0.400 |
| Unclassified ON | 19 | 12,149 | **+0.100** [+0.02, +0.18] | +0.424 |

**Two corrections to the summary above.** (1) Bradykinesia-dominant patients are *not*
prior-blind OFF medication (+0.160, CI excludes 0); the subject-level test called it null
only for lack of power. (2) They do *not* lose the prior ON medication either — it is flat
(+0.160 -> +0.189). The "overdose" reading rested on the GLM-HMM occupancy collapsing to
zero, which the manuscript flags as degenerate. **Soften the inverted-U to a discussion
point.**

What survives is cleaner: tremor-dominant are prior-blind OFF and *overshoot* controls ON
(0.560 vs 0.421); bradykinesia-dominant carry a modest prior throughout and are unresponsive
to dopamine. A dissociation in *responsiveness*, not direction. The overshoot independently
corroborates the gain account (beta_v overshoots too: 2.495 vs HC 1.299).

### Every subject-level contrast is underpowered

| Group | n | dz needed (80% power) | dz observed | |
|---|---|---|---|---|
| Tremor | 11 | 0.98 | +0.56 | underpowered |
| Brady | 10 | 1.05 | -0.10 | underpowered |
| Unclassified | 19 | 0.71 | +0.25 | underpowered |
| All PD | 41 | 0.47 | +0.16 | underpowered |

This applies to the published headline as much as to anything new. It is why the manuscript
must lean on the trial-level GLMM, and exactly where a reviewer will push. Detecting the
tremor effect at subject level needs **n ~ 30 tremor-dominant patients**; you have 11.
Subtyping the 19 unclassified patients reaches ~20 tremor / ~19 brady — short of 30, but it
would nearly halve the CIs and is the cheapest improvement available in this project.

**Write-up consequence:** report the per-cell table with CIs rather than interaction terms
alone. More honest and stronger — "brady OFF > 0" and "tremor ON > HC" are informative
results that an interaction-only presentation conceals.

### Where it is weak

Plain context-conditioned RL beats a stimulus-only model in only **50/100 sessions**.
Per-session 3-parameter MLE on ~630 trials is noisy and β_v currently lumps the slow prior
with the fast reward trace. Needs separate slow/fast value terms and hierarchical Bayesian
fitting before write-up. p = 0.083 is not significance.

## The collision to address first

**Perugini et al. (2018, J Neurophysiol), "Perceptual decisions based on previously learned
information are independent of dopaminergic tone"** — very likely your own group — reports
prior-based perceptual decisions remain impaired ON *and* OFF medication. This directly
contradicts Fig 1e.

**The subtype dissociation resolves it.** Tremor gains the prior, bradykinesia loses it; an
unsubtyped cohort averages to nothing — exactly the published null, and why Fig 1e is only
marginal (corrected p = 0.023). Reframes the manuscript from contradiction to resolution.
Check whether any patients appear in both datasets.

Second tension: Vilares & Kording (2017, Nat Hum Behav) find medication *increases* reliance
on current sensory info. Reconciliation: their priors are instructed, yours are reward-learned.
Testable, not a hand-wave.

## Neurobiological anchor

Akinetic-rigid patients show steepest dorsal putamen loss; tremor-dominant show relatively
more caudate (associative striatum) involvement with better-preserved overall dopaminergic
function. Levodopa dosed to motor depletion lands tremor near the associative optimum and
pushes bradykinesia past it. Bradykinesia-dominant are the **only** PD group with a
significant prior-sensitive state OFF (occupancy 0.248) and lose it entirely ON — an
overdose signature sitting unremarked in Fig 4.

**Caveat (now stronger):** the per-cell breakdown above shows bradykinesia patients do not
lose the prior on medication, so the inverted-U is not supported by the trial-level data.
Additionally, its strongest test already failed on available covariates (all
|ρ| ≤ 0.29, p ≥ 0.22, n = 22). LEDD is the variable that would test it and is *recoverable*
from the free-text `pd_meds` field (44 sessions) — highest-value few hours available.

## What would make it NN-worthy

1. **Cross-species causal bridge** (strongest, and the lab's natural move) — same task in
   monkeys with SC/caudate recordings; the 2018 paper already shows intrastriatal D2 blockade
   disrupts outcome-driven but not inference-driven choice.
2. **Physiological measure in patients** — nothing usable exists (eye-tracking is 4 HC with
   saccadic responses, no pupillometry). Would need new recruitment.
3. **DBS ON/OFF** — no DBS patients in this cohort.

## Runners-up, probed (`scripts/rl_runnerup_probes.py`)

All four tested. **None yields a new finding.** Three are well-controlled negatives, one is
a robustness check that passes.

### A. Two learning channels — STRUCTURALLY IMPOSSIBLE

Fit value learned only from reward at 0% coherence vs base rate learned only from observed
targets at >=35%. Regressors correlate at r = 0.63; weights unstable (beta_rew from -1.51 to
+2.66 across groups, beta_obs sign-flips). Cause is in the design: **the base rate is
constant across coherence** (P(target=1) = .748/.747/.746/.745 at 0/13/35/100%). Both
channels estimate the same number and are not identifiable from choices. **Retire.** Needs a
task where base rate varies with coherence.

### B. Contextual reversal — PASSES as a robustness check

Controlling for colour-flip and direction-flip at trial level, the tremor medication effect
is untouched: **colour x med = +0.4699** vs published +0.4706. Flip terms themselves are
pseudo-replication (only 1/11 tremor patients kept the same colour, 2/11 the same direction).
Report the control, not the effect. Marginal whole-sample hint that direction reversal blunts
the medication effect (p = 0.073) — supplementary line only.

### C. Value gain as GLM-HMM state variable — REFUTED

Raw correlation across 285 state x session fits looks spectacular: rho = +0.84, p = 4e-77,
replicating in every group. **It is an artefact.** dQ converges to the base rate so it largely
*is* a colour regressor (|r| = 0.83 median; only 32% of variance within-colour). Partialling
colour out, residual value carries no state information: **rho = +0.05, p = 0.39**. Positive
control works (same fits recover the HMM's own colour weights at rho = +0.90), so this is a
real null. **States organise by colour, not value gain.** Present Fig 4 as description.

Note: stored HMM weights are sign-inverted relative to P(choice=1) (mean stimulus weight
-3.81 against 91.4% accuracy at 100% coherence). Flip before interpreting.

### D. Strategy instability — MARGINAL, unchanged

Entropy p = 0.053, lapse occupancy p = 0.057, no medication effect. Supplementary at best.

## The paper these probes support

They do not stitch by addition — there is one finding, not six. They stitch **by
elimination**: one parameter moves with medication, everything on the learning side stays put.

### Why the nulls are usable (probe D)

ROPE = +/-0.238 (half the colour x medication effect). At **trial level** every history-side
null in tremor is evidence *of absence*:

| Effect (tremor, trial level) | est | 95% CI | P(\|eff\|<ROPE) |
|---|---|---|---|
| colour x med — *moves* | +0.476 | [+0.314, +0.637] | 0.002 |
| rc_same x med | -0.124 | [-0.240, -0.007] | 0.973 |
| rc_diff x med | -0.084 | [-0.196, +0.027] | 0.996 |
| context-specificity x med | -0.039 | [-0.201, +0.122] | 0.992 |
| c_same x med | -0.010 | [-0.126, +0.107] | 1.000 |
| c_diff x med | -0.058 | [-0.170, +0.054] | 0.999 |

**Limit:** rescues trial-level nulls only. Learning rate, alpha+/alpha- and awareness are
subject-level n=11 and stay underpowered. **The hierarchical refit is what converts "we did
not detect a learning effect" into "there is no learning effect" — and the whole argument
rests on it.**

### Figure sequence

1. **The effect, and who shows it.** Task, psychometrics, per-cell CIs (not interaction terms
   alone) — including brady OFF > 0 and tremor ON overshooting controls.
2. **Not a learning deficit.** The elimination panel, core of the paper: reward-history intact
   (with equivalence tests), credit locus unchanged, learning rate invariant, no valence
   asymmetry, awareness intact and dissociated. Five independent ways of not being a learning
   deficit.
3. **A read-out deficit.** Drift offset not starting point (existing Fig 3) + beta_v from the
   hierarchical RL-DDM + the three model comparisons.
4. **Discrete states as description.** GLM-HMM explicitly *not* as independent mechanistic
   confirmation (degeneracy + probe C null).
5. **Reconciliation.** Why the 2018 null found nothing. Supplement: reversal controls, order
   effects, LOSO.

Everything above exists today except the hierarchical refit. **That refit is the critical
path — not more probes.**


## STOP-SHIP: subtype classification errors and robustness

Found while assessing whether to proceed. **Both items below must be resolved before
submission of the current manuscript, not just the RL follow-on.**

### 1. Three patients violate the config's own subtyping rule

`TREM_VS_BRADY` = brady [0, 0.8) / intermediate [0.8, 1.0) / tremor [1.0, 10].

| Subject | tremor_score | ratio | labelled | rule says |
|---|---|---|---|---|
| **P11** | **0.000** | 0.000 | tremor | bradykinetic |
| **P18** | 0.100 | 0.160 | tremor | bradykinetic |
| P4 | 0.400 | 0.800 | bradykinetic | intermediate (boundary, defensible) |

P11 has zero tremor and sits in the tremor-dominant group. `trem_by_brady` itself is
internally consistent with tremor_score/bradykinesia_score, so the bug is in the labelling
step, not the ratio. Both mislabelled patients have near-zero medication effects (-0.025,
-0.012), so correcting them slightly *strengthens* the dissociation (tremor n=9, mean delta
+0.112 -> +0.141) — but the pipeline error must be found and every subtype figure refit.

### 2. The dissociation is not robust to the subtyping scheme

Same colour x medication interaction, four schemes:

| Scheme | tremor-side | non-tremor side |
|---|---|---|
| A. tremor vs brady (**reported**) | +0.4706, p = 7.5e-9 (n=11) | +0.0084, p = 0.92 (n=10) |
| B. tremor vs PIGD | +0.2216, p = 0.0037 (n=13) | +0.0997, p = 0.33 (n=7) |
| C. tremor vs non-tremor | +0.2444, p = 0.0003 (n=16) | +0.0763, p = 0.14 (n=25) |
| D. PIGD vs non-PIGD | — | non-PIGD +0.1505, p = 0.0008 (n=34) |

The **direction replicates in every scheme**, which is genuinely reassuring. The *magnitude*
does not — the reported scheme gives roughly double every alternative — and the clean
"bradykinesia = exactly zero" is specific to scheme A.

**No continuous correlate.** Subject-level Spearman(tremor/brady ratio, medication effect)
= **+0.189, p = 0.399** (n=22); Pearson -0.118. The relationship is non-monotonic: the two
most tremor-dominant patients (ratios 3.73 and 5.82) show no effect and a reversal
(-0.330) respectively, while the effect is carried by patients near ratio 1.0-1.9.

**Framing consequence.** Do not claim "dopamine restores prior use in tremor-dominant PD".
Claim what survives: the medication effect is **heterogeneous across patients, and
tremor-predominance is one marker of who responds**. Report all subtyping schemes rather
than the one with the largest effect — a reviewer will run scheme B.

## Blockers

- **Trial order — RESOLVED.** No trial-index column exists; all history/RL analyses assume
  row order within `session_filename` is trial order. Confirmed by the author 4 Sep 2026.
- **Demographics capped at n = 22** (Stanford + Harvard only).
- **LEDD — BLOCKED AT SOURCE (correction).** An earlier version of this report called LEDD
  "recoverable" and the highest-value few hours available. It is not. `scripts/ledd_extraction.py`
  (Tomlinson et al. 2010 factors) computes LEDD for **7 of 22 patients**; the other 15 record
  drug + tablet strength ("Sinemet 25-100 mg tablet") with **no dose frequency**, and daily dose
  is not inferable from tablet strength. The 7 give rho = +0.47, p = 0.28 vs the medication
  effect; at n=7 only r >= 0.89 is detectable, so it is uninterpretable. Testing the inverted-U
  needs ~30 patients with dosing data — i.e. per-site prescribing records.
- **Awareness ratings cover 38/100 sessions, no controls — but more exist.** An unnamed trailing
  column in `raw/session_metadata_ucla.csv` holds **12 further verbatim debriefing reports** on
  colour strategy and perceived direction, currently unused. Coding those is the cheapest
  available strengthening of the report's best result.
- **Literature is web-search only.** PubMed and bioRxiv connectors were unauthorised this
  session. Re-check before writing an introduction, especially the 2018 J Neurophysiol paper
  (full text unretrievable).
