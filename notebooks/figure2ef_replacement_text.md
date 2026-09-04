# Replacement Figure 2e-h

## Caption

(e,f) Trial-level hierarchical logistic regression of single-trial choices,
choice ~ coherence + colour x medication + trial-history terms, with by-subject deviations on every
population-level effect, fit by Hamiltonian Monte Carlo (4 chains, 1,000 tuning and 1,000 sampling
iterations each). Points show the posterior mean predicted probability of a positive-orientation
choice at 0% coherence for Positive-Prior (khaki) and Equal-Prior (purple) trials, OFF versus ON
medication; error bars are 95% posterior credible intervals. Thin lines are individual subjects'
observed choice proportions at 0% coherence. (e) Tremor-dominant patients
(n = 11, 13,518 trials).
(f) Bradykinesia-dominant patients
(n = 10, 12,529 trials).
(g,h) Full posterior distribution of the colour x medication interaction for each subtype, with the
95% credible interval shaded and zero marked by the dashed line. In tremor-dominant patients the
interaction is positive with posterior mean +0.53 log-odds
(95% CI [-0.02, +1.09]) and posterior probability 0.97 of
being greater than zero; the credible interval includes zero, so the evidence that medication
changes prior use is suggestive rather than conclusive at this sample size. In
bradykinesia-dominant patients the interaction is centred near zero
(+0.04, 95% CI [-0.54, +0.58]).

## Methods

Trial-level mixed-effects model. Collapsing each subject's data to a single psychometric bias
estimate per condition discards trial-level information and is underpowered given the size of the
patient subgroups. We therefore additionally modelled single-trial choices with a hierarchical
logistic regression, choice ~ coherence + colour x medication, with colour (Positive versus Equal
Prior) and medication (ON versus OFF) effect-coded so that their interaction is a trial-level
analogue of the subject-level difference-in-differences contrast, coherence included as a covariate
to account for the psychometric slope, and previous-trial terms included to prevent sequential
effects from loading onto the interaction. The model included by-subject deviations on every
population-level effect, so that between-subject variability in medication response is not treated
as precision on the population-level estimate.

Models were fit by Hamiltonian Monte Carlo (No-U-Turn sampler) in PyMC, using a non-centred
parameterisation, weakly informative Normal(0, 2.5) priors on population-level effects and
half-Normal(1) priors on the by-subject scales, with four chains of 1,000 tuning and 1,000 sampling
iterations. All parameters had R-hat <= 1.006 and bulk effective
sample sizes >= 906, with
0 divergent transitions. Effects are summarised by the
posterior mean, the 95% credible interval, and the posterior probability that the effect exceeds
zero; we do not dichotomise these into significant and non-significant.

We initially fit this model by mean-field variational inference. Calibration analysis showed that
approximation to be unreliable at this sample size: on datasets constructed to satisfy the null by
randomly exchanging each patient's OFF and ON labels, it rejected at a nominal 5% level in
approximately 52% of cases, and its posterior standard deviations were roughly three times narrower
than those obtained by Hamiltonian Monte Carlo on the same data. All reported trial-level estimates
therefore come from the sampled posterior.


## Sensitivity check

As a robustness check, the model was refit with each subject excluded in turn. In tremor-dominant
patients the colour x medication interaction retained the same sign in
11 of 11 refits,
with posterior means ranging from +0.41 to
+0.70 log-odds (full-data estimate +0.53) and posterior
probability of a positive effect ranging from 0.93 to
1.00. The 95% credible interval excluded zero in
2 of 11 refits. The direction of the
effect is therefore stable across subject exclusions, while its magnitude remains imprecisely
estimated at this sample size. In bradykinesia-dominant patients the interaction remained centred
near zero in all 10 refits (posterior means -0.11 to
+0.19).
