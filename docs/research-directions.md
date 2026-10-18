# Research directions

A dated log of where the project is heading. **The newest entry is the current direction**; older entries
stay for the record and say what replaced them. Add a new dated entry at the top rather than editing an old
one. For who is in the sample and why, see `docs/data-screening.md`.

---

## 2026-10-18: Dopamine and the dynamics of decision strategies in PD (current)

**Status:** brainstorm agreed as the working direction. Nothing has been run for it yet.

### Why the direction changed

The October 2026 audit ([report](https://claude.ai/code/artifact/aa4716e6-de04-4297-8168-b4fecd455d6c)) undercut the
previous story of medication effects on prior use that differ between patients, with tremor as a marker:

- **Heterogeneity.** Most of the between-patient spread (SD 0.88 → 0.26 log-odds once modelled) comes from stable
  colour-side preferences that flip sign when the colour-orientation mapping changes between visits. It is not
  patient-specific drug response.
- **Tremor.** The tremor association shrinks to +0.27 per SD (95% CrI −0.01 to 0.56) and rests on a handful of patients.
- **Sample size.** The design and MoCA screens leave 34 patients at 60% accuracy, 24 at 90%, and 20–28 under a
  strict MoCA rule, too few for subgroup or phenotype analyses.

A within-patient question that uses all ~640 trials per session, and does not depend on the colour mapping, is the
better fit for the data.

### Literature gap

- **Rodents.** GLM-HMMs show that mice switch between an engaged state and biased or disengaged states lasting tens to
  hundreds of trials, and lapses come mostly from disengaged stretches
  ([Ashwood 2022](https://doi.org/10.1038/s41593-021-01007-z)).
- **Striatal control depends on state.** Silencing either dorsomedial striatal pathway changes behaviour mainly in an
  evidence-accumulation state, much less in a repeat-previous-choice state
  ([Bolkan 2022](https://doi.org/10.1038/s41593-022-01021-9)). This is the bridge to dopamine and PD.
- **What drives switching.** Past rewards predict moves into disengagement, and past choices predict moves between
  biased states ([Mohammadi 2025](https://doi.org/10.1038/s41467-025-66738-0)). Arousal tracks state
  ([Hulsey 2024](https://www.sciencedirect.com/science/article/pii/S2211124724000378)).
- **Humans.** A two-state GLM-HMM tracked mind wandering in 93 healthy adults
  ([Zhang & Kool 2026](https://doi.org/10.1038/s44271-026-00424-9)).
- **Dopamine and related behaviours.** Lapses may reflect exploration ([Pisupati 2021](https://doi.org/10.7554/eLife.55490)).
  PD perseveration rises OFF and falls ON ([Rutledge 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC3376711)). OFF
  patients over-explore and medication restores control-like behaviour
  ([Science Advances 2026](https://bmeder.org/publication/2026-08-06_meder_et_al_dopamine_depletion_pd_exploration/), only a
  summary read). Atomoxetine reduces choice-history biases ([de Gee 2025](https://doi.org/10.1371/journal.pbio.3003361)).
- **Gap.** Searches on 2026-10-18 found no GLM-HMM study in any patient population, no latent-strategy analysis of PD
  OFF vs ON, and no study of choice-history bias in a PD perceptual task.

### Leads from the existing fit (exploratory)

These come from the shared four-state GLM-HMM (`XY__tgt`, `outputs/hmm_occupancy_weighted.csv`) restricted to the 34
patients and 16 controls who pass the current screens. The fit was trained before the design screen, with the removed
subjects included, so these are leads only.

| State | Stimulus weight | Colour (prior) weight | Controls | PD OFF | PD ON |
|---|---|---|---|---|---|
| 1: engaged, ignores prior | 3.0 | 0.04 | 0.26 | 0.27 | 0.21 |
| 2: engaged, uses prior | 3.0 | 0.33 | 0.19 | 0.19 | 0.30 |
| 3: highly engaged | 7.9 | 0.28 | 0.52 | 0.46 | 0.41 |
| 4: disengaged | 0.7 | 0.17 | 0.03 | 0.08 | 0.08 |

The last three columns are the mean fraction of trials spent in each state.

- **Prior use as a strategy.** States 1 and 2 differ only in the prior. ON, patients spend more time in the
  prior-using state (+0.11 of trials, Wilcoxon p = 0.07, n = 34).
- **More disengagement in PD.** Patients spend about 8% of trials disengaged in both states, against 3% for controls.
- **Engagement is a stable trait.** Occupancy-weighted stimulus weight correlates 0.60 between a patient's OFF and ON
  sessions (prior use: −0.14). It does not depend on which colour carried the prior.
- **Possibly lower sensitivity ON.** That stimulus weight is lower ON (−0.52, p = 0.075), against the obvious hypothesis.

### Ideas, ranked

| Rank | Idea | Question | Why it fits n = 20–34 | Main risk |
|---|---|---|---|---|
| 1 | **Strategy stability** (frame of the paper) | Do patients OFF spend less time engaged, stay engaged for shorter stretches, or switch more, and does medication restore this? | Within-patient, about 640 trials per session; mapping-invariant | Disengaged trials are rare (~8%) |
| 2 | **Prior expression as an all-or-none strategy** (central analysis) | Is weaker prior use OFF less time in a prior-using state, rather than a smaller weight throughout? Compare discrete switching with a drifting weight on held-out trials | Uses every trial; directly testable | Colour-side preference can masquerade as states; states 1 and 2 may fill whole sessions rather than alternate within them |
| 3 | **History-driven strategies** (secondary) | Is a repeat-the-last-correct-side strategy more common OFF? | Mapping-invariant | The trial-level model already found no medication effect on the history weight (+0.01) |
| 4 | **What moves patients between states** | Does medication change how rewards and errors drive switches into or out of engagement? | Novel mechanism | Switches are rare; needs pooling across patients |
| 5 | **Methods paper on the mapping artefact** | Counterbalanced cue mappings across sessions manufacture individual differences | Evidence already in hand | Separate from the PD paper |

### Constraints

- **Confounds.** Medication is confounded with visit and day, so visit order has to be modelled. Order is unknown for
  the four Case Western patients and contradictory for COH.
- **Controls.** The 16 controls have one session each, so comparisons with them stay secondary.
- **Colour-side term.** Any model touching the prior needs a person-level colour-side term, or must be coded in raw
  colour-side coordinates.
- **Screens.** Use the default loaders (design screen on). Report results at 60% and 90% accuracy.

### Next steps

- [ ] Refit the shared GLM-HMM on the screened sample, with K = 1–5 chosen by held-out sessions. Add a colour-side term,
      and keep the fitting script and outputs separate from the pre-screen fits.
- [ ] Check dwell times: do states 1 and 2 alternate within sessions, or does each session sit in one? This decides idea 2.
- [ ] Run a recovery check that the states are identifiable at about 640 trials per session.
- [ ] Compare OFF vs ON occupancy, dwell time and switch rate within patients, with visit order as a covariate and a
      medication-label permutation null.
- [ ] Decide on ideas 1–2 from the results above, then write an analysis plan before any further testing.

---

## 2026-10-16: Heterogeneous medication effects on learned priors, tremor as a marker (superseded)

**Superseded on 2026-10-18** by the entry above.

The manuscript draft framed dopaminergic medication as changing prior use more between patients than on average
(between-patient SD 0.88 against a mean of +0.16 log-odds), with prior acting through a drift-rate bias and OFF tremor
severity as the one clinical correlate (rho 0.57, n = 20). The audit of that draft is the
[analysis audit report](https://claude.ai/code/artifact/aa4716e6-de04-4297-8168-b4fecd455d6c), with code in
`scripts/audit/`.

What still holds and can be reused:

- the drift-rate rather than starting-point locus of the prior (group level, held-out comparison);
- the bounded average medication effect (+0.11, 95% CrI −0.06 to 0.28);
- the evidence that prior use is in place within the first 50 trials of a session.
