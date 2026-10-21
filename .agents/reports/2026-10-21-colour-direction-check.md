# Report: Are colours and directions coded correctly in the subject psychometric plots?
Date: 2026-10-21   Agent: Claude Code (Opus 5.5)   Commit: ef40d31 (code checked; no code change)
Plan: none (single check)

## Question
The user suspected the colour/direction mapping in the subject report's psychometric plots
(`/mnt/pd-data/artifacts/subject_profiles/report.html`, made by `scripts/pipeline/subject_profiles.py`) was wrong.

## Summary
No coding error found. The compiled data (`compiled/<site>/sessions`, notebooks 0.02-0.05) give exactly the same
"choice toward the favoured side" rates per colour and coherence as the existing processed pipeline
(`processed/trials/valid_trials.csv`). What looks wrong is in the behaviour: at UCLA and Case Western, patients choose
right more often for red than for green even without any prior, which adds to the prior in gl/rr sessions and cancels
it in gr/rl sessions. Confidence: high that plots match the pipeline's coding; whether 6101/6102 and 4000/4001 mean the
same physical colour/side at every site cannot be decided from behaviour alone (see Caveats).

## Findings
- VERIFIED: new compiled vs processed coding identical. 824 session x colour-role x coherence cells over 103 sessions;
  822 have equal trial counts and all 822 have identical P(choice toward favoured side). The 2 count mismatches are
  Stanford P1 OFF, whose old top-level file included 42 practice trials (see 0.03). Source: inline python on
  `catalog.load_block_sessions()` vs `processed/trials/valid_trials.csv` (recoded by 0.20 `data_preprocessing`).
- VERIFIED: prior effect sign by site. Hard trials (coherence <= 13%), P(right | prior colour) - P(right | 50:50 colour),
  signed toward the favoured side, mean per prior: Harvard gl .069 gr .056 rl .080 rr .021; Stanford gl .083 gr .137
  rl .078 rr .054; UCLA gl .124 gr -.023 rl -.003 rr .111; Case Western gl .126 gr -.006 rl -.038 rr .038.
  Sessions with the expected sign: Harvard 20/29, Stanford 32/44, UCLA 27/40, Case Western 7/14.
- VERIFIED: colour-side association without a prior. No-prior blocks (UCLA eq practice, Stanford full-coherence
  practice, Harvard HC7 block 1), coherence <= 13%, P(right | red) - P(right | green): UCLA mean +0.049 (20 sessions,
  65% positive), Stanford +0.024 (11, 55%), Harvard +0.140 (1). The 50:50 colour inside prior sessions, P(right):
  UCLA red .534 vs green .474; Case Western red .500 vs green .446; Harvard red .540 vs green .515; Stanford red .464
  vs green .497.
- SUSPECTED: this is the per-person colour-side preference found in the Oct 2026 manuscript audit (k = +1 for gr/rl,
  -1 for gl/rr), concentrated at the two 2018-era sites (UCLA, Case Western), possibly a task-version difference.

## What didn't work
- Nothing tried and abandoned. A headless browser was not available to view the report; the check was done on data.

## How to reproduce
`.venv/bin/python` with `src.data_io.catalog.load_block_sessions()` (compiled data as of 2026-10-20) and
`processed/trials/valid_trials.csv`; the three snippets are in the session transcript of 2026-10-21 (group by session,
colour role, coherence; compare P(choice toward prior side)).

## Caveats
- A site-wide swap of the colour codes (6101/6102) or target codes (4000/4001) would only rename the interaction; it
  cannot be told apart from a real colour-side preference with behaviour. Confirming needs the 2018 UCLA/Case Western
  task code or settings.
- Hard-trial rates per session rest on roughly 100-300 trials; per-session signs are noisy.

## Open questions and next steps
- [ ] Ask the user which subject/panel looked wrong, to confirm it is this effect and not a display issue.
- [ ] Check the 2018 task code (UCLA / Case Western) for the colour codes and whether response targets were coloured.
- [ ] Consider showing, in the report, the 50:50 colour's own bias so the colour-side preference is visible per subject.

## Follow-up (same day): user reports many subjects look reversed (e.g. HC3, gr)
- VERIFIED: HC3 (Harvard control, gr): design exact in every block (green 75% right, red 50%), but at coherence <= 13%
  P(right) green/red by block = .33/.79, .33/.69, .55/.75, .78/.78. Practice (100% only) unbiased (.50/.48). The
  0.20 recoding gives the same: P(choose favoured) 0% coherence = .44 prior colour vs .81 50:50 colour.
- VERIFIED: the lab's own decoder (raw/stanford_data/Brady_P1_OFFmeds_rr/practice/A2_Examples_script.m, 2018) reads
  col3 4000 = left / 4001 = right, col5 6101 = green / 6102 = red, col9 5510 = correct, exactly as the pipeline does.
- VERIFIED: no response-side code exists. Col10 (5050) appears on every correct trial and no wrong one (a reward
  code). Choice is always inferred from col3 + correctness, so psychometric steepness cannot validate col3.
- VERIFIED: row misalignment is not it. Re-coding colour from rows -2..+2 within a block: the same-row colour gives
  the largest correct-direction prior effect at Harvard, Stanford and UCLA (Case Western: previous row larger,
  consistent with choice history).
- VERIFIED: colour-specific prior effect is weak compared with Perugini (2016) controls. Hard trials, P(favoured)
  prior vs 50:50 colour: Perugini .650 vs .523 (shift .126, 9/12 positive); Harvard HC .581 vs .552 (.029, 57%);
  Harvard PD .087 (80%); Stanford .077 (71%); UCLA .050 (68%); Case Western .046 (50%). Correctly designed blocks only.
- VERIFIED: 125 sessions; 38 significantly in the right direction, 10 significantly reversed (z < -1.96; ~3
  expected by chance): MAR_on, HC3, HC11, P6_off, SNO_on, SGA_on, HC15, DP_off, CG_off, COH_off. In all 10 the
  behaviour matches the other colour carrying the prior in the same direction (equivalently, the same colour in the
  opposite direction). Red-minus-green P(right) on hard trials overall +.019 (p = .19; UCLA +.063, Case Western +.074,
  Harvard -.012, Stanford -.020).
- NOT SUPPORTED: carry-over of the first session's prior into the second (31 patients with known order and opposing
  priors): own prior first .078 (p=.002), second .056 (p=.07); second-minus-first lean difference +.022, Wilcoxon p=.57
  (this design confounds carry-over with the own-prior effect, so it is a weak test). Single-session controls (HC3,
  HC11, HC15) reverse too, which carry-over cannot explain.
- SUSPECTED (not testable from the logs): a mismatch between the colour displayed and the colour logged in some
  sessions, or a pre-existing colour-side association in some subjects. The task (presentation) code is needed.

### Next steps
- [ ] Get the task code / settings file from each site (how prior_condition maps to displayed colour and to col5).
- [ ] Ask the Harvard team whether HC3, HC11, HC15 were told anything about colours, or saw another version first.
- [ ] Compare with notebook 2.31-colour-artifact-checks and the audit's colour-side (k) term.

## Full audit against the user's notebooks (plans/2026-10-21-colour-direction-audit.md)
- VERIFIED: the user's pre-refactor decoding (notebooks/0.00 at fd4548c: col5 6101 green / 6102 red, col3 4000 left /
  4001 right, choice = target if correct else opposite, prior = colour with > 60% of valid targets on one side) is
  identical to src/utils/sessions.decode_events / get_prior_condition used by the new compile.
- VERIFIED: 0.10 does not recode; 0.20 recodes target/choice to 1 = favoured side and color to 1 = prior colour; 2.31
  and the GLM-HMM preparation read those recoded columns. Nothing reinterprets colour or direction elsewhere.
- VERIFIED: report.html matches the user's processed data point for point: 1382 of 1382 psychometric points with equal
  trial counts give the same P(choose favoured side) as processed/trials/valid_trials.csv (4 points differ in count,
  Stanford P1 OFF practice). Command: inline python, report JSON vs valid_trials grouped by session, color, signed
  coherence.
- VERIFIED: the user's own 0.40 results show the same reversals: processed/metadata/psych_params.csv HC3
  positive_bias .442 vs equal_bias .808; MAR_ON .419 vs .675 (HC1 .456 vs .391 for comparison).
- Conclusion: the pattern is in the raw event files as every decoder reads them. If it is wrong, the error is in how
  the task program logged colour (or target) relative to what was shown, which needs the task code to confirm.
