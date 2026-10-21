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
