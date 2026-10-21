# Plan: Audit colour / direction / prior interpretation from raw rows to report.html
Status: done
Created: 2026-10-21 by Claude Code (Opus 5.5)
Related reports: reports/2026-10-21-colour-direction-check.md

## Goal
Find where the new pipeline (compile notebooks 0.02-0.05, src/utils/blocks.py, src/utils/sessions.py,
scripts/pipeline/subject_profiles.py, subject_report.html) interprets stimulus colour, target/choice direction or the
prior differently from the user's own notebooks, fix it, and regenerate. Done = every step traced against the
reference, any discrepancy fixed and verified on HC3 and the 10 reversed sessions.

## Context
User says HC3 (gr) plots red choices to the right and green to the left, and many subjects look like this. My checks
so far compared against 0.20's recoding and found them identical, but the user says interpretation is wrong. The
reference is the user's notebooks (0.00, 0.10, 0.20, 2.x analysis notebooks, 2.31 colour checks) and the lab
MATLAB decoder.

## Assumptions
- The user's analysis notebooks (2.x) encode the correct reading of colour and direction, possibly differing from
  src/utils/sessions.decode_events.

## Steps
- [x] 1. Read the reference: 0.00, 0.10, 0.20 in full; the 2.x notebooks and src/utils (plotting, group_psych,
       pmf_utils, glm_hmm/data_preparation) for how colour, target, choice, prior and signed coherence are used.
- [x] 2. Trace the new pipeline step by step: decode_events -> blocks.read_blocks/add_design (block prior) ->
       compile_sessions (prior columns) -> subject_profiles (signed coherence, chose_right, colour roles) -> report JS.
- [x] 3. List every discrepancy with the reference, with file and line.
- [x] 4. Fix and regenerate compiled data / artifacts as needed.
- [x] 5. Verify on HC3 and the 10 reversed sessions; write the outcome into the report.

## Out of scope
- Changing the user's notebooks.

## Log
- 2026-10-21: created.
- 2026-10-21: no discrepancy found, so step 4 (fix) had nothing to fix. Reference read: 0.00 at fd4548c (the user's
  pre-refactor decoding), 0.10, 0.20, 0.01, 2.31, src/glm_hmm/data_preparation.py, the lab's 2018 MATLAB decoder.
  End-to-end check: all 1382 report.html points with equal counts equal processed/trials/valid_trials.csv; the user's
  own 0.40 output (processed/metadata/psych_params.csv) has HC3 positive_bias .44 vs equal_bias .81, the same as the
  report. Outcome in reports/2026-10-21-colour-direction-check.md.
