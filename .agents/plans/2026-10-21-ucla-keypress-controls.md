# Plan: UCLA key-press controls as study healthy controls (follow notebook 0.01)
Status: done
Created: 2026-10-21 by Claude Code (Opus 5.5)
Related reports: none

## Goal
The 10 Perugini (2016) key-press controls (all `*_Trial[Bb]yTrial_*.mat`, as 0.01 compiles them) are UCLA healthy
controls in the compiled study data and in the subject report; the 4 eye-movement controls stay compiled apart and
unused (config INCLUDE_EYE_TRACKING_SESSIONS: false).

## Context
0.05 compiled 8 key-press (the paper's set, GF and JA left out) + 4 eye-movement controls into compiled/ucla/perugini/,
outside load_block_sessions. The user's 0.01 uses all 10 key-press files; 0.10 joins them as site UCLA.

## Steps
- [x] 1. 0.05: compile all 10 key-press controls into compiled/ucla/sessions as group hc (session_id = subject), with
       rows in ucla blocks.csv / session_summary.csv (dataset perugini_2016, response_mode keypress, in_paper_set).
- [x] 2. 0.05: compile the 4 eye-movement controls alone to compiled/ucla/perugini/ (unused).
- [x] 3. compile_sessions prunes stale session files in its folder (the old *_perugini_keypress files).
- [x] 4. Revert the uncommitted Perugini-loading edits in subject_profiles.py / subject_report.html; UCLA controls now
       arrive through load_block_sessions.
- [x] 5. Update READMEs (compiled, artifacts) and STATUS decisions.
- [x] 6. Re-run 0.05 (check UCLA patients unchanged, 10 controls equal the old compiled asmHC KP sessions), regenerate
       the report, commit.

## Log
- 2026-10-21: created.
- 2026-10-21: done. 0.05 re-run: 40 UCLA patient sessions unchanged; 10 key-press controls (AV BC BF EM ES GF GP JA
  MRM SY) compiled to compiled/ucla/sessions/<subject>.parquet, each equal trial for trial to the old
  compiled_old/sessions/<subject>_asmHC_KP_*.parquet (same priors); 8 stale *_perugini_keypress files pruned from
  compiled/ucla/perugini (now the 4 eye-movement controls only). Report: 81 subjects incl. 10 UCLA controls.
