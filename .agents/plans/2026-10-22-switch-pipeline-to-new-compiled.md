# Plan: Switch the downstream pipeline (0.10 -> 0.20 -> analyses) to the new compiled data
Status: active (awaiting user approval; nothing changed yet)
Created: 2026-10-22 by Claude Code (Opus 5.5)
Related reports: reports/2026-10-21-colour-direction-check.md

## Goal
0.10 and 0.20 read the new per-site compiled data (compiled/<site>/sessions, notebooks 0.02-0.05) instead of the
old compiled/sessions/<stem>.parquet layout, and write processed/ tables with the same schema, so 0.30, 0.40 and every
1.x-4.x notebook and script keeps working unchanged.

## Context (survey of 2026-10-22)
- Analysis layer (1.10, 2.10-2.34, 3.10-3.50, 4.00-4.20, scripts/ddm, glm_hmm) reads only catalog.load_trials /
  load_metadata (processed stage) or processed datasets: no change needed if 0.20's outputs keep their schema.
- Must change:
  - 0.10: reads catalog.files("compiled_session") = compiled/sessions/*.parquet (old layout), hard-codes site lists,
    asserts data subjects == metadata subjects, filters eye-tracking by old file stems.
  - 0.20: removes prior == "eq" sessions (UCLA eq is now the practice phase inside the session), keys on
    session_filename / file_name (values change: "P1_off.parquet" instead of "P1_OFFmeds_rr.parquet").
  - catalog: "compiled_session" dataset points at the old folder; _compiled_session_accuracy raises if a subject has
    two biased sessions for one medication state (UCLA explicit sessions would trigger it).
- Retire: 0.00 (reads top-level raw .mat files that no longer exist) and 0.01 (replaced by 0.05 section 7).
- Audit scripts that read the compiled stage: a01 (parses the prior from old file names; obsolete), a08 (uses UCLA
  "eq" sessions; would read practice phase instead), a21 (uses _compiled_session_accuracy). a20 joins on file_name.
- Content differences the switch brings in (checked 2026-10-22):
  - data subjects not in raw metadata: HC10, HC11, HC14, HC15, HC16 (0.10's assert would fail);
  - metadata subjects with no data: CB1, JW, KR, MEG (Perugini eye-movement, excluded by config);
  - 6 explicit sessions (<id>_off_explicit) must be excluded from the main analysis;
  - 15268 practice trials in 92 sessions are now in the session tables (phase == "practice") and must be dropped;
  - P1 OFF loses its 42 practice trials (old compile included them); DCAM OFF present as before.

## Assumptions
- Processed outputs keep session_id = "<SUBJECT>_<OFF|ON>" for patients and "<subject>" for controls.
- Practice and explicit sessions are excluded from the main analysis; design screening stays at load time.

## Steps
- [ ] 1. catalog: point compiled-stage readers at the new data (compiled_trials / compiled_metadata stay, written by
       the new 0.10); remove or redirect the "compiled_session" dataset.
- [ ] 2. 0.10: build the aggregate from catalog.load_block_sessions(): phase == "trial", drop *_explicit; site from
       the `site` column (UCLA / Case_Western / Stanford / Harvard strings as before); keep SESSION_COLUMNS (+ block
       columns?); replace the hard assert with a printed reconciliation (HC10-16 without metadata; eye-movement
       controls without data) and decide how to handle them (user decision).
- [ ] 3. 0.20: drop the prior == "eq" filters (no eq sessions any more); keep everything else.
- [ ] 4. Mark 0.00 and 0.01 as superseded (header cell) or remove them (user decision).
- [ ] 5. Audit scripts: retire a01, update a08 (practice phase), check a20/a21.
- [ ] 6. Verify: run 0.10 and 0.20 into a scratch PD_PROCESSED_DIR first; compare processed trials/metadata with the
       current processed/ (expected differences: P1 OFF practice removed, new HCs if added); then run for real.
- [ ] 7. Re-run 0.30 and 0.40 and spot-check downstream notebooks load.

## Out of scope
- Re-running model fits (DDM, GLM-HMM, MCMC); list which results become stale instead.

## Log
- 2026-10-22: created from the survey; awaiting approval.
