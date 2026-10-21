# Status
Last updated: 2026-10-21 by Claude Code (Opus 5.5)

## Current focus
Rebuilding the data pipeline from raw: every site compiled from raw files with checks (notebooks 0.02 Harvard,
0.03 Stanford, 0.04 Case Western, 0.05 UCLA + Perugini) into /mnt/pd-data/compiled/; per-subject profiles and an
interactive report in /mnt/pd-data/artifacts/subject_profiles/ (scripts/pipeline/subject_profiles.py).

## Active plans
- none

## Recent reports (newest first, keep about 10)
- reports/2026-10-21-colour-direction-check.md : plot colour/direction coding matches the processed pipeline exactly;
  UCLA/Case Western show a red->right / green->left association even without a prior.

## Known issues
- Downstream pipeline (0.10 -> 0.20 -> analyses) still expects the old compiled layout; not yet switched to
  compiled/<site>/ (see compiled/README.md).
- BEHAVIOR.min_rt = 0.1 is in seconds but reaction times are in ms throughout, so the pipeline's RT floor is 0.1 ms
  (no valid trial is under 100 ms today, so no effect yet). subject_profiles.py converts to 100 ms.
- No metadata for Harvard HC10, HC11, HC14-HC16, Stanford P2, P8, P10, P14-P16 (UPDRS only as scanned PDFs), or any
  Case Western patient; COH off_med_first rows conflict.
- Stanford P14 OFF: 458/640 trials timed out, almost all at coherence < 100% (psychometric rests on 9-19 trials per
  low coherence).
- Design problems (two-prior sessions, prior changes between blocks) are flagged per block in compiled data; see
  docs/data-screening.md and select_blocks in src/utils/blocks.py.

## Decisions
- 2026-10-20: compiled data name sessions by subject + medication only; raw-name labels (Trem/Brady, prior) are not
  carried over (user request).
- 2026-10-20: UCLA eq files are practice blocks; UCLA explicit sessions are separate sessions (<id>_off_explicit);
  Perugini 2016 controls compiled apart (compiled/ucla/perugini/), 8 key-press + 4 eye-movement (user decision).
- 2026-10-20: shareable reports/figures go to /mnt/pd-data/artifacts/ (user request).
- 2026-10-21: colour/direction coding checked and kept (reports/2026-10-21-colour-direction-check.md).
