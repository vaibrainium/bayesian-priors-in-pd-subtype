# Table 1. Participant demographics

Values computed directly from `processed_metadata_all_data_accu_60.csv` (final, post-exclusion sample). Mean ± SD unless noted. See footnotes for data-completeness caveats — several fields were not available for all subjects in the merged metadata and should be verified against original per-site records before publication.

| | Healthy Controls | All PD | Tremor-dominant | Bradykinesia-dominant | Intermediate |
|---|---|---|---|---|---|
| N (subjects) | 18 | 41 | 11 | 10 | 1 |
| Sessions | 1 (no medication) | 2 (OFF + ON) | 2 (OFF + ON) | 2 (OFF + ON) | 2 (OFF + ON) |
| Site (N) | UCLA 10, Harvard 8 | UCLA 15, Stanford 16, Harvard 6, Case Western 4 | — | — | — |
| Age, years¹ | not available in merged metadata | 64.1 ± 8.9 (n=22) | 64.6 ± 10.2 (n=11) | 64.3 ± 7.8 (n=10) | — |
| Sex (M/F)¹ | not available in merged metadata | 15 / 7 (n=22) | 6 / 5 | 9 / 1 | — |
| Years since diagnosis¹ | — | 3.8 ± 2.9 (n=22) | 4.2 ± 3.1 | 3.4 ± 2.8 | — |
| MoCA¹ | — | 28.3 ± 2.3 (n=27) | 27.0 ± 2.0 (n=6) | 27.8 ± 1.3 (n=5) | — |
| Hoehn & Yahr stage¹ | — | 1.9 ± 0.3 (n=14) | 1.8 ± 0.4 (n=6) | 2.0 ± 0.0 (n=7) | — |
| MDS-UPDRS III, OFF¹ | — | 20.4 ± 7.6 (n=22) | 19.2 ± 6.5 | 21.2 ± 9.1 | — |
| MDS-UPDRS III, ON¹ | — | 14.6 ± 7.9 (n=22) | — | — | — |
| Tremor subscore (OFF)¹ ² | — | 0.61 ± 0.35 (n=22) | 0.84 ± 0.25 | 0.34 ± 0.24 | — |
| Bradykinesia subscore (OFF)¹ ² | — | 0.74 ± 0.39 (n=22) | 0.59 ± 0.31 | 0.89 ± 0.44 | — |
| Tremor/Bradykinesia ratio (OFF) | — | — | > 1.0 (classification cutoff) | < 0.8 (classification cutoff) | 0.8–1.0 |

**Notes**

1. **Data completeness:** demographic and MDS-UPDRS item-level fields were present only for the 22/41 PD subjects recruited at Stanford and Harvard (whose itemized ratings were entered into the merged metadata table); the remaining 19 PD subjects (15 UCLA, 4 Case Western) and all 18 healthy controls have these fields blank in `processed_metadata_all_data_accu_60.csv`, even though the subjects themselves are not missing. This is very likely a data-merge gap (e.g., the UCLA cohort's demographics may live in the original Perugini/Thakur records rather than this file) rather than a true absence of the data — **pull final Table 1 values for those subjects from the original per-site enrollment records** rather than reporting these as "not collected."
2. Tremor and bradykinesia subscores are the subject's mean rating (0–4 scale) across the MDS-UPDRS items listed in `config/main.yaml` (`TREMOR_ITEMS`, `BRADYKINESIA_ITEMS`), computed from the OFF-medication exam only, per `notebooks/0.30-data-updrs-scores-compilation.ipynb`.
3. Motor subtype (tremor-dominant / bradykinesia-dominant / intermediate) was assigned from the ratio of these two subscores, thresholded at < 0.8 / 0.8–1.0 / > 1.0 respectively (`config/main.yaml`, `TREM_VS_BRADY`); the one intermediate-ratio patient was excluded from subtype-specific analyses (Figs. 2–4).
4. MoCA ≤ 26 was an a priori exclusion criterion (`main_config.moca_rejection`, 8 subjects), applied before the sample above.
