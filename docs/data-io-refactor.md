# Separating file I/O from analysis

October 2026. Branch `dev`, on top of `37d515c`.

Notebooks used to build their own paths and call `read_csv` / `to_csv` / `savefig` directly. Every
read and write now goes through one package, `src/data_io`, and the notebooks contain only analysis.
Re-running every notebook gives the same results as before the change (see [Verification](#verification)).

## Why

The audit before the change found:

- **One file overwritten by three stages.** `processed_metadata_all_data_accu_60.csv` was written by
  0.20, then read and saved back to the same file by 0.30 (UPDRS) and 0.40 (psychometric
  parameters). Its content depended on which notebooks had run and in what order. Re-running 0.20
  silently dropped the UPDRS and psych columns. Re-running 0.30 turned its yes/no columns into NaN,
  because it mapped `"yes"/"no"` to `1/0` on a file that already held `1/0`.
- **Text getting more corrupted on every run.** Each stage read that file as `latin1` and wrote it
  as UTF-8, so "Didn't" in `notes` had been double-encoded several times over.
- **A second file out of sync with the first.** `..._ddm_params.csv` was a full copy of the
  metadata plus DDM columns, and its `*_psych_alpha` columns no longer matched the main file. The
  line that wrote it was commented out in 3.40.
- **Writes into the data directory**, e.g. the MCMC cache, and a dead line in 0.00 that built a
  save path inside `raw/`.
- **Paths retyped everywhere.** The same filenames appeared about 25 times, with different
  `read_csv` options. `accu_60` was hard-coded instead of coming from the config. Seven notebooks
  each pasted their own `OUTPUT_DIR` guard.
- **Outputs in odd places.** 2.25 saved figures into `notebooks/`, and 3.x relied on `os.chdir`
  so a relative `dissemination/...` path would work.
- **Undeclared dependencies.** 2.27 and 4.20 read 2.26's output with nothing saying so.
- **Functions copied between notebooks.**

## The new layout

```
src/data_io/
  paths.py     roots and write rules
  catalog.py   named datasets, load/save, session metadata
  cache.py     storage for fingerprinted fit caches
```

### Where code may write (`paths.py`)

| Root | Rule |
|---|---|
| `raw/` | Read only. Always refused. |
| `compiled/`, `processed/` | Derived data. Written only by the pipeline stages (0.x, 3.10, 3.40, `scripts/`). |
| `outputs/` (gitignored) | Analysis results: tables, figures, analysis-only caches. |
| `dissemination/` | Tracked slide decks. |

`writable(path)` resolves symlinks before checking, so a link inside a writable root that points
into `raw/` is refused. Every root can be redirected with `PD_RAW_DIR`, `PD_COMPILED_DIR`,
`PD_PROCESSED_DIR` or `PD_OUTPUTS_DIR`. Use these to try a pipeline change on a scratch copy.

File names no longer carry an accuracy tag such as `accu_60`. The accuracy threshold is applied
when data are loaded (see [Accuracy screen at load time](#accuracy-screen-at-load-time)).

### Using it in a notebook

```python
from src.data_io import catalog

trials   = catalog.load_trials()                      # valid trials; "all" for every trial
metadata = catalog.load_metadata()                    # add ddm_params=True for the DDM fits
bundle   = catalog.load("shared_glm_hmm_final", run="XY__tgt")

catalog.save_table(df, "subtype_comparison.csv")      # -> outputs/
catalog.save_figure(fig, "subtype_comparison")        # -> outputs/subtype_comparison.pdf + .png
```

Every shared file is registered in `catalog.DATASETS` with its location, read/write options and the
notebook or script that produces it. To add a dataset, add an entry there.

### Session metadata, one file per stage

| Stage | Writes `processed/metadata/…` | Reads |
|---|---|---|
| 0.20 | `sessions.csv` (includes `easy_accuracy_pct`) | compiled tables |
| 0.30 | `updrs.csv` | sessions only |
| 0.40 | `psych_params.csv` | sessions + updrs |
| 3.40 | `ddm_params.csv` | DDM model fits |

Each stage writes only the columns it added or changed, keyed by `session_id`.
`catalog.load_metadata()` joins them in the order above. Row order and column order are the same as
the old combined file, so code that indexes rows by position (`grp_indices`) still works. A stage
that lacks rows for some sessions raises an error instead of silently leaving gaps.

### Caches

`mcmc_cache.py` and `robustness.py` keep their model code and their cache-validity rules. Only the
file handling moved to `cache.py`. The fingerprint is computed exactly as before, so all existing
caches still load without refitting. The cache locations are also unchanged: `processed/mcmc_fits`
and `outputs/robustness_cache`. Notebooks that read the reported draws directly now call
`mcmc_cache.cached_draws(name)`.

**Full posterior in the MCMC cache (added 13 Oct 2026).** The cache used to keep only the
population-level draws (`beta`). So 2.26 and 2.28 sampled the model again to get the per-subject
draws, then asserted the new population draws matched the cache bit for bit. That assertion failed
after the rebuild. Re-sampling with identical data, settings, seed and library versions
reproduced itself exactly on one machine, but not the fit cached from another kernel. NUTS draws
are only bit-reproducible within the same compute environment. `fit_or_load` now also caches `sd_u`,
`z_u` and the subject order, and 2.26 and 2.28 read the reported posterior instead of re-sampling.
Every notebook therefore uses the same draws, whichever machine sampled them. A cache written
before this change has no `z_u`, so it is sampled once more on its next use.

### Functions moved out of notebooks

Only functions that were identical in every copy and didn't rely on notebook variables were moved:

| Function | Was in | Now in |
|---|---|---|
| `determine_choice`, `get_prior_condition` | 0.00, 0.01, 0.10 | `src/utils/sessions.py` |
| `new_accumulators`, `collect_psych`, `plot_mean_psych` | 2.20, 2.30 | `src/utils/group_psych.py` |
| `get_simulated_data` | 3.20, 3.30, 3.40 | `src/ddm/utils.py` |
| `observed_bias` | 2.25, 2.30 | `src/utils/mcmc_cache.py` |

### Compiling sessions (added 12 Oct 2026)

`compiled/` used to mix two things: hand-copied raw `.mat` files and Perugini `.parquet` tables.
A `.mat` session was then decoded into trials in three different notebooks. Now each raw session
is decoded once, by `src/utils/sessions.py`, and saved as `compiled/sessions/<stem>.parquet`. Every
session table has the same columns (`SESSION_COLUMNS`), including who ran it, under which
medication and with which prior.

| Source | Compiled by | Stem |
|---|---|---|
| Per-session `.mat` files at the top of `raw/` (UCLA, Stanford, Case Western) | 0.00 | the raw file name, e.g. `AJL_OFFmeds_gr` |
| Harvard controls and patients (trial fragments in `raw/harvard_data/`) | 0.00 | `HC1_healthy_gr`, `P25_OFFmeds_rl` |
| Perugini (2016) controls, key-press and eye-tracking | 0.01 | `AV_asmHC_KP_gr`, `CB1_asmHC_EM_rl` |

The top-level raw files for P25–P27 are older copies of the Harvard sessions, so 0.00 skips them.
0.10 joins the session tables in name order, which makes the row order of the trial tables
fixed from now on. HC medication is stored as a real missing value rather than the string
`"None"`, which used to become missing only because pandas parses it that way when reading a CSV.

0.20 changes:
- `file_name` names the session with a real prior. For the 8 UCLA patients who also ran an
  equal-prior session (CG, COH, DP, FUR, LBR, MAR, MBY, SMI), it used to name the `_eq` file.
- A line that did nothing (blanking prior columns for rows already removed) is gone.
- `data_slice` now splits each session at its middle trial in both trial tables. Before, it took
  the halfway point from the all-trials table and applied it to the valid-trials table, so the two
  halves didn't match. It is off in the config (`data_slice: 0`), so no existing result used it.

`openpyxl` was added to the dependencies; 0.01 needs it to read the eye-tracking reports.

### Accuracy screen at load time (added 12 Oct 2026)

0.20 no longer removes sessions below `min_easy_accuracy`. It records each session's accuracy on
valid 100%-coherence trials as `easy_accuracy_pct` and keeps every session. The loaders apply the
screen:

```python
catalog.load_trials()                          # config threshold (0.6)
catalog.load_trials(min_easy_accuracy=0.9)     # any other threshold, no re-run needed
catalog.load_metadata(min_easy_accuracy=None)  # every session
catalog.load_trials(stage="compiled", min_easy_accuracy=0.9)  # decoded sessions before preprocessing
```

`stage="compiled"` returns the sessions as decoded, before preprocessing: every subject (including
MoCA-rejected ones and incomplete patients), equal-prior sessions, and sides and colours as
recorded ("left"/"right", "green"/"red"). There the loader computes each session's accuracy from the
session tables, using the same function 0.20 uses to fill `easy_accuracy_pct`
(`src.utils.sessions.easy_accuracy_pct`). Equal-prior sessions don't count towards the screen at either
stage, so a given threshold passes the same subjects at both stages, apart from those 0.20 removes.

```python
catalog.load_metadata(stage="compiled")        # raw session metadata + site + easy_accuracy_pct, same screen
```

Accuracy is calculated exactly as the preprocessing always has, per session:
`np.mean(outcome on valid 100%-coherence trials) * 100`. It is stored in percent as
`easy_accuracy_pct`, and a session fails when it is below `min_easy_accuracy * 100`. Those lines now
live in `src.utils.sessions.easy_accuracy_pct`, so 0.20 and the compiled-stage loader share them.
Derived CSVs are read back exactly (`float_precision="round_trip"`); pandas' default parser could
be one bit off. That's why `trem_by_brady` used to drift slightly each time a stage re-read the
old combined file.

A subject passes only if **every** one of their sessions reaches the threshold. That is exactly
what the old pipeline did: rejecting one session of a patient left them without a pair, so they
were dropped. The comparison is written the same way as the old rejection test, so the same
sessions pass, including at the boundary.

The pipeline stages (0.30, 0.40) read with `min_easy_accuracy=None`, so their files cover every
session and any threshold works. Fitted models are the exception. The DDM, GLM-HMM and MCMC fits
were run on the subjects passing 60%, so `load_metadata(ddm_params=True)` with a lower threshold
raises an error naming the missing sessions.

`catalog.load("trials_valid")` and the other unscreened files refuse to load directly, so nothing
can read them without the screen by accident.

## Verification

Both the original code and the refactored code were run in isolated copies of the repo. Their
writes were pointed at scratch directories, so the shared data was never touched until the final
step.

| Check | Result |
|---|---|
| Notebook outputs, cell by cell (28 notebooks) | Identical, apart from the new "saved …" messages and a directory-listing order. |
| Result files in `outputs/` | 35 byte-identical. 11 PDFs differ only in their timestamp; their PNGs are pixel-identical. |
| 2.25's moved figure and text | Identical to the copies it used to write into `notebooks/`. |
| Trial tables and `ddm/behavior_data.csv` from the new 0.20 / 3.10 | Byte-identical to the files in `processed/`. |
| 0.10 compiled tables, old vs new code | Identical, rows in the same order. |
| Joined metadata vs the old combined file | Same 100 × 87 shape and column order. Differences are explained under Findings. |
| DDM parameters rebuilt from the model pickles | All 22 columns exactly equal to the old file. |
| Notebook lint (unused or undefined names) | No new warnings. |
| Fitting scripts | All import and run `--help`; the fits themselves were not run. |

After that, the refactored 0.20, 0.30, 0.40 and the GPU-free cells of 3.40 were run on the real
data. That added the four files in `processed/metadata/`. Every pre-existing file in `processed/`
was checksummed before and after and is byte-identical. 1.10 then ran on the real data and matched.

### Not executed here

- **3.20, 3.30, and 3.40's simulation cells** need a CUDA GPU. Their other cells were compared and
  match.
- **0.00 and 0.01** weren't run in this first pass. Since the 12 Oct changes both run; see the
  verification of the compile and screening changes below.
- **The fitting scripts** (Slurm jobs) were checked by import only.

### Verification of the compile and screening changes

The new pipeline (0.00 → 3.10) was run on a scratch copy against the real `raw/`, then compared
with the renamed `compiled2/` and `processed2/`.

| Check | Result |
|---|---|
| Compiled trial table, per session | 145 of 147 sessions identical in every column. The two that differ are P3_OFF and P4_ON, the corrected duplicate (see Findings). |
| Compiled metadata | Identical except line endings inside free-text `pd_meds` notes. |
| Valid and all trials, loaded at 60% | Identical rows and columns to `processed2/`, apart from P3 and P4. |
| Metadata, loaded at 60% | Same sessions in the same order. Differences: the new `easy_accuracy_pct` column, the `file_name` fix (16 rows) and extension (`.parquet`), plus the known `pd_meds`/`notes`/`trem_*` items below. |
| Loaded at 90% | Same subjects as the old `processed_metadata_all_data_accu_90.csv`, plus P4 (corrected file). |
| Compiled vs processed stage | At every threshold, processed subjects are a subset of compiled ones; the extras are MoCA-rejected subjects. Valid trials per session are identical, and `easy_accuracy_pct` is identical for all 120 sessions. Recomputing it with the original 0.20 loop gives the same values, and the screen drops the same 12 subjects the old rejection did. |
| Exact CSV reads vs `processed2/` | `reaction_time` and five UPDRS-derived scores differ in the last bit only (at most 2.5e-16 relative; 7e-15 absolute). Subtypes, UPDRS totals and psychometric parameters are identical. |
| Subjects kept by threshold | none: 70, 50%: 67, 60%: 58, 70%: 54, 80%: 50, 90%: 42 |
| Analysis notebooks on the new layout | 1.10, 2.10, 2.20, 2.22, 3.40, 3.50, 4.00, 4.10, 4.20 run. 4.01 fails because the GLM-HMM fits were trained on the duplicated P4_ON data. |

The notebooks that sample MCMC models (2.24–2.31) weren't run on the new layout. Their caches
don't match the new data, because P3 is now screened out and the trial tables' row order is new,
so they will re-sample on the next run.

## Findings

These problems came to light during the work. None of them is caused by the refactor.

1. **2.28 and 2.29 don't reproduce the files in `outputs/`.** Re-running them with the committed
   code, before or after the refactor, gives different numbers from the files dated 23 Sep. In
   `covariate_adjustment.csv`, `col:med` moves by up to 0.11 and `P(> 0)` by up to 0.13. In
   `prior_re_sensitivity.csv`, `P(> 0)` moves by up to 0.02. Those files were written before
   commit `4ae76d2` added the notebooks. Either the code changed after that run, or the sampler
   isn't reproducible across machines. **Check this against the reviewer response before relying
   on those numbers.**
2. **The old metadata file was out of date.** In 38 PD sessions whose patients couldn't be
   subtyped, `trem_vs_non-trem` and `pigd_vs_non-pigd` were "non-tremor"/"non-pigd"; the current
   0.30 code sets NaN. The file predates that fix. Nothing outside 0.30 reads these columns.
3. **The free-text columns are fixed.** `notes` and `pd_meds` no longer accumulate encoding damage.
   Raw metadata is still decoded as `latin1`, so a raw `’` shows as a control character `\x92`.
   Switching raw to `cp1252` would fix that if it matters.
4. **`trem_by_brady` for P7_OFF** differs by 2e-16. The old file had lost its last digit after
   repeated CSV round trips.

5. **P3_OFF and P4_ON were the same recording.** In the old `compiled/`, `P3_OFFmeds_rr.mat` and
   `P4_ONmeds_gl.mat` hold identical trial data (543 trials, 73% easy-trial accuracy), saved twice
   under different names. The raw files were corrected on 12 Oct 2026. P3_OFF now has 820 trials
   and **scores 52%, so P3 no longer passes the 60% screen** (58 subjects instead of 59). P4_ON has
   643 trials and scores 94%, so P4 now also passes at 90%. Every result made before the
   correction used the duplicate for both sessions, and all the fits in `processed2/` were trained
   on it.

## Open items

- **Re-fit after rebuilding.** The DDM, GLM-HMM and MCMC fits need to be re-run on the rebuilt
  `processed/`. The data changed (P3, P4) and the trial tables' row order is new.

- **Old data folders.** `compiled2/`, `processed2/` and the other renamed folders are the
  pre-rebuild copies. Nothing in the repo reads them. Keep them until the rebuild and the re-fits
  are checked. The 10 HC `.mat` files written to `compiled/` on 12 Oct by the old 0.00 aren't read
  by the new pipeline and can be deleted.
- **Old copies still in git.** `notebooks/figure2ef_*.pdf/png/md` are now produced in `outputs/`
  by 2.25; the committed copies under `notebooks/` can be removed with `git rm`.
- **GLM-HMM plotting helpers not moved.** The ones duplicated in 4.01 and 4.10 rely on notebook
  variables (`feats`, `behavior_df`, `STATE_COLORS`). Moving them means adding those as parameters.
- **Diverged copies not merged.** Some duplicated functions differ between notebooks
  (`collect_psych` in 2.10, `plot_param_comparison` in 3.x, `compute_group_psychometrics` in
  4.x). Merging them means deciding which version is right.
- **3.20 and 3.30** should be re-run on the GPU machine to confirm the simulation cells.
