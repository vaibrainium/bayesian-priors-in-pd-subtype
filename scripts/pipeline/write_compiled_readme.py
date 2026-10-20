"""Write compiled/README.md (what the compiled data are, how they are laid out and how to read them) and
artifacts/README.md (what each artifacts folder holds and what makes it).

The counts come from the compiled session summaries and the artifacts folders, so re-run this after recompiling or
regenerating:

    .venv/bin/python -m scripts.pipeline.write_compiled_readme
"""

from datetime import date

from src.data_io import catalog
from src.data_io.paths import ROOTS

SITES = {  # site folder: (name, notebook)
    "harvard": ("Harvard", "0.02-harvard-sessions-check-and-compile"),
    "stanford": ("Stanford", "0.03-stanford-sessions-check-and-compile"),
    "case_western": ("Case Western", "0.04-case-western-sessions-check-and-compile"),
    "ucla": ("UCLA", "0.05-ucla-sessions-check-and-compile"),
}


def site_counts():
    rows = []
    for site, (name, notebook) in SITES.items():
        summary = catalog.load(f"compiled_{site}_session_summary")
        blocks = catalog.load(f"compiled_{site}_blocks")
        compiled = summary[summary.n_trial_blocks > 0]
        explicit = compiled.session_id.str.endswith("_explicit").sum()
        people = ", ".join(f"{n} {'patients' if g == 'pd' else 'controls'}" for g, n in compiled.groupby("group").subject_id.nunique().items())
        rows.append(f"| `{site}/` | {name} | {len(compiled)}" + (f" ({explicit} explicit)" if explicit else "") + f" | {people} | "
                    f"{len(blocks)} ({int((~blocks.use).sum())} left out) | `{notebook}` |")
    perugini = catalog.load("compiled_perugini_session_summary")
    modes = ", ".join(f"{n} {m.replace('_', ' ')}" for m, n in perugini.response_mode.value_counts().items())
    rows.append(f"| `ucla/perugini/` | Perugini et al. (2016) | {len(perugini)} ({modes}) | {perugini.subject_id.nunique()} controls | "
                f"{len(perugini)} | `{SITES['ucla'][1]}` |")
    return "\n".join(rows)


README = """# Compiled data

Every session of the study, decoded from the raw task files into one trial table per session, with block-level
labels and per-file and per-session check tables. Generated on {today} by `scripts/pipeline/write_compiled_readme.py`;
re-run it after recompiling.

- **Raw data only.** Each site notebook reads `raw/<site>_data/` (never the pre-combined copies, except where a site
  has nothing rawer) and checks every file before compiling it. The notebooks hold the full findings.
- **Nothing is screened.** Every recorded session is here, whatever its accuracy or design; sessions with problems
  are flagged in `session_summary.csv` (`issues`). Screening happens downstream.
- **Labels come from the data.** A session's prior is the one its trials show. Labels in raw folder and file names
  (subtype `Brady_`/`Trem_`, intended prior such as `_gl`) are wrong in places and are not carried over.

## Contents

| folder | site | sessions | people | raw files (left out) | written by notebook |
|---|---|---|---|---|---|
{counts}

Each site folder holds

| file | one row per | contents |
|---|---|---|
| `sessions/<session_id>.parquet` | trial | the session's trials: practice blocks, then trial blocks, each in time order |
| `session_summary.csv` | session | block counts, trials, prior, design and order checks, `issues` (and `notes`) |
| `blocks.csv` | raw file | timestamps, clock checks, design, whether it was used (`use`) and why not (`excluded_reason`) |

`ucla/perugini/` holds `sessions/` and `session_summary.csv` only.

## Naming

- **Sessions** are named by subject and medication state only: `P25_off`, `LHO_on`, `HC3` (controls have no
  medication state). UCLA's explicit-instruction sessions add `_explicit` (`CG_off_explicit`); Perugini sessions are
  `<subject>_perugini_keypress` / `_eye`.
- **Blocks** are named `<session_id>/<path inside the raw session folder>`, e.g. `P10_off/Practice/221209_102703.mat`.
  At Case Western and UCLA, where a session is one file, they are `<session_id>/trial` and `<session_id>/practice`.
- Subject codes: Harvard `P25`-`P34` and `HC1`-`HC16`, Stanford `P1`-`P24`, Case Western and UCLA initials.

## Session tables

| column | meaning |
|---|---|
| `subject_id`, `group`, `medication` | `pd` / `hc`; `off` / `on` (missing for controls) |
| `session_id`, `session_filename` | session name; `<session_id>.parquet` |
| `prior`, `prior_color`, `prior_direction` | the session's prior (the one most of its complete trial blocks show): `gr` = green targets 75% right, `rl` = red 75% left, ...; `gr+rl` both colours biased; `none`. Colour and direction only for a single-colour prior |
| `color`, `coherence`, `target` | stimulus: `green`/`red`; 100, 35, 13, 0 (%); correct side `left`/`right` |
| `is_valid`, `outcome`, `choice`, `reaction_time` | valid trial; 1 correct / 0 wrong; side chosen; reaction time (ms) |
| `phase`, `block`, `trial_in_block`, `block_file` | `practice` or `trial`; block number within its phase; trial number within the block; block name |
| `block_prior`, `block_prior_color`, `block_prior_direction` | the block's own prior |
| `block_design_error`, `block_design_ok` | departure from 75:25 on the prior colour and 50:50 on the other; within the load-time tolerance (trial blocks only) |
| `block_matches_session_prior`, `block_partial` | the block shows the session's prior (trial blocks only); fewer than 150 trials shown |
| `dataset`, `response_mode` | Perugini sessions only: `perugini_2016`; `keypress` / `eye_movement` |

## Reading and selecting blocks

```python
from src.data_io import catalog
from src.utils.blocks import select_blocks

trials = catalog.load_block_sessions()                       # every site except Perugini, one table, with `site`
correct = select_blocks(trials, design_ok=True)              # trial blocks run as 75:25 / 50:50
other = select_blocks(trials, design_ok=True, matches_session_prior=False, relabel=True)  # a block with a prior of its own
practice = select_blocks(trials, phase="practice")
perugini = catalog.load("compiled_perugini_session", filename="AV_perugini_keypress.parquet")
```

`select_blocks` options: `phase` (`"trial"` default, `"practice"`, `None`), `design_ok`, `matches_session_prior`,
`priors` (e.g. `{{"gr+rl"}}`), `include_partial`, `relabel` (give the selection its blocks' own prior).

## Site notes

- **Harvard** (block files, `{{pd,hc}}/<session>/{{practice,trial}}/YYMMDD_HHMMSS.mat`). Every file's filename, MAT header
  and event clock are checked against each other. Six sessions ran with both colours biased (`gr+rl`); P33 OFF
  and HC7 changed prior after block 1; P28 OFF has a partial block. 15 misfiled P25-P27 practice files were removed
  from raw on 2026-10-19 (see `docs/data-screening.md`).
- **Stanford** (block files directly in the session folder, practice in `Practice/`). Same checks; clock shifts from
  sleep or clock corrections are kept. P15 ON has no trial blocks and is not compiled; P9 ON ran with both colours
  biased; P3 ON, P8 OFF and P22 ON changed prior between blocks; P2-P8 practised at all coherences, later patients at
  100% only. Several folder names give the wrong prior.
- **Case Western** (one file per session). Each session is one block. 12 of 14 file headers are a batch re-save, so
  recording dates are lost; five patients ran OFF then ON in one sitting.
- **UCLA** (one file per session). The equal-prior `eq` file run just before a session is its practice block. Six
  explicit-instruction OFF sessions are separate sessions. All file headers are batch re-saves; the sessions ran on
  more than one PC.
- **Perugini et al. (2016)**: the paper's 12 age-matched healthy controls, from the UCLA folder. Kept apart from the
  study and not read by `load_block_sessions`.

## Not here yet

- Session metadata (demographics, clinical scores); it is still compiled by `0.10` from
  `raw/session_metadata_detailed_all_data.csv`.
- The downstream pipeline (`0.00`/`0.01` -> `0.10` -> `0.20` -> analyses) is not yet switched to these folders; it
  expects the earlier layout (`compiled/sessions/<stem>.parquet`, last built copy in `compiled_old/`).
"""


ARTIFACT_FOLDERS = {  # folder: (contents, made by)
    "subject_profiles": ("`report.html`, an interactive page of every subject (open in a browser); a figure per subject "
                         "(`<subject>.png` / `.pdf`); `profiles.csv`, one row per subject; `psychometrics.csv`, one row per subject, "
                         "session, prior condition and colour with the fitted parameters", "`scripts/pipeline/subject_profiles.py`"),
    "harvard": ("`session_timeline.png` / `.pdf`: every used Harvard block on its session's clock", "notebook `0.02`"),
    "stanford": ("`session_timeline.png` / `.pdf`: every used Stanford block on its session's clock", "notebook `0.03`"),
}

ARTIFACTS_README = """# Artifacts

Reports and figures made from the compiled data, to look at or share. Everything here is regenerated by the scripts and
notebooks named below (do not edit files here by hand). Listing generated on {today} by
`scripts/pipeline/write_compiled_readme.py`.

| folder | contents | files | made by |
|---|---|---|---|
{rows}

The data behind them are in `compiled/` (see its README).
"""


def artifact_rows():
    rows = []
    for folder, (contents, made_by) in ARTIFACT_FOLDERS.items():
        path = ROOTS["artifacts"] / folder
        n = sum(1 for f in path.iterdir() if f.is_file()) if path.exists() else 0
        rows.append(f"| `{folder}/` | {contents} | {n} | {made_by} |")
    return "\n".join(rows)


def main():
    text = README.format(today=date.today().isoformat(), counts=site_counts())
    catalog.save(text, "compiled_readme")
    catalog.save(ARTIFACTS_README.format(today=date.today().isoformat(), rows=artifact_rows()), "artifacts_readme")


if __name__ == "__main__":
    main()
