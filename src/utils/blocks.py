"""Checking and compiling sessions recorded as separate block files (compile stage: 0.02 Harvard, 0.03 Stanford).

Several sites ran each session as one or more practice blocks followed by several trial blocks, one MAT file per
block named `YYMMDD_HHMMSS.mat`. Each site notebook lists its files in its own folder layout as a table with
`LISTING_COLUMNS`: `phase` ("practice", "trial" or "unexpected"), `path` (relative to the site folder, used only to
read the file) and `file` (`<session_id>/<path inside the session folder>`, the block's name in everything saved).
Compiled outputs identify a session only by subject and medication state; labels in raw folder names (subtype,
intended prior) are unreliable and are not carried over, so `path` is dropped before saving (`INTERNAL_COLUMNS`).
Everything after the listing is shared:

    blocks, decoded = read_blocks(listing, root)   # timestamps, clocks and contents of every file
    check_timing(blocks); mark_use(blocks)         # which files were recorded in their session
    runs = add_provenance(blocks)                  # where the left-out files were recorded
    add_design(blocks, decoded)                    # colour splits and the prior each block implies
    summary = session_summary(blocks)              # per-session checks, `issues` and `notes`
    compile_sessions(blocks, decoded, summary, dataset)

Three clocks date each block:

- **filename**: when the task program started the block (the first trial can follow some minutes later);
- **MAT header** (`Created on:`): when MATLAB saved it, about 15 s after the last trial;
- **`time` matrix**: event times in ms on the PC's monotonic clock, which runs on through a session.

The header offset, header time minus the last clock value, is therefore constant across a session's blocks, with two
exceptions on the same day: it steps up when the PC sleeps between blocks (the monotonic clock pauses), and steps
down when the wall clock is corrected (a time sync). A block within `CLOCK_TOLERANCE_S` of its session's offset is
"on clock", one recorded on the session's day within `MAX_CLOCK_SHIFT_S` has a "clock shift" and is kept, and
anything else is "off clock": it was not recorded in that session (a renamed copy, or a file from another day).
"""

import hashlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

from config import main_config
from src.data_io import catalog
from src.utils.sessions import SESSION_COLUMNS, decode_events, prior_design_error

LISTING_COLUMNS = ["session_id", "subject_id", "group", "medication", "phase", "file", "path"]
INTERNAL_COLUMNS = ["path", "content_md5"]  # read-time only; not saved
# Per-trial block columns of the compiled tables; `select_blocks` filters on them.
BLOCK_COLUMNS = ["session_id", "phase", "block", "block_file", "trial_in_block", "block_prior", "block_prior_color", "block_prior_direction",
                 "block_design_error", "block_design_ok", "block_matches_session_prior", "block_partial"]

CLOCK_TOLERANCE_S = 60  # a block's header offset may differ from its session's by this much
MAX_CLOCK_SHIFT_S = 3 * 3600  # larger same-day shifts than this are not taken for a sleep or a clock correction
RECORDING_SLACK_S = 60  # header - filename may fall this far short of the recorded duration (a clock correction mid-block)
MIN_TRIALS_PER_BLOCK = 10  # fewer rows (or an event matrix without 10 columns) is an aborted start
FULL_BLOCK = 150  # trial blocks with fewer colour-shown trials than this are partial (a full block shows 160)
MIN_SESSION_TRIALS = 600  # the intended session length, in colour-shown trials
BIAS_PCT = 60  # a colour whose targets go more than this % to one side carries a prior
DESIGN_TOLERANCE = main_config.BEHAVIOR.prior_design_tolerance
EPOCH = pd.Timestamp("2000-01-01")  # any fixed origin; only differences of header offsets matter
SINGLE_PRIORS = ("gr", "gl", "rr", "rl")


def header_time(mat):
    """The `Created on:` time in a MAT-file header (MATLAB pads single-digit days with a space)."""
    found = re.search(r"Created on: (.+?\d{4})", mat["__header__"].decode("latin1"))
    return pd.to_datetime(" ".join(found.group(1).split()), format="%a %b %d %H:%M:%S %Y") if found else pd.NaT


# --------------------------------------------------------------------------------------------- #
# Reading and timing
# --------------------------------------------------------------------------------------------- #


def read_blocks(listing, root):
    """Each listed file's timestamps, clock range and content hash, and the decoded trials of every 10-column block."""
    rows, decoded = [], {}
    for entry in listing.to_dict("records"):
        path = root / entry["path"]
        mat = scipy.io.loadmat(path)
        event, time = mat["event"], mat["time"]
        if event.shape[1] == 10:
            decoded[entry["file"]] = decode_events(event, time)
        rows.append({
            **entry,
            "filename_time": pd.to_datetime(path.stem, format="%y%m%d_%H%M%S", errors="coerce"),
            "header_time": header_time(mat),
            "n_rows": event.shape[0],
            "n_cols": event.shape[1],
            "clock_start_s": time[0, 0] / 1000,
            "clock_end_s": time[time > 0].max() / 1000,
            "content_md5": hashlib.md5(event.tobytes() + time.tobytes()).hexdigest(),
        })
    return pd.DataFrame(rows), decoded


def _complete(blocks):
    return (blocks.n_cols == 10) & (blocks.n_rows >= MIN_TRIALS_PER_BLOCK)


def check_timing(blocks):
    """Add the timing columns and `clock_status` (on clock / clock shift / off clock / no trial blocks)."""
    blocks["duration_s"] = blocks.clock_end_s - blocks.clock_start_s
    blocks["header_minus_filename_s"] = (blocks.header_time - blocks.filename_time).dt.total_seconds()
    blocks["wait_s"] = blocks.header_minus_filename_s - blocks.duration_s  # program start -> first trial, plus the save
    blocks["header_offset_s"] = (blocks.header_time - EPOCH).dt.total_seconds() - blocks.clock_end_s

    trial = blocks[_complete(blocks) & (blocks.phase == "trial")]
    blocks["session_date"] = blocks.session_id.map(trial.groupby("session_id").filename_time.agg(lambda t: t.dt.date.mode()[0]))
    blocks["clock_drift_s"] = blocks.header_offset_s - blocks.session_id.map(trial.groupby("session_id").header_offset_s.median())
    blocks["header_same_day"] = blocks.header_time.dt.date == blocks.filename_time.dt.date
    blocks["saved_after_recording"] = blocks.wait_s >= -RECORDING_SLACK_S

    drift = blocks.clock_drift_s.abs()
    on_day = blocks.header_time.dt.date == blocks.session_date
    blocks["clock_status"] = np.select(
        [blocks.session_date.isna(), drift <= CLOCK_TOLERANCE_S, on_day & (drift <= MAX_CLOCK_SHIFT_S)],
        ["no trial blocks", "on clock", "clock shift"], default="off clock")
    same_content = blocks.groupby("content_md5").file.transform(lambda f: [", ".join(sorted(set(f) - {x})) for x in f])
    blocks["duplicate_of"] = same_content.replace("", None)
    return blocks


def excluded_reason(block):
    """Why a file is left out (the first reason that applies), or None."""
    if block.phase == "unexpected":
        return "outside the practice and trial locations"
    if block.n_cols != 10 or block.n_rows < MIN_TRIALS_PER_BLOCK:
        return f"aborted ({block.n_rows} x {block.n_cols} event matrix)"
    if block.clock_status == "no trial blocks":
        return "session has no trial blocks"
    if block.clock_status == "off clock":
        return f"not on this session's clock (saved {block.header_time:%Y-%m-%d %H:%M})"
    if not block.saved_after_recording:
        return f"saved {-block.wait_s:.0f} s before its trials ended"
    return None


def mark_use(blocks):
    """Add `excluded_reason` and `use`; a later file repeating a used file of the same session is left out too."""
    blocks["excluded_reason"] = blocks.apply(excluded_reason, axis=1)
    kept = blocks.excluded_reason.isna()
    repeat = kept & blocks[kept].sort_values("filename_time").duplicated(["session_id", "content_md5"]).reindex(blocks.index, fill_value=False)
    blocks.loc[repeat, "excluded_reason"] = "repeats an earlier block of this session"
    blocks["use"] = blocks.excluded_reason.isna()
    return blocks


def add_provenance(blocks):
    """Where each left-out file was recorded; returns one row per recording run.

    `recorded_with` names the sessions run the same day whose used blocks share the file's clock (the clock survives
    overnight when the PC is not restarted, so other days are not evidence). Left-out files sharing a day and an
    offset form one `recording_run`. `copy_of_used` is the used file with identical contents, if any.
    """
    used = blocks[blocks.use]
    left_out = blocks[~blocks.use].sort_values("header_time")
    if left_out.empty:
        return pd.DataFrame()

    def recorded_with(block):
        same_day = used[used.header_time.dt.date == block.header_time.date()]
        same_clock = same_day[(same_day.header_offset_s - block.header_offset_s).abs() <= CLOCK_TOLERANCE_S]
        return ", ".join(same_clock.session_id.unique()) or "no session that day"

    blocks.loc[left_out.index, "recorded_with"] = left_out.apply(recorded_with, axis=1)
    new_run = (left_out.header_time.dt.date != left_out.header_time.dt.date.shift()) | (left_out.header_offset_s.diff().abs() > CLOCK_TOLERANCE_S)
    blocks.loc[left_out.index, "recording_run"] = new_run.cumsum()
    original = used.set_index("content_md5").file
    blocks.loc[left_out.index, "copy_of_used"] = left_out.content_md5.map(original[~original.index.duplicated()])
    return blocks[~blocks.use].groupby("recording_run").agg(
        saved_from=("header_time", "min"), saved_to=("header_time", "max"), n_files=("file", "size"),
        recorded_with=("recorded_with", "first"), filed_under=("session_id", lambda s: ", ".join(s.unique())),
        reasons=("excluded_reason", lambda r: ", ".join(r.str.split(" \\(").str[0].unique())),
        copies_of=("copy_of_used", lambda d: ", ".join(d.dropna().unique())))


# --------------------------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------------------------- #


def implied_prior(green_right_pct, red_right_pct):
    """`gr`, `rl`, ... for each colour sending more than `BIAS_PCT` % of targets to one side, joined; `none` if neither."""
    parts = [f"{colour}{'r' if pct > 50 else 'l'}" for colour, pct in (("g", green_right_pct), ("r", red_right_pct)) if abs(pct - 50) > BIAS_PCT - 50]
    return "+".join(parts) or "none"


def add_design(blocks, decoded):
    """Per 10-column block: trials shown per colour, % right, coherences, implied prior, design error, partial."""
    rows = {}
    for file, trials in decoded.items():
        shown = trials.dropna(subset=["color", "target"])
        row = {"n_shown": len(shown), "coherences": ",".join(str(int(c)) for c in sorted(trials.coherence.dropna().unique()))}
        for colour in ("green", "red"):
            targets = shown.loc[shown.color == colour, "target"]
            row[f"n_{colour}"] = len(targets)
            row[f"{colour}_right_pct"] = 100 * (targets == "right").mean() if len(targets) else np.nan
        row["block_prior"] = implied_prior(row["green_right_pct"], row["red_right_pct"])
        rows[file] = row
    design = pd.DataFrame.from_dict(rows, orient="index")
    design["design_error"] = prior_design_error(pd.concat([d.assign(session_filename=f) for f, d in decoded.items()]))["prior_design_error"]
    for column in design:
        blocks[column] = blocks.file.map(design[column])
    blocks["partial"] = (blocks.phase == "trial") & (blocks.n_shown < FULL_BLOCK)
    return blocks


# --------------------------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------------------------- #


def _session_checks(session):
    used = session[session.use]
    practice = used[used.phase == "practice"].sort_values("filename_time")
    trial = used[used.phase == "trial"].sort_values("filename_time")
    if trial.empty:
        return pd.Series({"n_practice_blocks": len(practice), "n_trial_blocks": 0, "n_files_left_out": int((~session.use).sum()),
                          "issues": "no trial blocks", "notes": ""})
    full = trial[~trial.partial]
    priors = full.block_prior.value_counts()
    timeline = used.sort_values("filename_time")
    gaps = (trial.filename_time.iloc[1:].to_numpy() - trial.header_time.iloc[:-1].to_numpy()) / pd.Timedelta(seconds=1)
    shifted = used[used.clock_status == "clock shift"]
    checks = {
        "date": ", ".join(sorted({f"{d:%Y-%m-%d}" for d in used.filename_time})),
        "n_practice_blocks": len(practice),
        "n_trial_blocks": len(trial),
        "n_partial_blocks": int(trial.partial.sum()),
        "n_files_left_out": int((~session.use).sum()),
        "n_trials": int(trial.n_rows.sum()),
        "n_shown": int(trial.n_shown.sum()),
        "block_priors": " | ".join(trial.block_prior + np.where(trial.partial, "*", "")),
        "session_prior": priors.index[0] if len(priors) else None,
        "prior_consistent": trial.block_prior.nunique() == 1,
        "design_ok": bool((full.design_error <= DESIGN_TOLERANCE).all()),
        "max_design_error": full.design_error.max(),
        "practice_before_trial": bool(len(practice) and practice.header_time.max() <= trial.filename_time.min()
                                      and practice.clock_end_s.max() <= trial.clock_start_s.min()),
        "practice_to_trial_min": (trial.filename_time.min() - practice.header_time.max()).total_seconds() / 60 if len(practice) else np.nan,
        "block_order_consistent": timeline.sort_values("clock_start_s").file.tolist() == timeline.file.tolist() == timeline.sort_values("header_time").file.tolist(),
        "trial_blocks_overlap": bool((gaps < -RECORDING_SLACK_S).any()),
        "one_day": used.filename_time.dt.date.nunique() == 1,
        "reaches_600": int(trial.n_shown.sum()) >= MIN_SESSION_TRIALS,
        "max_clock_shift_min": shifted.clock_drift_s.abs().max() / 60 if len(shifted) else 0.0,
    }
    issues = []
    if not checks["prior_consistent"]:
        issues.append(f"block priors differ ({checks['block_priors']})")
    if not checks["design_ok"]:
        issues.append(f"design off 75:25 / 50:50 (error {checks['max_design_error']:.2f})")
    if not len(practice):
        issues.append("no practice recorded in this session")
    elif not checks["practice_before_trial"]:
        issues.append("practice not before trial")
    if not checks["block_order_consistent"] or checks["trial_blocks_overlap"]:
        issues.append("block order or overlap")
    if not checks["one_day"]:
        issues.append("blocks on several days")
    if not checks["reaches_600"]:
        issues.append(f"{checks['n_shown']} trials")
    if checks["n_partial_blocks"]:
        issues.append(f"{checks['n_partial_blocks']} partial block(s)")
    if checks["n_files_left_out"]:
        issues.append(f"{checks['n_files_left_out']} file(s) left out")
    checks["issues"] = "; ".join(issues)
    checks["notes"] = (f"clock shift of up to {checks['max_clock_shift_min']:.1f} min in {len(shifted)} block(s) (sleep or clock correction)"
                       if len(shifted) else "")
    return pd.Series(checks)


def session_summary(blocks):
    """One row per session, in listing order: block counts, prior, the order checks, `issues` and `notes`."""
    first = blocks.drop_duplicates("session_id").set_index("session_id")[["subject_id", "group", "medication"]]
    rows = {session_id: _session_checks(session).to_dict() for session_id, session in blocks.groupby("session_id", sort=False)}
    columns = max(rows.values(), key=len).keys()  # a session with trial blocks has every column, in order
    checks = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=columns)
    return first.join(checks).reset_index()


def compile_sessions(blocks, decoded, summary, dataset, extra_columns=()):
    """Save each session with trial blocks as one trial table: used practice then trial blocks, each in time order.

    The table has `SESSION_COLUMNS` plus `BLOCK_COLUMNS`. `prior` is the session's prior (the one most complete
    trial blocks imply); `prior_color` and `prior_direction` are filled only when it is a single colour. Every trial
    also carries its block's own prior (`block_prior`, `block_prior_color`, `block_prior_direction`), the block's
    design error and whether it ran the intended design (`block_design_ok`, within the load-time design tolerance),
    whether it implies the session's prior (`block_matches_session_prior`) and whether it is partial, so blocks can be
    selected later with `select_blocks`. Practice blocks are never `block_design_ok` or `block_matches_session_prior`.
    `extra_columns` are summary columns copied onto every trial (e.g. the Perugini `dataset` and `response_mode`).
    """
    for session in summary[summary.n_trial_blocks > 0].itertuples():
        parts = []
        for phase in ("practice", "trial"):
            phase_blocks = blocks[(blocks.session_id == session.session_id) & blocks.use & (blocks.phase == phase)]
            phase_blocks = phase_blocks.sort_values("filename_time" if "filename_time" in blocks else "clock_start_s")
            for number, block in enumerate(phase_blocks.itertuples(), start=1):
                trials = decoded[block.file].copy()
                trials["phase"], trials["block"], trials["block_file"] = phase, number, block.file
                trials["trial_in_block"] = np.arange(1, len(trials) + 1)
                trials["block_prior"] = block.block_prior
                trials["block_prior_color"], trials["block_prior_direction"] = prior_parts(block.block_prior)
                trials["block_design_error"] = block.design_error
                trials["block_design_ok"] = phase == "trial" and block.design_error <= DESIGN_TOLERANCE
                trials["block_matches_session_prior"] = phase == "trial" and block.block_prior == session.session_prior
                trials["block_partial"] = bool(block.partial)
                parts.append(trials)
        trials = pd.concat(parts, ignore_index=True)
        trials["prior"] = session.session_prior
        trials["prior_color"], trials["prior_direction"] = prior_parts(session.session_prior)
        trials["session_id"], trials["subject_id"] = session.session_id, session.subject_id
        trials["group"], trials["medication"] = session.group, session.medication
        trials["session_filename"] = f"{session.session_id}.parquet"
        for column in extra_columns:
            trials[column] = getattr(session, column)
        catalog.save(trials[SESSION_COLUMNS + BLOCK_COLUMNS + list(extra_columns)], dataset, filename=f"{session.session_id}.parquet")


# --------------------------------------------------------------------------------------------- #
# Sessions recorded as whole files (Case Western 0.04, UCLA 0.05)
# --------------------------------------------------------------------------------------------- #
# Some sites saved each session (and each practice run) as one file, with no per-block files or timestamped names.
# Such a file is listed like a block (`LISTING_COLUMNS`) and compiled as a single block; what can be checked is the
# clock (one continuous run, its pauses), the design over the session, and practice ordering on the clock.

PAUSE_S = 20  # a gap between trial starts longer than this is a pause
STRETCH = 160  # trials per stretch for the design-over-time check (a block's length at the block sites)
BATCH_SAVE_MIN_FILES, BATCH_SAVE_S = 3, 5  # this many files saved within this many seconds were re-saved together


def load_mat(path):
    """`event`, `time` and `__header__` of a MAT file, whether MATLAB v5 or v7.3 (HDF5)."""
    try:
        return scipy.io.loadmat(path)
    except NotImplementedError:  # v7.3
        import hdf5storage

        mat = hdf5storage.loadmat(str(path))
        with open(path, "rb") as handle:
            header = handle.read(116)  # v7.3 files keep the text header in their first bytes
        return {"__header__": header, "__v73__": True, "event": np.asarray(mat["event"]), "time": np.asarray(mat["time"])}


def read_session_files(listing, root):
    """Each listed whole-session file's header time, clock range, pauses and validity, and its decoded trials."""
    rows, decoded = [], {}
    for entry in listing.to_dict("records"):
        mat = load_mat(root / entry["path"])
        event, time = mat["event"], mat["time"]
        if event.shape[1] == 10:
            decoded[entry["file"]] = decode_events(event, time)
        starts = time[:, 0] / 1000
        gaps = np.diff(starts)
        pauses = gaps[gaps > PAUSE_S]
        rows.append({
            **entry,
            "header_time": header_time(mat),
            "mat_version": "7.3" if mat.get("__v73__") else "5",
            "n_rows": event.shape[0],
            "n_cols": event.shape[1],
            "clock_start_s": starts[0],
            "clock_end_s": time[time > 0].max() / 1000,
            "clock_backwards": int((gaps < 0).sum()),
            "n_pauses": len(pauses),
            "longest_pause_min": pauses.max() / 60 if len(pauses) else 0.0,
            "valid_pct": 100 * decoded[entry["file"]].is_valid.mean() if entry["file"] in decoded else np.nan,
            "content_md5": hashlib.md5(event.tobytes() + time.tobytes()).hexdigest(),
        })
    blocks = pd.DataFrame(rows)
    blocks["duration_min"] = (blocks.clock_end_s - blocks.clock_start_s) / 60
    blocks["use"] = blocks.n_cols == 10
    return blocks, decoded


def batch_saves(header_times):
    """True for files whose header falls in a cluster of `BATCH_SAVE_MIN_FILES` or more saved within `BATCH_SAVE_S`
    of each other: they were re-saved together, so the header dates the batch, not the recording."""
    ordered = header_times.sort_values()
    cluster = (ordered.diff().dt.total_seconds() > BATCH_SAVE_S).cumsum()
    return (cluster.map(cluster.value_counts()) >= BATCH_SAVE_MIN_FILES).reindex(header_times.index)


def stretch_priors(trials, size=STRETCH):
    """The prior implied by each consecutive stretch of `size` colour-shown trials (the last only if half full)."""
    shown = trials.dropna(subset=["color", "target"]).reset_index(drop=True)
    priors = []
    for start in range(0, len(shown), size):
        part = shown.iloc[start:start + size]
        if len(part) >= size // 2:
            right = part.target.eq("right").groupby(part.color).mean() * 100
            priors.append(implied_prior(right.get("green", 50), right.get("red", 50)))
    return priors


def session_file_summary(blocks):
    """One row per whole-file session (exactly one trial file each, any practice files), in listing order."""
    rows = []
    for session_id, session in blocks.groupby("session_id", sort=False):
        trial, practice = session[session.phase == "trial"], session[session.phase == "practice"]
        if len(trial) != 1:
            raise ValueError(f"{session_id}: {len(trial)} trial files, expected one")
        t = trial.iloc[0]
        stretches = t.stretch_priors.split(" | ") if t.stretch_priors else []
        row = {
            "session_id": session_id, "subject_id": t.subject_id, "group": t.group, "medication": t.medication,
            "header_time": t.header_time, "header_is_batch_save": bool(t.header_is_batch_save), "duration_min": t.duration_min,
            "n_pauses": t.n_pauses, "longest_pause_min": t.longest_pause_min, "clock_backwards": t.clock_backwards,
            "n_trials": t.n_rows, "n_shown": t.n_shown, "valid_pct": t.valid_pct, "session_prior": t.block_prior,
            "stretch_priors": " | ".join(stretches), "prior_consistent": len(set(stretches)) <= 1,
            "max_design_error": t.design_error, "design_ok": t.design_error <= DESIGN_TOLERANCE,
            "n_trial_blocks": 1, "n_practice_blocks": len(practice),
            "practice_before_trial": bool(practice.clock_end_s.max() <= t.clock_start_s) if len(practice) else np.nan,
            "practice_to_trial_min": (t.clock_start_s - practice.clock_end_s.max()) / 60 if len(practice) else np.nan,
            "reaches_600": t.n_shown >= MIN_SESSION_TRIALS,
        }
        issues = []
        if row["clock_backwards"]:
            issues.append(f"clock runs backwards {row['clock_backwards']} time(s)")
        if not row["prior_consistent"]:
            issues.append(f"prior changes ({row['stretch_priors']})")
        if not row["design_ok"]:
            issues.append(f"design off 75:25 / 50:50 (error {row['max_design_error']:.2f})")
        if not len(practice):
            issues.append("no practice file")
        elif not row["practice_before_trial"]:
            issues.append("practice not before trial on the clock")
        if not row["reaches_600"]:
            issues.append(f"{row['n_shown']:.0f} trials")
        row["issues"] = "; ".join(issues)
        rows.append(row)
    return pd.DataFrame(rows)


def add_stretch_priors(blocks, decoded):
    """Add `stretch_priors`: each file's stretch priors, joined with " | "."""
    blocks["stretch_priors"] = blocks.file.map(lambda f: " | ".join(stretch_priors(decoded[f])) if f in decoded else "")
    return blocks


def prior_parts(prior):
    """(`prior_color`, `prior_direction`) for a single-colour prior such as `gl`; (None, None) otherwise."""
    if prior not in SINGLE_PRIORS:
        return None, None
    return {"g": "green", "r": "red"}[prior[0]], {"r": "right", "l": "left"}[prior[1]]


def select_blocks(trials, phase="trial", design_ok=None, matches_session_prior=None, priors=None, include_partial=True, relabel=False):
    """Trials of the blocks meeting every given condition (None leaves a condition out).

    `trials` is a compiled block-session table (`catalog.load_block_sessions`).

    - `phase`: "trial" (default), "practice", or None for both;
    - `design_ok`: True keeps blocks run as the intended design (75:25 on one colour, 50:50 on the other), dropping
      two-prior and no-prior blocks; False keeps only those;
    - `matches_session_prior`: True keeps blocks implying the session's prior (the one most of its complete blocks
      show), False the others (e.g. P3 ON's first block, `gl` in a `gr` session);
    - `priors`: block priors to keep, e.g. {"gr+rl"};
    - `include_partial`: keep partial (aborted) blocks;
    - `relabel`: set `prior`, `prior_color` and `prior_direction` to each block's own prior, so the selection can be
      analysed as a condition of its own.

    Examples: `select_blocks(t, design_ok=True)` for every correctly run block; `select_blocks(t, priors={"gr+rl"})`
    for the two-prior blocks; `select_blocks(t, matches_session_prior=False, design_ok=True, relabel=True)` for the
    correctly run blocks that used another prior than the rest of their session.
    """
    keep = pd.Series(True, index=trials.index)
    if phase is not None:
        keep &= trials.phase == phase
    if design_ok is not None:
        keep &= trials.block_design_ok == design_ok
    if matches_session_prior is not None:
        keep &= trials.block_matches_session_prior == matches_session_prior
    if priors is not None:
        keep &= trials.block_prior.isin(priors)
    if not include_partial:
        keep &= ~trials.block_partial
    selected = trials[keep].copy()
    if relabel:
        selected["prior"] = selected.block_prior
        selected["prior_color"], selected["prior_direction"] = selected.block_prior_color, selected.block_prior_direction
    return selected


# --------------------------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------------------------- #

PRACTICE_COLOUR, TRIAL_COLOUR = "#eb6834", "#2a78d6"  # categorical slots 2 and 1 of the reference palette


def plot_timeline(blocks, summary, title):
    """Each used block at its wall-clock time (header time minus its recorded duration), in minutes from the session's
    first trial block. Trial blocks carry their implied prior; hatched blocks are partial."""
    used = blocks[blocks.use].copy()
    used["start"] = used.header_time - pd.to_timedelta(used.duration_s, unit="s")
    t0 = used[used.phase == "trial"].groupby("session_id").start.min()
    used["start_min"] = (used.start - used.session_id.map(t0)).dt.total_seconds() / 60
    used["length_min"] = used.duration_s / 60
    order = summary.session_id.tolist()

    fig, ax = plt.subplots(figsize=(11, 0.28 * len(order) + 1))
    for y, session_id in enumerate(order):
        for block in used[used.session_id == session_id].itertuples():
            colour = PRACTICE_COLOUR if block.phase == "practice" else TRIAL_COLOUR
            ax.broken_barh([(block.start_min, block.length_min)], (y - 0.35, 0.7), facecolor=colour, edgecolor="white",
                           linewidth=1, hatch="////" if block.partial else None)
            if block.phase == "trial" and block.length_min > 3:
                ax.text(block.start_min + block.length_min / 2, y, block.block_prior, ha="center", va="center", fontsize=6.5, color="white")
    ax.axvline(0, color="0.6", linewidth=0.8, zorder=0)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("minutes from the first trial block", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PRACTICE_COLOUR), plt.Rectangle((0, 0), 1, 1, color=TRIAL_COLOUR),
                       plt.Rectangle((0, 0), 1, 1, facecolor=TRIAL_COLOUR, hatch="////", edgecolor="white")],
              labels=["practice", "trial", "partial trial block"], loc="lower right", frameon=False, fontsize=8)
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def _names(sessions):
    return ", ".join(sessions) if len(sessions) else "none"


def findings(blocks, summary, runs):
    """The headline results as sentences, generated from the tables."""
    s = summary.set_index("session_id")
    trials = s[s.n_trial_blocks > 0]
    reasons = blocks.loc[~blocks.use, "excluded_reason"].str.split(" \\(").str[0].value_counts()
    lines = [f"{len(blocks)} files in {len(s)} sessions; {blocks.use.sum()} used, {(~blocks.use).sum()} left out"
             + (" (" + ", ".join(f"{n} {reason}" for reason, n in reasons.items()) + ")." if len(reasons) else ".")]
    for run in runs.itertuples():
        when = f"{run.saved_from:%Y-%m-%d %H:%M}" + (f"-{run.saved_to:%H:%M}" if run.saved_to > run.saved_from else "")
        copies = f"; copies of {run.copies_of}" if run.copies_of else ""
        lines.append(f"{run.n_files} left-out file(s) ({run.reasons}) saved {when}, on the clock of {run.recorded_with}, "
                     f"filed under {run.filed_under}{copies}.")
    lines += [
        f"Sessions without trial blocks: {_names(s.index[s.n_trial_blocks == 0])}.",
        f"No practice recorded in-session: {_names(trials.index[trials.n_practice_blocks == 0])}.",
        f"Practice not finished before the first trial block: {_names(trials.index[(trials.n_practice_blocks > 0) & ~trials.practice_before_trial.astype(bool)])}.",
        f"Blocks out of order or trial blocks overlapping: {_names(trials.index[~trials.block_order_consistent.astype(bool) | trials.trial_blocks_overlap.astype(bool)])}.",
        f"Blocks on more than one day: {_names(trials.index[~trials.one_day.astype(bool)])}.",
        "Clock shifts within a session (sleep or clock correction; blocks kept): "
        + ("; ".join(f"{i} (up to {m:.1f} min)" for i, m in trials.max_clock_shift_min.items() if m > 0) or "none") + ".",
        "Block priors differ within the session: "
        + ("; ".join(f"{i} ({b})" for i, b in trials.loc[~trials.prior_consistent.astype(bool), "block_priors"].items()) or "none") + ".",
        "Design off 75:25 / 50:50 in a complete block: "
        + ("; ".join(f"{i} ({p})" for i, p in trials.loc[~trials.design_ok.astype(bool), "session_prior"].items()) or "none") + ".",
        "Partial trial blocks: "
        + ("; ".join(f"{r.file} ({r.n_shown:.0f} trials)" for r in blocks[blocks.use & blocks.partial].itertuples()) or "none") + ".",
        f"Under {MIN_SESSION_TRIALS} trials: {_names(trials.index[~trials.reaches_600.astype(bool)])}; "
        f"range {trials.n_shown.min():.0f}-{trials.n_shown.max():.0f}, median {trials.n_shown.median():.0f}.",
    ]
    return lines
