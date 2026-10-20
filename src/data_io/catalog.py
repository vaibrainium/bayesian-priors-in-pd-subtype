"""Named datasets: every file that more than one notebook or script touches, in one place.

Notebooks ask for data by name (`load_trials()`, `load_metadata()`, `load("ddm_behavior")`) and hand results back
the same way (`save(df, "metadata_updrs")`, `save_table(df, "subtype_comparison.csv")`), so file
names, directories and read options live here instead of being retyped in each notebook. A missing
file names the notebook that produces it.

Session metadata is built by several pipeline stages. Each stage writes its own file, keyed by
`session_id`, and `load_metadata` joins them. No stage rewrites another stage's file, so re-running
one stage can never silently drop or corrupt the columns another stage added.

Compiled and processed data keep every session, whatever its accuracy. `load_trials` and
`load_metadata` read either stage (`stage="compiled"` or `"processed"`) and keep only subjects whose
sessions all reach `min_easy_accuracy` on valid 100%-coherence trials (the config value unless
given), and whose sessions all ran the intended design: one 75:25 colour and one 50:50 colour
(`prior_design_tolerance`, see `src.utils.sessions.prior_design_error`). Changing either threshold or the
stage needs no re-run of the pipeline.
"""

import json
import pickle
from functools import lru_cache
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import main_config
from src.data_io.paths import ROOTS, writable
from src.utils.sessions import easy_accuracy_pct, prior_design_error, valid_trials

DEFAULT_MIN_EASY_ACCURACY = main_config.BEHAVIOR.min_easy_accuracy
DEFAULT_PRIOR_DESIGN_TOLERANCE = main_config.BEHAVIOR.prior_design_tolerance
PRIOR_BASE_RATES = tuple(main_config.BEHAVIOR.prior_base_rates)
MIN_RT = main_config.BEHAVIOR.min_rt
STAGES = ("compiled", "processed")


@dataclass(frozen=True)
class Dataset:
    root: str  # a key of paths.ROOTS
    pattern: str  # path relative to the root; may use caller-supplied fields
    produced_by: str | None = None  # notebook or script that writes it (None for raw inputs)
    read_kwargs: dict = field(default_factory=dict)
    write_kwargs: dict = field(default_factory=dict)
    screened: bool = False  # holds every session; read through load_trials / load_metadata


_CSV_NO_INDEX = {"index": False}
# Derived CSVs are written with full float precision; read them back bit for bit (pandas' default
# parser can be off in the last bit, which made a value drift each time a stage re-read it).
_CSV_EXACT = {"float_precision": "round_trip"}

DATASETS = {
    # raw inputs (read only)
    "raw_session_metadata": Dataset("raw", "session_metadata_detailed_all_data.csv", read_kwargs={"encoding": "latin1"}),
    "raw_session_file": Dataset("raw", "{filename}"),  # per-session .mat files (UCLA, Stanford, Case Western)
    "raw_harvard_sessions": Dataset("raw", "harvard_data/{subject_type}"),
    "raw_stanford_sessions": Dataset("raw", "stanford_data"),
    "raw_case_western_sessions": Dataset("raw", "case_western_data"),
    "raw_ucla_sessions": Dataset("raw", "ucla_data"),
    "raw_perugini": Dataset("raw", "ucla_data/Perugini_2016"),
    "raw_perugini_keypress": Dataset("raw", "ucla_data/Perugini_2016/Reports_HC_Exp2_KeyPress"),
    "raw_perugini_eye": Dataset("raw", "ucla_data/Perugini_2016/Reports_HC_Exp2"),
    # compiled: one trial table per session, then the aggregate tables
    "compiled_session": Dataset("compiled", "sessions/{filename}", "0.00 / 0.01"),
    # Harvard, compiled from the raw block files with timing and design checks
    "compiled_harvard_session": Dataset("compiled", "harvard/sessions/{filename}", "0.02"),
    "compiled_harvard_blocks": Dataset("compiled", "harvard/blocks.csv", "0.02", read_kwargs={**_CSV_EXACT, "parse_dates": ["filename_time", "header_time"]}, write_kwargs=_CSV_NO_INDEX),
    "compiled_harvard_session_summary": Dataset("compiled", "harvard/session_summary.csv", "0.02", read_kwargs=_CSV_EXACT, write_kwargs=_CSV_NO_INDEX),
    # Stanford, the same way
    "compiled_stanford_session": Dataset("compiled", "stanford/sessions/{filename}", "0.03"),
    "compiled_stanford_blocks": Dataset("compiled", "stanford/blocks.csv", "0.03", read_kwargs={**_CSV_EXACT, "parse_dates": ["filename_time", "header_time"]}, write_kwargs=_CSV_NO_INDEX),
    "compiled_stanford_session_summary": Dataset("compiled", "stanford/session_summary.csv", "0.03", read_kwargs=_CSV_EXACT, write_kwargs=_CSV_NO_INDEX),
    # Case Western: one file per session, compiled as a single block
    "compiled_case_western_session": Dataset("compiled", "case_western/sessions/{filename}", "0.04"),
    "compiled_case_western_blocks": Dataset("compiled", "case_western/blocks.csv", "0.04", read_kwargs={**_CSV_EXACT, "parse_dates": ["header_time"]}, write_kwargs=_CSV_NO_INDEX),
    "compiled_case_western_session_summary": Dataset("compiled", "case_western/session_summary.csv", "0.04", read_kwargs=_CSV_EXACT, write_kwargs=_CSV_NO_INDEX),
    # UCLA: one file per session (equal-prior warm-up file as practice), regular and explicit sessions
    "compiled_ucla_session": Dataset("compiled", "ucla/sessions/{filename}", "0.05"),
    "compiled_ucla_blocks": Dataset("compiled", "ucla/blocks.csv", "0.05", read_kwargs={**_CSV_EXACT, "parse_dates": ["header_time"]}, write_kwargs=_CSV_NO_INDEX),
    "compiled_ucla_session_summary": Dataset("compiled", "ucla/session_summary.csv", "0.05", read_kwargs=_CSV_EXACT, write_kwargs=_CSV_NO_INDEX),
    "compiled_readme": Dataset("compiled", "README.md", "scripts/pipeline/write_compiled_readme.py"),
    # Perugini et al. (2016) healthy controls, kept apart from the study data (not read by load_block_sessions)
    "compiled_perugini_session": Dataset("compiled", "ucla/perugini/sessions/{filename}", "0.05"),
    "compiled_perugini_session_summary": Dataset("compiled", "ucla/perugini/session_summary.csv", "0.05", read_kwargs=_CSV_EXACT, write_kwargs=_CSV_NO_INDEX),
    "compiled_trials": Dataset("compiled", "aggregate_all_data.csv", "0.10", read_kwargs=_CSV_EXACT, write_kwargs=_CSV_NO_INDEX, screened=True),
    "compiled_metadata": Dataset("compiled", "aggregate_all_metadata.csv", "0.10", read_kwargs=_CSV_EXACT, write_kwargs=_CSV_NO_INDEX, screened=True),
    # processed trials: every trial, and the valid trials above the reaction-time floor
    "trials_all": Dataset("processed", "trials/all_trials.csv", "0.20", read_kwargs=_CSV_EXACT, write_kwargs=_CSV_NO_INDEX, screened=True),
    "trials_valid": Dataset("processed", "trials/valid_trials.csv", "0.20", read_kwargs=_CSV_EXACT, write_kwargs=_CSV_NO_INDEX, screened=True),
    # processed session metadata, one file per stage
    "metadata_sessions": Dataset("processed", "metadata/sessions.csv", "0.20", read_kwargs=_CSV_EXACT, write_kwargs=_CSV_NO_INDEX, screened=True),
    "metadata_updrs": Dataset("processed", "metadata/updrs.csv", "0.30", read_kwargs=_CSV_EXACT, write_kwargs=_CSV_NO_INDEX),
    "metadata_psych_params": Dataset("processed", "metadata/psych_params.csv", "0.40", read_kwargs=_CSV_EXACT, write_kwargs=_CSV_NO_INDEX),
    "metadata_ddm_params": Dataset("processed", "metadata/ddm_params.csv", "3.40", read_kwargs=_CSV_EXACT, write_kwargs=_CSV_NO_INDEX),
    # DDM
    "ddm_behavior": Dataset("processed", "ddm/behavior_data.csv", "3.10", read_kwargs=_CSV_EXACT, write_kwargs=_CSV_NO_INDEX),
    "ddm_fits": Dataset("processed", "ddm", "scripts/ddm/ddm_fitting.py"),  # one subdirectory per model variant
    # shared-basis GLM-HMM
    "shared_glm_hmm_cv": Dataset("processed", "shared_glm_hmm/{run}/all_subjects.pkl", "scripts/shared_glm_hmm/fit_shared_basis.py"),
    "shared_glm_hmm_final": Dataset("processed", "shared_glm_hmm/{run}/all_subjects_final.pkl", "scripts/shared_glm_hmm/finetune_shared_basis.py"),
    "shared_glm_hmm_iclbic": Dataset("processed", "shared_glm_hmm/{run}/selection_iclbic.csv", "scripts/shared_glm_hmm/compute_stability.py", read_kwargs={"index_col": 0}),
    "shared_glm_hmm_stability": Dataset("processed", "shared_glm_hmm/{run}/selection_stability.csv", "scripts/shared_glm_hmm/compute_stability.py", read_kwargs={"index_col": 0}),
    # analysis outputs that another notebook reads back
    "participant_medication_effects": Dataset("outputs", "participant_medication_effects.csv", "2.26"),
}

# Joined onto `metadata_sessions` by `load_metadata`, in this order.
METADATA_STAGES = ("metadata_updrs", "metadata_psych_params")
METADATA_KEY = "session_id"


def path(dataset, **fields):
    """Location of a named dataset. `fields` fill the placeholders in its pattern."""
    entry = DATASETS[dataset]
    return ROOTS[entry.root] / entry.pattern.format(**fields)


def files(dataset, **fields):
    """Files matching a dataset whose fields hold glob patterns, e.g. `files("compiled_session", filename="*.mat")`.

    Returned in filesystem order, not sorted: the compiled trial table is concatenated in this
    order, and sorting it would reorder every row downstream.
    """
    pattern = path(dataset, **fields)
    return list(pattern.parent.glob(pattern.name))


def output_path(filename, root="outputs"):
    """A writable path under `outputs/` (or another writable root), for writers with no wrapper here."""
    return writable(ROOTS[root] / filename)


# --------------------------------------------------------------------------------------------- #
# Reading and writing by file type
# --------------------------------------------------------------------------------------------- #


def read(file, **kwargs):
    """Read one file, choosing the reader from its suffix."""
    suffix = file.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(file, **kwargs)
    if suffix == ".xlsx":
        return pd.read_excel(file, **kwargs)
    if suffix == ".pkl":
        with open(file, "rb") as handle:
            return pickle.load(handle)
    if suffix == ".npz":
        with np.load(file, **kwargs) as arrays:
            return dict(arrays)
    if suffix == ".mat":
        import hdf5storage

        return hdf5storage.loadmat(file.as_posix(), **kwargs)
    if suffix == ".json":
        return json.loads(file.read_text())
    raise ValueError(f"no reader for {file.suffix} files ({file})")


def write(obj, file, **kwargs):
    """Write one object to a path inside the writable roots, choosing the writer from its suffix."""
    file = writable(file)
    suffix = file.suffix.lower()
    if suffix == ".csv":
        obj.to_csv(file, **kwargs)
    elif suffix == ".parquet":
        obj.to_parquet(file, **kwargs)
    elif suffix == ".pkl":
        with open(file, "wb") as handle:
            pickle.dump(obj, handle)
    elif suffix == ".npz":
        np.savez_compressed(file, **obj)
    elif suffix == ".mat":
        import hdf5storage

        hdf5storage.savemat(file.as_posix(), obj, **{"format": "7.3", **kwargs})
    elif suffix == ".json":
        file.write_text(json.dumps(obj, indent=2, **kwargs) + "\n")
    elif suffix in (".md", ".txt", ".html"):
        file.write_text(obj)
    else:
        raise ValueError(f"no writer for {file.suffix} files ({file})")
    return file


def _load(dataset, **fields):
    file = path(dataset, **fields)
    if not file.exists():
        source = DATASETS[dataset].produced_by
        hint = f"; it is written by {source}" if source else ""
        raise FileNotFoundError(f"{dataset} not found at {file}{hint}")
    return read(file, **DATASETS[dataset].read_kwargs)


def load(dataset, **fields):
    """Read a named dataset, with its registered read options."""
    if DATASETS[dataset].screened:
        raise ValueError(f"{dataset} holds every session; read it with load_trials or load_metadata, which apply the accuracy screen")
    return _load(dataset, **fields)


def save(obj, dataset, **fields):
    """Write a named dataset, with its registered write options."""
    file = write(obj, path(dataset, **fields), **DATASETS[dataset].write_kwargs)
    print(f"saved {dataset} -> {file}")


# --------------------------------------------------------------------------------------------- #
# Session metadata
# --------------------------------------------------------------------------------------------- #


def save_metadata_stage(metadata, name, columns=None, base=None):
    """Write one stage's metadata columns, keyed by session, without touching any other stage.

    Pass the stage's `columns`, or the `base` table the stage started from to save every column it
    added or changed.
    """
    if not metadata[METADATA_KEY].is_unique:
        raise ValueError(f"{name}: {METADATA_KEY} is not unique")
    if columns is None:
        if base is None:
            raise ValueError("pass either columns or base")
        if not metadata[METADATA_KEY].equals(base[METADATA_KEY]):
            raise ValueError(f"{name}: rows differ from the base table")
        columns = [c for c in metadata.columns if c not in base.columns or not metadata[c].equals(base[c])]
    columns = [c for c in columns if c != METADATA_KEY]
    print(f"{name}: {len(columns)} columns")
    save(metadata[[METADATA_KEY, *columns]], name)


def _join_stage(metadata, stage, name):
    """Add (or overwrite) the stage's columns, aligned on session_id and in the base row order."""
    if not stage[METADATA_KEY].is_unique:
        raise ValueError(f"{name}: {METADATA_KEY} is not unique")
    missing = set(metadata[METADATA_KEY]) - set(stage[METADATA_KEY])
    if missing:
        raise ValueError(
            f"{name} has no rows for {len(missing)} sessions: {sorted(missing)}. Re-run {DATASETS[name].produced_by}, "
            "or raise min_easy_accuracy if that stage was only run for the sessions that pass it."
        )
    aligned = stage.set_index(METADATA_KEY).reindex(metadata[METADATA_KEY])
    for column in aligned.columns:
        metadata[column] = aligned[column].to_numpy()
    return metadata


def passing_subjects(sessions, min_easy_accuracy=DEFAULT_MIN_EASY_ACCURACY, prior_design_tolerance=DEFAULT_PRIOR_DESIGN_TOLERANCE):
    """Subjects whose every session in `sessions` passes both load-time screens.

    `sessions` has one row per session with `subject_id`, `easy_accuracy_pct` and, for the design
    screen, `prior_design_error`. A session fails the accuracy screen when its accuracy (%) is below
    `min_easy_accuracy * 100`, and the design screen when either colour's presented targets depart
    from their intended split (75:25 for the prior colour, 50:50 for the equal colour) by more than
    `prior_design_tolerance`. Screening
    by subject rather than session keeps patients paired: a patient whose OFF or ON session fails is
    dropped entirely. `None` switches a screen off; a missing value (NaN) never fails a session.
    """
    failed = pd.Series(False, index=sessions.index)
    if min_easy_accuracy is not None:
        # The preprocessing's session-rejection test, as it has always been applied.
        failed |= sessions["easy_accuracy_pct"] < min_easy_accuracy * 100
    if prior_design_tolerance is not None:
        if "prior_design_error" not in sessions:
            raise ValueError("the design screen needs a prior_design_error column; pass prior_design_tolerance=None to skip it")
        failed |= sessions["prior_design_error"] > prior_design_tolerance
    return set(sessions["subject_id"]) - set(sessions.loc[failed, "subject_id"])


def _session_accuracy(stage):
    """One row per screened session (`subject_id`, `easy_accuracy_pct`, `prior_design_error`) for a stage.

    Processed: the accuracy column 0.20 stores. Compiled: computed from the session tables with the same
    function. Equal-prior sessions are not screened, as in processed data, which leaves them out;
    their subject passes or fails on the biased sessions. The design measure is computed from the
    stage's trial table at load time (see `_session_design`).
    """
    if stage == "processed":
        sessions = _load("metadata_sessions")[["subject_id", "file_name", "easy_accuracy_pct"]]
        key = "file_name"
    else:
        sessions = _compiled_session_accuracy()[["subject_id", "session_filename", "easy_accuracy_pct"]]
        key = "session_filename"
    design = _session_design(stage)
    sessions = sessions.assign(prior_design_error=sessions[key].map(design["prior_design_error"]).to_numpy())
    return sessions[["subject_id", "easy_accuracy_pct", "prior_design_error"]]


def _session_design(stage):
    """`prior_design_error` and its parts per `session_filename`, from every presented trial (cached per file version)."""
    file = path("trials_all" if stage == "processed" else "compiled_trials")
    return _session_design_cached(stage, file.stat().st_mtime_ns).copy()


@lru_cache(maxsize=4)
def _session_design_cached(stage, _mtime_ns):
    trials = _load("trials_all" if stage == "processed" else "compiled_trials")
    if stage == "compiled":
        trials = trials[trials["prior"] != "eq"]
    return prior_design_error(trials, PRIOR_BASE_RATES)


def _compiled_session_accuracy():
    """`subject_id`, `medication`, `easy_accuracy_pct` for each biased-prior compiled session."""
    trials = _load("compiled_trials")
    trials = trials[trials["prior"] != "eq"]
    accuracy = easy_accuracy_pct(trials, MIN_RT)
    sessions = trials.drop_duplicates("session_filename").set_index("session_filename")[["subject_id", "medication"]]
    sessions = sessions.join(accuracy).rename_axis("session_filename").reset_index()
    if sessions.duplicated(["subject_id", "medication"]).any():
        raise ValueError("more than one biased-prior session for a subject and medication state")
    return sessions


def _check_stage(stage):
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, not {stage!r}")


def load_metadata(stages=METADATA_STAGES, ddm_params=False, min_easy_accuracy=DEFAULT_MIN_EASY_ACCURACY, stage="processed", prior_design_tolerance=DEFAULT_PRIOR_DESIGN_TOLERANCE):
    """Session metadata for the subjects passing both screens (`None` for both keeps every subject).

    `stage="processed"` (default): one row per session, the 0.20 session table with each later
    stage's columns joined on. `stages` limits the join to the stages a pipeline step depends on
    (0.30 reads the bare session table, 0.40 needs the UPDRS subtypes), and `ddm_params=True` adds
    the per-session DDM parameters.

    `stage="compiled"`: the 0.10 metadata (raw session metadata plus experiment site), before any
    preprocessing, with each session's `easy_accuracy_pct` computed from the session tables; `stages`
    and `ddm_params` do not apply.

    The index is reset, so row labels are positions.
    """
    _check_stage(stage)
    keep = passing_subjects(_session_accuracy(stage), min_easy_accuracy, prior_design_tolerance)
    if stage == "compiled":
        if ddm_params:
            raise ValueError("DDM parameters are joined onto processed metadata only")
        metadata = _load("compiled_metadata")
        metadata = metadata[metadata["subject_id"].isin(keep)].reset_index(drop=True)
        # each row is one subject x treatment ("OFF" / "ON", missing for controls)
        accuracy = _compiled_session_accuracy().set_index(["subject_id", "medication"])["easy_accuracy_pct"]
        rows = pd.MultiIndex.from_arrays([metadata["subject_id"], metadata["treatment"].str.lower()])
        metadata["easy_accuracy_pct"] = accuracy.reindex(rows).to_numpy()
        return metadata
    metadata = _load("metadata_sessions")
    metadata = metadata[metadata["subject_id"].isin(keep)].reset_index(drop=True)
    for name in (*stages, *(["metadata_ddm_params"] if ddm_params else [])):
        metadata = _join_stage(metadata, _load(name), name)
    return metadata


def load_trials(kind="valid", min_easy_accuracy=DEFAULT_MIN_EASY_ACCURACY, stage="processed", prior_design_tolerance=DEFAULT_PRIOR_DESIGN_TOLERANCE):
    """Trial table for the subjects passing both screens (`None` for both keeps every subject).

    `kind="valid"`: valid trials above the reaction-time floor, the input to most analyses.
    `kind="all"`: every trial, including invalid ones (the GLM-HMM needs the full sequence).

    `stage="processed"` (default): after 0.20, i.e. without the MoCA-rejected subjects, the
    equal-prior sessions and incomplete patients, with target, choice and colour recoded relative to
    each session's prior. `stage="compiled"`: the decoded sessions as recorded (sides and colours
    as "left"/"right", "green"/"red"), every session included.
    """
    _check_stage(stage)
    if kind not in ("valid", "all"):
        raise ValueError(f"kind must be 'valid' or 'all', not {kind!r}")
    if stage == "compiled":
        trials = _load("compiled_trials")
        if kind == "valid":
            trials = valid_trials(trials, MIN_RT)
    else:
        trials = _load({"valid": "trials_valid", "all": "trials_all"}[kind])
    keep = passing_subjects(_session_accuracy(stage), min_easy_accuracy, prior_design_tolerance)
    return trials[trials["subject_id"].isin(keep)].reset_index(drop=True)


BLOCK_SITES = ("harvard", "stanford", "case_western", "ucla")  # sites compiled with block labels (0.02-0.05; Case Western and UCLA: one file per block)


def load_block_sessions(sites=BLOCK_SITES, session_ids=None):
    """The compiled trial tables of the block-recorded sites as one table, with a `site` column.

    Every session is included as recorded (no accuracy or design screen), each trial labelled with its block, so
    blocks can be picked with `src.utils.blocks.select_blocks`, e.g. only the correctly run blocks of a session that
    ran with two priors. `session_ids` (e.g. ["P33_off", "HC7"]) limits the sessions read.
    """
    tables = []
    for site in sites:
        for file in sorted(files(f"compiled_{site}_session", filename="*.parquet")):
            if session_ids is None or file.stem in session_ids:
                tables.append(read(file).assign(site=site))
    if not tables:
        raise FileNotFoundError(f"no compiled sessions for {sites} {session_ids or ''}; they are written by 0.02-0.05")
    return pd.concat(tables, ignore_index=True)


# --------------------------------------------------------------------------------------------- #
# Analysis outputs
# --------------------------------------------------------------------------------------------- #


def save_table(df, filename, **kwargs):
    """Write a results table to `outputs/`. `kwargs` go to `DataFrame.to_csv`."""
    write(df, ROOTS["outputs"] / filename, **kwargs)
    print(f"saved {filename}")


def save_arrays(filename, **arrays):
    """Write named arrays to a compressed `.npz` in `outputs/`."""
    write(arrays, ROOTS["outputs"] / filename)
    print(f"saved {filename}")


def save_text(text, filename):
    """Write a text or Markdown file to `outputs/`."""
    write(text, ROOTS["outputs"] / filename)
    print(f"saved {filename}")


def save_figure(fig, name, formats=("pdf", "png"), dpi=200, root="outputs"):
    """Save `fig` as `<name>.<format>` for each format, cropped to its contents.

    `dpi` applies to raster formats.
    """
    files = []
    for fmt in formats:
        file = writable(ROOTS[root] / f"{name}.{fmt}")
        fig.savefig(file, bbox_inches="tight", **({"dpi": dpi} if fmt == "png" else {}))
        files.append(file)
    print("saved", ", ".join(f.name for f in files))
