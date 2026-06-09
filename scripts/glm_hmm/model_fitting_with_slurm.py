"""Slurm-aware launcher around :mod:`model_fitting`.

The fitting work is embarrassingly parallel across ``(config, group)`` pairs: each
pair runs its own shared global fit once and then the requested CV modes, writing a
self-contained ``<group>.pkl``. This wrapper enumerates those pairs as a flat,
deterministic list of *work units* and lets a single unit be selected by index via
``--job_id`` -- exactly what a Slurm array task needs (``$SLURM_ARRAY_TASK_ID``).

It reuses the building blocks from :mod:`model_fitting` (data loading, the shared
global fit, per-mode CV + persistence), so the actual modelling lives in one place.

Run modes
---------
* ``--list-jobs``            print the index -> (config, group) mapping and exit;
                             the count tells you the Slurm ``--array`` range.
* ``--job_id N``             run only work unit N (one Slurm array task).
* neither flag              run every unit in sequence (local full run, like
                             ``model_fitting.py`` itself).

Resume: a unit's CV mode is skipped if its ``<group>.pkl`` already exists and is
non-empty, unless ``--force`` -- so a preempted/requeued array task picks up where it
left off instead of recomputing finished groups.

Examples
--------
    python3 scripts/glm_hmm/model_fitting_with_slurm.py --config normalized_stimulus --list-jobs
    python3 scripts/glm_hmm/model_fitting_with_slurm.py --config normalized_stimulus standardized_stimulus --job_id 3
    python3 scripts/glm_hmm/model_fitting_with_slurm.py            # all configs, all groups, locally
"""

import argparse
from pathlib import Path

import numpy as np

import model_fitting as mf


def cv_output_dir(processed_dir: Path, name: str, cv_mode: str) -> Path:
    """Output directory for one (config, cv_mode).

    Mirrors the suffix convention in ``model_fitting.main`` so the wrapper and the
    standalone script write to (and resume from) the same locations. ``name`` is the
    experiment key the config is registered under in ``glm_hmm.experiments.CONFIGS``.
    """
    dir_suffix = "__global_pooled_cv" if cv_mode == "pooled" else "__session_pooled_cv"
    return processed_dir / "glm_hmm" / f"{name}{dir_suffix}"


def unit_done(output_dir: Path, group_name: str) -> bool:
    """True if a complete (non-empty) output pickle already exists for this group."""
    pkl = output_dir / f"{group_name}.pkl"
    return pkl.exists() and pkl.stat().st_size > 0


def run_unit(config, name: str, group_name: str, session_ids: list, data, cv_modes: list, processed_dir: Path, force: bool):
    """Fit one (config, group) work unit across the requested CV modes.

    Computes the shared global fit once (:func:`model_fitting.prepare_group`) and reuses
    it across modes. Skips modes whose output already exists unless ``force``. ``name`` is
    the experiment key the config is registered under. Returns the chosen state count when
    pooled CV ran, otherwise ``None``.
    """
    output_dirs = {}
    for cv_mode in cv_modes:
        output_dir = cv_output_dir(processed_dir, name, cv_mode)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[cv_mode] = output_dir

    pending = cv_modes if force else [m for m in cv_modes if not unit_done(output_dirs[m], group_name)]
    if not pending:
        print(f"  skip {name} / {group_name}: all CV modes already on disk")
        return None

    print(f"Preparing {name} / {group_name} group (global fit) with sessions: {session_ids}")
    prep = mf.prepare_group(data, session_ids, config)

    best = None
    for cv_mode in pending:
        print(f"  -> {cv_mode} CV for {group_name}")
        result = mf.run_cv_for_group(prep, config, name, cv_mode, output_dirs[cv_mode], group_name)

        if cv_mode == "pooled":
            selection = mf.select_best_n_states(result, prep["session_data"], rule=config.selection_rule, tol=config.selection_tol)
            best = selection["best"]
            print(f"     best = {selection['best']} states (peak {selection['best_unconstrained']}, rule={selection['rule']}); bits/trial: {np.round(selection['mean_bits'], 4).tolist()}")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        nargs="*",
        choices=list(mf.CONFIGS),
        default=None,
        help="Aliases of configs to fit (space-separated); omit to fit every config in the registry.",
    )
    parser.add_argument(
        "--cv-mode",
        nargs="+",
        choices=["session", "pooled"],
        default=["session", "pooled"],
        help="Cross-validation strategies to run (space-separated); omit to run both.",
    )
    parser.add_argument(
        "--job_id",
        type=int,
        default=None,
        help="Run only the single (config, group) work unit at this index (for Slurm array jobs). See --list-jobs for the mapping. Omit to run every unit in sequence.",
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="Print the (config, group) work-unit enumeration and exit; the count sets the Slurm --array range.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even when an output .pkl already exists (default: skip completed CV modes for resumability).",
    )
    args = parser.parse_args()

    config_names = list(args.config or mf.CONFIGS)

    processed_dir = Path(mf.dir_config.data.processed)
    data, metadata = mf.load_data(processed_dir)
    subject_groups = mf.get_group_session_ids(metadata)

    # Deterministic flat work-unit list: config order (CLI/registry) x group order
    # (dict insertion). The same --config string yields the same index -> unit mapping
    # across all array tasks, so --job_id always picks a well-defined unit.
    jobs = [(name, gname, sids) for name in config_names for gname, sids in subject_groups.items()]

    if args.list_jobs:
        print(f"{len(jobs)} work units (set Slurm --array=0-{len(jobs) - 1}):")
        for idx, (name, gname, _) in enumerate(jobs):
            print(f"  {idx}: {name} / {gname}")
        return

    if args.job_id is not None:
        if not 0 <= args.job_id < len(jobs):
            parser.error(f"--job_id {args.job_id} out of range [0, {len(jobs) - 1}] for {len(jobs)} work units")
        name, group_name, session_ids = jobs[args.job_id]
        print(f"\n=== job {args.job_id}/{len(jobs) - 1}: {name} / {group_name} (cv_modes={args.cv_mode}) ===")
        run_unit(mf.CONFIGS[name], name, group_name, session_ids, data, args.cv_mode, processed_dir, args.force)
        return

    # Full run: every unit in sequence (local use). Aggregate pooled selections per config.
    for name in config_names:
        config = mf.CONFIGS[name]
        print(f"\n=== Fitting config: {name} (cv_modes={args.cv_mode}) ===")
        best_states = {}
        for group_name, session_ids in subject_groups.items():
            best = run_unit(config, name, group_name, session_ids, data, args.cv_mode, processed_dir, args.force)
            if best is not None:
                best_states[group_name] = best
        if "pooled" in args.cv_mode and best_states:
            print(f"\nbest_states ({config.selection_rule}) for {name} = {best_states}")


if __name__ == "__main__":
    main()
