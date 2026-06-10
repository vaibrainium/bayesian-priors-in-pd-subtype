"""Finetune GLM-HMM models on the full data after cross-validated state selection.

For each CV pickle produced by ``model_fitting.py``, this script:

1. Reads the stored ``best_k`` (or recomputes it via the 1-SEM rule).
2. Refits one model per session at ``best_k``, warm-started from the best CV fold.
3. For pooled-CV bundles, also fits one *pooled* model over all sessions at ``best_k``
   (clean group-level weights, stored under ``"pooled"``).

Output is written beside the CV pickle as ``<group>_final.pkl`` with keys:
  ``model``   -- dict with ``models``, ``fit_lls``, ``best_folds`` (+ ``pooled`` for pooled-CV)
  ``best_k``  -- chosen state count
  ``cv_type`` -- ``"pooled_cv"`` or ``"session_cv"``
  ``data``    -- per-session DataFrames (from the source bundle)
  ``config``  -- model config dict (from the source bundle)

Run modes
---------
* ``--list-jobs``   print the index -> (model_dir, group) mapping and exit.
* ``--job_id N``    process only work unit N (for Slurm array jobs).
* neither flag      process every unit in sequence (local full run).

Resume: a unit is skipped when its ``<group>_final.pkl`` already exists and is
non-empty, unless ``--force`` is passed.

Examples
--------
    python3 scripts/glm_hmm/model_finetuning.py --list-jobs
    python3 scripts/glm_hmm/model_finetuning.py --job_id 3
    python3 scripts/glm_hmm/model_finetuning.py --cv-mode pooled --n-iters 2500
    python3 scripts/glm_hmm/model_finetuning.py --force
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from config import dir_config
from src.glm_hmm.cv_utils import bernoulli_null_loglik, pooled_bits_per_trial
from src.glm_hmm.fitting_utils import group_wise_fit, session_wise_fit

DESIRED_ORDER = ["asmHC", "Tremor_OFF", "Brady_OFF", "Tremor_ON", "Brady_ON"]
_ORDER_MAP = {name: i for i, name in enumerate(DESIRED_ORDER)}


# --------------------------------------------------------------------------- #
# CV scoring helpers
# --------------------------------------------------------------------------- #
def _session_bits(bundle):
    """Session-wise CV bits/trial: per-session scores averaged across sessions."""
    sw = bundle["session_wise"]
    states = np.array(list(sw["models"][0].keys()))
    n_trials, null = [], []
    for df in bundle["data"].values():
        valid = ~df["invalid_idx"].values
        y = df["choices"].values[valid]
        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        n_trials.append(int(valid.sum()))
        null.append((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
    n_trials, null = np.array(n_trials), np.array(null)
    bits = (sw["test_ll"].sum(2) - null[:, None]) / (n_trials[:, None] * np.log(2))
    return states, bits.mean(0), bits.std(0) / np.sqrt(bits.shape[0])


def _pooled_bits(bundle):
    """Pooled CV bits/trial: one held-out pass vs the Bernoulli null."""
    gpc = bundle["group_pooled_cv"]
    null_ll, n_valid = bernoulli_null_loglik(bundle["data"])
    mean_bits, sem_bits = pooled_bits_per_trial(gpc["test_ll"], gpc["n_test_trials"], null_ll, n_valid)
    return np.asarray(gpc["state_range"]), mean_bits, sem_bits


def _parsimonious_k(states, mean_bits, sem_bits):
    """Smallest K within one fold-SEM of the peak (the 1-SEM rule)."""
    best_idx = int(np.argmax(mean_bits))
    cutoff = mean_bits[best_idx] - sem_bits[best_idx]
    chosen_idx = next(i for i in range(best_idx + 1) if mean_bits[i] >= cutoff)
    return int(states[chosen_idx])


def get_best_k(bundle, cv_type):
    """Return the stored best_k, or recompute via the 1-SEM rule if absent."""
    if bundle.get("best_k") is not None:
        return int(bundle["best_k"])
    bits_fn = _pooled_bits if cv_type == "pooled_cv" else _session_bits
    states, mean_bits, sem_bits = bits_fn(bundle)
    return _parsimonious_k(states, mean_bits, sem_bits)


# --------------------------------------------------------------------------- #
# Design-matrix reconstruction
# --------------------------------------------------------------------------- #
def session_arrays(bundle):
    """Rebuild (observations, inputs, masks) from the saved per-session DataFrames."""
    feats = bundle["config"]["model_features"]
    observations, inputs, masks = [], [], []
    for df in bundle["data"].values():
        observations.append(df["choices"].values.reshape(-1, 1).astype(int))
        inputs.append(np.asarray(df[feats], dtype=float))
        masks.append(df["mask"].values.reshape(-1, 1).astype(bool))
    return observations, inputs, masks


def _prior_sigma(bundle):
    return bundle["config"].get("prior_sigma", 2.0)


# --------------------------------------------------------------------------- #
# Finetuning routines
# --------------------------------------------------------------------------- #
def _fit_per_session(bundle, best_k, init_params, best_folds, n_iters):
    sessions = list(bundle["data"].keys())
    observations, inputs, masks = session_arrays(bundle)
    models_s, fit_lls_s = session_wise_fit(
        observations,
        inputs,
        masks,
        n_sessions=len(sessions),
        init_params=init_params,
        n_states=best_k,
        n_iters=n_iters,
        prior_sigma=_prior_sigma(bundle),
    )
    models = {session: models_s[idx] for idx, session in enumerate(sessions)}
    fit_lls = {session: fit_lls_s[idx] for idx, session in enumerate(sessions)}
    return {"models": models, "fit_lls": fit_lls, "best_folds": best_folds}


def finetune_session(bundle, best_k, n_iters=2500):
    """Per-session models; each warm-started from its own best-generalizing CV fold."""
    sw = bundle["session_wise"]
    sessions = list(bundle["data"].keys())
    state_idx = list(sw["models"][0].keys()).index(best_k)

    init_params = {"glm_weights": {}, "transition_matrices": {}}
    best_folds = {}
    for idx, session in enumerate(sessions):
        best_fold = int(np.nanargmax(sw["test_ll"][idx, state_idx, :]))
        m = sw["models"][idx][best_k][best_fold]
        init_params["glm_weights"][idx] = m.observations.params
        init_params["transition_matrices"][idx] = m.transitions.params
        best_folds[session] = best_fold

    return _fit_per_session(bundle, best_k, init_params, best_folds, n_iters)


def finetune_pooled(bundle, best_k, n_iters=2500):
    """Per-session models + one pooled model, all warm-started from the group's best pooled fold."""
    gpc = bundle["group_pooled_cv"]
    state_idx = int(np.where(np.asarray(gpc["state_range"]) == best_k)[0][0])
    best_fold = int(np.nanargmax(gpc["test_ll"][state_idx]))
    init = gpc["models"][best_k][best_fold]
    init_flat = {
        "glm_weights": init.observations.params,
        "transition_matrices": init.transitions.params,
    }

    sessions = list(bundle["data"].keys())
    init_params = {
        "glm_weights": {idx: init.observations.params for idx in range(len(sessions))},
        "transition_matrices": {idx: init.transitions.params for idx in range(len(sessions))},
    }
    best_folds = {session: best_fold for session in sessions}
    result = _fit_per_session(bundle, best_k, init_params, best_folds, n_iters)

    observations, inputs, masks = session_arrays(bundle)
    pooled_model, pooled_fit_ll = group_wise_fit(
        observations,
        inputs,
        masks,
        init_flat,
        n_states=best_k,
        n_iters=n_iters,
        prior_sigma=_prior_sigma(bundle),
    )
    result["pooled"] = pooled_model
    result["pooled_fit_ll"] = pooled_fit_ll
    return result


# --------------------------------------------------------------------------- #
# Work-unit enumeration
# --------------------------------------------------------------------------- #
def _cv_type(model_dir_name: str) -> str:
    return "pooled_cv" if model_dir_name.endswith("__global_pooled_cv") else "session_cv"


def enumerate_jobs(glm_hmm_dir: Path, cv_mode_filter=None) -> list[tuple[Path, Path]]:
    """Return a deterministic list of (model_dir, group_pkl) pairs to finetune.

    ``cv_mode_filter`` is an optional set of ``{"pooled", "session"}`` to restrict which
    CV types are included. Model dirs are discovered alphabetically; within each dir,
    group pickles are sorted by the preferred group order (DESIRED_ORDER).
    """
    jobs = []
    for model_dir in sorted(glm_hmm_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        ctype = _cv_type(model_dir.name)
        if cv_mode_filter and ctype.split("_")[0] not in cv_mode_filter:
            continue
        for pkl in sorted(model_dir.glob("*.pkl"), key=lambda p: _ORDER_MAP.get(p.stem, float("inf"))):
            if not pkl.stem.endswith("_final"):
                jobs.append((model_dir, pkl))
    return jobs


# --------------------------------------------------------------------------- #
# Core processing
# --------------------------------------------------------------------------- #
def unit_done(group_pkl: Path) -> bool:
    out = group_pkl.parent / f"{group_pkl.stem}_final.pkl"
    return out.exists() and out.stat().st_size > 0


def run_unit(model_dir: Path, group_pkl: Path, n_iters: int, force: bool) -> None:
    """Finetune one (model_dir, group) pair and save ``<group>_final.pkl``."""
    out_path = model_dir / f"{group_pkl.stem}_final.pkl"
    if not force and unit_done(group_pkl):
        print(f"  skip {model_dir.name} / {group_pkl.stem}: _final.pkl already on disk")
        return

    with open(group_pkl, "rb") as f:
        bundle = pickle.load(f)

    ctype = _cv_type(model_dir.name)
    best_k = get_best_k(bundle, ctype)
    finetune = finetune_pooled if ctype == "pooled_cv" else finetune_session

    print(f"  {group_pkl.stem:12s} best_k={best_k}  [{ctype}]  fitting {n_iters} iters ...")
    finetuned = finetune(bundle, best_k, n_iters=n_iters)

    payload = {
        "model": finetuned,
        "best_k": best_k,
        "cv_type": ctype,
        "data": bundle["data"],
        "config": bundle["config"],
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    extra = " + pooled" if "pooled" in finetuned else ""
    print(f"    -> {len(finetuned['models'])} session models{extra}  saved: {out_path.name}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cv-mode",
        nargs="+",
        choices=["session", "pooled"],
        default=None,
        help="Restrict to these CV types (space-separated); omit to process both.",
    )
    parser.add_argument(
        "--job_id",
        type=int,
        default=None,
        help="Run only work unit N (for Slurm array jobs; see --list-jobs).",
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="Print the index -> (model_dir, group) mapping and exit.",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=2500,
        help="EM iterations for the finetuning fits (default: 2500).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refit even when a _final.pkl already exists.",
    )
    args = parser.parse_args()

    glm_hmm_dir = Path(dir_config.data.processed) / "glm_hmm"
    cv_mode_filter = set(args.cv_mode) if args.cv_mode else None
    jobs = enumerate_jobs(glm_hmm_dir, cv_mode_filter)

    if args.list_jobs:
        print(f"{len(jobs)} work units (set Slurm --array=0-{len(jobs) - 1}):")
        for idx, (model_dir, pkl) in enumerate(jobs):
            print(f"  {idx:3d}: {model_dir.name} / {pkl.stem}")
        return

    if args.job_id is not None:
        if not 0 <= args.job_id < len(jobs):
            parser.error(f"--job_id {args.job_id} out of range [0, {len(jobs) - 1}]")
        model_dir, group_pkl = jobs[args.job_id]
        print(f"\n=== job {args.job_id}/{len(jobs) - 1}: {model_dir.name} / {group_pkl.stem} ===")
        run_unit(model_dir, group_pkl, args.n_iters, args.force)
        return

    # Full sequential run.
    current_dir = None
    for model_dir, group_pkl in jobs:
        if model_dir != current_dir:
            print(f"\n=== {model_dir.name} ===")
            current_dir = model_dir
        run_unit(model_dir, group_pkl, args.n_iters, args.force)


if __name__ == "__main__":
    main()
