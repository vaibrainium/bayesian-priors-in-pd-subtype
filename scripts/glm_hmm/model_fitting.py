"""Fit GLM-HMM models per subject group (HC, PD subtype x medication state).

For each group this script builds session-wise design matrices and runs a global
fit to obtain initialisations, then cross-validates with one of two strategies
selected by ``--cv-mode`` and pickles the models together with the data and config:

* ``session`` (default) -- one GLM-HMM per session, cross-validated within each
  session. Output: ``glm_hmm/<name>/<group>.pkl`` (key ``session_wise``).
* ``pooled`` -- all of a group's sessions pooled into a single model per fold
  (Ashwood et al. 2022 style), which has the trial count to recover multi-state
  structure that per-session CV misses. Output:
  ``glm_hmm/<name>__global_pooled_cv/<group>.pkl`` (key ``group_pooled_cv``); the
  most parsimonious state count per group is also reported.

Model variants to fit are declared in the ``CONFIGS`` registry in
``glm_hmm.experiments``; each is a ``GlmHmmConfig`` differing only in its model
specification (features, lags, bias, ...). Run one with ``--config <alias>`` or omit
the flag to run all of them in sequence. ``--cv-mode`` is orthogonal to ``--config``.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import dir_config
from src.glm_hmm.config import GlmHmmConfig
from src.glm_hmm.cv_utils import group_wise_fit_cv, select_best_n_states, session_wise_fit_cv
from src.glm_hmm.data_preparation import get_group_session_ids, process_sessions
from src.glm_hmm.experiments import CONFIGS
from src.glm_hmm.fitting_utils import global_fit


# --------------------------------------------------------------------------- #
# Fitting and persistence
# --------------------------------------------------------------------------- #
def build_session_data(session_ids: list, inputs: list, unnorm_inputs: list, choices: list, masks: list, invalid_idx: list, config: GlmHmmConfig) -> dict:
    """Build the ``{session_id: DataFrame}`` payload of design matrix, choices and masks."""
    session_data = {}
    for idx, session_id in enumerate(session_ids):
        session_inputs = inputs[idx]
        invalid_flag = np.zeros(choices[idx].shape[0], dtype=bool)
        invalid_flag[invalid_idx[idx]] = True

        df = {
            "choices": choices[idx].ravel(),
            "stimulus": unnorm_inputs[idx][:, 0],
            "mask": masks[idx].ravel(),
            "invalid_idx": invalid_flag,
        }
        for i, feat in enumerate(config.model_features):
            df[feat] = session_inputs[:, i]

        session_data[session_id] = pd.DataFrame(df)
    return session_data


def save_models(output_dir: Path, subject_group: str, payload: dict):
    """Pickle the assembled models-and-data payload for one group."""
    with open(output_dir / f"{subject_group}.pkl", "wb") as f:
        pickle.dump(payload, f)


def prepare_group(data: pd.DataFrame, session_ids: list, config: GlmHmmConfig) -> dict:
    """Run the shared prefix once for a subject group: design matrices, global fit, inits.

    The global fit and its per-state initialisations depend only on the config and
    data, not on the CV strategy, so this is computed once and reused across cv modes
    (avoids re-running the expensive global fit when fitting both session and pooled).
    """
    unnorm_inputs, inputs, choices, masks, invalid_idx = process_sessions(data, session_ids, config)

    global_models, fit_lls = global_fit(
        observations=choices,
        inputs=inputs,
        masks=masks,
        state_range=config.state_range,
        n_iters=config.n_iter_global,
        n_initializations=config.n_inits,
    )

    # Pick the best initialisation per state count to seed the CV folds.
    init_params = {"glm_weights": {}, "transition_matrices": {}}
    for n_states in config.state_range:
        best_idx = fit_lls[n_states].index(max(fit_lls[n_states]))
        init_params["glm_weights"][n_states] = global_models[n_states][best_idx].observations.params
        init_params["transition_matrices"][n_states] = global_models[n_states][best_idx].transitions.params

    return {
        "inputs": inputs,
        "choices": choices,
        "masks": masks,
        "n_sessions": len(session_ids),
        "init_params": init_params,
        "global_fits": {"models": global_models, "fits_lls": fit_lls, "init_params": init_params},
        "session_data": build_session_data(session_ids, inputs, unnorm_inputs, choices, masks, invalid_idx, config),
    }


def run_cv_for_group(prep: dict, config: GlmHmmConfig, cv_mode: str, output_dir: Path, subject_group: str):
    """Run one CV strategy on an already-prepared group and persist the result.

    Reuses the shared ``prep`` (design matrices, global fit, inits) from
    :func:`prepare_group`. Returns the ``group_pooled_cv`` dict in pooled mode (for
    downstream state selection), otherwise ``None``.
    """
    init_params = prep["init_params"]

    if cv_mode == "pooled":
        models_cv, train_ll, test_ll, n_test_trials, n_train_trials = group_wise_fit_cv(
            observations=prep["choices"],
            inputs=prep["inputs"],
            masks=prep["masks"],
            init_params=init_params,
            state_range=config.state_range,
            n_iters=config.n_iter_cv,
            k_folds=config.k_folds,
            tolerance=config.tolerance,
            fitting_method=config.fitting_method,
            n_initializations=config.n_inits_cv,
        )
        group_pooled_fits = {
            "models": models_cv,
            "train_ll": train_ll,
            "test_ll": test_ll,
            "n_test_trials": n_test_trials,
            "n_train_trials": n_train_trials,
            "state_range": np.array(config.state_range),
            "n_initializations": config.n_inits_cv,
        }
        save_models(
            output_dir,
            subject_group,
            {
                "global": prep["global_fits"],
                "group_pooled_cv": group_pooled_fits,
                "data": prep["session_data"],
                "config": config.to_serializable(),
            },
        )
        return group_pooled_fits

    # Default: session-wise CV (one model per session).
    models_cv, train_ll, test_ll = session_wise_fit_cv(
        observations=prep["choices"],
        inputs=prep["inputs"],
        masks=prep["masks"],
        n_sessions=prep["n_sessions"],
        init_params=init_params,
        state_range=config.state_range,
        n_iters=config.n_iter_cv,
        k_folds=config.k_folds,
        tolerance=config.tolerance,
        fitting_method=config.fitting_method,
    )
    save_models(
        output_dir,
        subject_group,
        {
            "global": prep["global_fits"],
            "session_wise": {"models": models_cv, "train_ll": train_ll, "test_ll": test_ll},
            "data": prep["session_data"],
            "config": config.to_serializable(),
        },
    )
    return None


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data(processed_dir: Path):
    """Load the trial-level data and metadata, coercing missing labels to -1."""
    data = pd.read_csv(processed_dir / "processed_all_data_accu_60_all.csv")
    for col in ("choice", "target", "outcome"):
        data[col] = data[col].fillna(-1).astype(int)

    metadata = pd.read_csv(processed_dir / "processed_metadata_all_data_accu_60.csv")
    return data, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        nargs="*",
        choices=list(CONFIGS),
        default=None,
        help="Aliases of configs to fit (space-separated); omit to fit every config in the registry.",
    )
    parser.add_argument(
        "--cv-mode",
        nargs="+",
        choices=["session", "pooled"],
        default=["session", "pooled"],
        help="Cross-validation strategies to run (space-separated); omit to run both. 'session' fits one model per session; 'pooled' pools a group's sessions into one model per fold.",
    )
    args = parser.parse_args()

    configs = [CONFIGS[alias] for alias in (args.config or CONFIGS)]

    processed_dir = Path(dir_config.data.processed)
    data, metadata = load_data(processed_dir)
    subject_groups = get_group_session_ids(metadata)

    for config in configs:
        print(f"\n=== Fitting config: {config.name} (cv_modes={args.cv_mode}) ===")
        # One output dir per requested CV mode; the global fit is shared across them.
        output_dirs = {}
        for cv_mode in args.cv_mode:
            dir_suffix = "__global_pooled_cv" if cv_mode == "pooled" else "__session_pooled_cv"
            output_dir = processed_dir / "glm_hmm" / f"{config.name}{dir_suffix}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dirs[cv_mode] = output_dir

        best_states = {}  # per-group chosen state count, pooled mode only
        for group_name, session_ids in subject_groups.items():
            print(f"Preparing {group_name} group (global fit) with sessions: {session_ids}")
            prep = prepare_group(data, session_ids, config)

            for cv_mode in args.cv_mode:
                print(f"  -> {cv_mode} CV for {group_name}")
                result = run_cv_for_group(prep, config, cv_mode, output_dirs[cv_mode], group_name)

                if cv_mode == "pooled":
                    selection = select_best_n_states(result, prep["session_data"], rule=config.selection_rule, tol=config.selection_tol)
                    best_states[group_name] = selection["best"]
                    print(f"     best = {selection['best']} states (peak {selection['best_unconstrained']}, rule={selection['rule']}); bits/trial: {np.round(selection['mean_bits'], 4).tolist()}")

        if "pooled" in args.cv_mode:
            print(f"\nbest_states ({config.selection_rule}) for {config.name} = {best_states}")


if __name__ == "__main__":
    main()
