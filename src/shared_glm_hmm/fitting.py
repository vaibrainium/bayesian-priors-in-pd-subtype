"""Thin orchestration for the shared-basis fit and finetune.

The heavy lifting is reused from ``src.glm_hmm`` (``process_sessions``, ``global_fit``,
``group_wise_fit_cv``, ``group_wise_fit``, ``session_wise_fit``, ``select_best_n_states``).
The functions here only assemble those primitives over the single pooled group, mirroring
the orchestration in ``scripts/glm_hmm/model_fitting.py`` / ``model_finetuning.py`` (which
are plain scripts, not importable packages) without modifying them. Output pickles share
the exact schema those scripts produce, so the existing analysis notebooks read them
unchanged.
"""

import numpy as np
import pandas as pd

from src.glm_hmm.config import GlmHmmConfig
from src.glm_hmm.cv_utils import group_wise_fit_cv
from src.glm_hmm.data_preparation import process_sessions
from src.glm_hmm.fitting_utils import global_fit, group_wise_fit, session_wise_fit


# --------------------------------------------------------------------------- #
# Fitting (mirrors model_fitting.py: prepare_group + pooled branch)
# --------------------------------------------------------------------------- #
def build_session_data(session_ids, inputs, unnorm_inputs, choices, masks, invalid_idx, config):
    """Per-session ``{session_id: DataFrame}`` of design matrix, choices and masks."""
    session_data = {}
    for idx, session_id in enumerate(session_ids):
        invalid_flag = np.zeros(choices[idx].shape[0], dtype=bool)
        invalid_flag[invalid_idx[idx]] = True
        df = {
            "choices": choices[idx].ravel(),
            "stimulus": unnorm_inputs[idx][:, 0],
            "mask": masks[idx].ravel(),
            "invalid_idx": invalid_flag,
        }
        for i, feat in enumerate(config.model_features):
            df[feat] = inputs[idx][:, i]
        session_data[session_id] = pd.DataFrame(df)
    return session_data


def prepare_group(data, session_ids, config):
    """Design matrices + global fit + best per-state initialisations (one shared group)."""
    unnorm_inputs, inputs, choices, masks, invalid_idx = process_sessions(data, session_ids, config)

    global_models, fit_lls = global_fit(
        observations=choices,
        inputs=inputs,
        masks=masks,
        state_range=config.state_range,
        n_iters=config.n_iter_global,
        n_initializations=config.n_inits,
        prior_sigma=config.prior_sigma,
    )

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


def run_pooled_cv(prep, config, name):
    """Pooled (group-level) CV over the shared group; returns the saveable payload dict."""
    models_cv, train_ll, test_ll, n_test_trials, n_train_trials = group_wise_fit_cv(
        observations=prep["choices"],
        inputs=prep["inputs"],
        masks=prep["masks"],
        init_params=prep["init_params"],
        state_range=config.state_range,
        n_iters=config.n_iter_cv,
        k_folds=config.k_folds,
        tolerance=config.tolerance,
        fitting_method=config.fitting_method,
        n_initializations=config.n_inits_cv,
        prior_sigma=config.prior_sigma,
    )
    group_pooled_cv = {
        "models": models_cv,
        "train_ll": train_ll,
        "test_ll": test_ll,
        "n_test_trials": n_test_trials,
        "n_train_trials": n_train_trials,
        "state_range": np.array(config.state_range),
        "n_initializations": config.n_inits_cv,
    }
    return {
        "global": prep["global_fits"],
        "group_pooled_cv": group_pooled_cv,
        "data": prep["session_data"],
        "config": config.to_serializable(name),
    }


# --------------------------------------------------------------------------- #
# Finetuning at a chosen K (mirrors model_finetuning.finetune_pooled)
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


def finetune_pooled(bundle, best_k, n_iters=2500):
    """Per-session models + one pooled model at ``best_k``, warm-started from the best fold."""
    prior_sigma = bundle["config"].get("prior_sigma", 2.0)
    gpc = bundle["group_pooled_cv"]
    state_idx = int(np.where(np.asarray(gpc["state_range"]) == best_k)[0][0])
    best_fold = int(np.nanargmax(gpc["test_ll"][state_idx]))
    init = gpc["models"][best_k][best_fold]
    init_flat = {
        "glm_weights": init.observations.params,
        "transition_matrices": init.transitions.params,
    }

    sessions = list(bundle["data"].keys())
    observations, inputs, masks = session_arrays(bundle)

    init_params = {
        "glm_weights": {idx: init.observations.params for idx in range(len(sessions))},
        "transition_matrices": {idx: init.transitions.params for idx in range(len(sessions))},
    }
    models_s, fit_lls_s = session_wise_fit(
        observations, inputs, masks,
        n_sessions=len(sessions),
        init_params=init_params,
        n_states=best_k,
        n_iters=n_iters,
        prior_sigma=prior_sigma,
    )
    result = {
        "models": {s: models_s[i] for i, s in enumerate(sessions)},
        "fit_lls": {s: fit_lls_s[i] for i, s in enumerate(sessions)},
        "best_folds": {s: best_fold for s in sessions},
    }

    pooled_model, pooled_fit_ll = group_wise_fit(
        observations, inputs, masks, init_flat,
        n_states=best_k, n_iters=n_iters, prior_sigma=prior_sigma,
    )
    result["pooled"] = pooled_model
    result["pooled_fit_ll"] = pooled_fit_ll
    return result
