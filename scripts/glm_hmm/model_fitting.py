import copy
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn import preprocessing

from config import dir_config
from src.glm_hmm.cv_utils import session_wise_fit_cv
from src.glm_hmm.fitting_utils import global_fit


def get_group_session_ids(metadata):

    def get_session_ids(meta, subjects, treatment_label):
        """Return sorted session IDs for subjects in a given treatment state."""
        return sorted(meta[(meta["subject_id"].isin(subjects)) & (meta["treatment"].str.upper() == treatment_label.upper())]["session_id"].dropna().unique().tolist())

    # --- HC sessions (is_pd == 0, single session per subject) ---
    hc_meta = metadata[metadata["is_pd"] == 0]
    hc_session_ids = sorted(hc_meta["session_id"].dropna().unique().tolist())

    # --- PD subtype × medication groups ---
    pd_meta = metadata[metadata["is_pd"] == 1]
    tremor_subjects = sorted(pd_meta[pd_meta["trem_vs_brady_type"] == "tremor"]["subject_id"].unique().tolist())
    brady_subjects = sorted(pd_meta[pd_meta["trem_vs_brady_type"] == "bradykinetic"]["subject_id"].unique().tolist())
    intermed_subjects = sorted(pd_meta[pd_meta["trem_vs_brady_type"] == "intermediate"]["subject_id"].unique().tolist())

    tremor_off_ids = get_session_ids(pd_meta, tremor_subjects, "OFF")
    tremor_on_ids = get_session_ids(pd_meta, tremor_subjects, "ON")
    brady_off_ids = get_session_ids(pd_meta, brady_subjects, "OFF")
    brady_on_ids = get_session_ids(pd_meta, brady_subjects, "ON")

    return {
        "HC": hc_session_ids,
        "Tremor_OFF": tremor_off_ids,
        "Tremor_ON": tremor_on_ids,
        "Brady_OFF": brady_off_ids,
        "Brady_ON": brady_on_ids,
    }


def extract_previous_trial_data(session_data, valid_idx, first_trial):
    npr.seed(1)
    n_trials = session_data.shape[0] - first_trial
    prev_data = {}
    signed_coherence = session_data.signed_coherence.values / 100  # standardize coherence to be between -1 and 1
    choice = session_data.choice.values * 2 - 1  # Convert to -1/1
    target = session_data.target.values * 2 - 1  # Convert to -1/1

    # For each previous trial feature, create an array for each trial back
    for var in PREV_TRIAL_FEATURES:
        prev_data[var] = np.empty((n_trials, N_TRIALS_BACK), dtype=float)

    # Loop through each trial starting from first_trial
    for i in range(first_trial, session_data.shape[0]):
        valid_before = valid_idx[valid_idx < i][-N_TRIALS_BACK:]
        padded = np.pad(valid_before, (N_TRIALS_BACK - len(valid_before), 0), "constant", constant_values=0)
        for var in PREV_TRIAL_FEATURES:
            var_col = var[5:] if var.startswith("prev_") else var

            if var_col == "coherence":
                vals = signed_coherence
            elif var_col == "target":
                vals = target
            elif var_col == "choice":
                vals = choice
            elif var_col == "choice_outcome":
                vals = choice * session_data.outcome.values
            elif var_col == "choice_coherence":
                vals = choice * signed_coherence
            elif var_col == "coherence_choice_outcome":
                vals = choice * session_data.outcome.values * signed_coherence
            else:
                vals = session_data[var_col].values

            prev_data[var][i - first_trial] = vals[padded]
    return prev_data


def prepare_input_data(data, valid_idx, first_trial):
    n_trials = data.shape[0] - first_trial
    X = np.zeros((1, n_trials, INPUT_DIM))
    # Fill current trial features
    for idx, feat in enumerate(CURRENT_TRIAL_FEATURES):
        if feat == "normalized_stimulus":
            X[0, :, idx] = data.signed_coherence.values[first_trial:] / 100
        elif feat == "bias":
            X[0, :, idx] = 1
        else:
            X[0, :, idx] = data[feat].values[first_trial:]

    # Fill previous trial features
    prev_data = extract_previous_trial_data(data, valid_idx, first_trial)
    col_idx = len(CURRENT_TRIAL_FEATURES)
    for var in PREV_TRIAL_FEATURES:
        for n in range(N_TRIALS_BACK):
            X[0, :, col_idx] = prev_data[var][:, n]
            col_idx += 1
    return list(X)


def process_sessions(data, session_ids, seed=42):

    inputs_session_wise = []
    choices_session_wise = []
    invalid_idx_session_wise = []
    masks_session_wise = []

    npr.seed(seed)
    for session_id in session_ids:
        trial_data = data[data["session_id"] == session_id].reset_index(drop=True)

        valid_idx = np.where(trial_data.outcome != -1)[0]
        if len(valid_idx) < N_TRIALS_BACK:
            raise ValueError(f"Session {session_id} has fewer valid trials ({len(valid_idx)}) than n_trials_back ({N_TRIALS_BACK}). Consider reducing n_trials_back or excluding this session.")

        first_trial = valid_idx[N_TRIALS_BACK - 1] + 1

        # Prepare inputs and choices for the session
        inputs = prepare_input_data(trial_data, valid_idx, first_trial)
        choices = trial_data.choice.values.reshape(-1, 1).astype("int")[first_trial:]

        # Adjust invalid_idx and prepare mask
        invalid_idx = np.where(choices == -1)[0]

        if "masked" in MODEL_NAME:
            # For training, replace -1 with a random sample from 0,1
            choices[invalid_idx, 0] = np.random.choice(2, invalid_idx.shape[0])

            mask = np.ones_like(choices, dtype=bool)
            mask[invalid_idx] = 0
        else:
            assert "masked" in MODEL_NAME, "Invalid trials should only be masked in models with 'masked' in their name."

        masks_session_wise.append(mask)
        inputs_session_wise += inputs
        choices_session_wise.append(choices)
        invalid_idx_session_wise.append(invalid_idx)

    unnormalized_inputs_session_wise = copy.deepcopy(inputs_session_wise)
    for idx_session in range(len(session_ids)):
        row_mask = masks_session_wise[idx_session][:, 0]
        for feat_idx, feat in enumerate(MODEL_FEATURES):
            if feat != "bias":
                inputs_session_wise[idx_session][row_mask, feat_idx] = preprocessing.scale(inputs_session_wise[idx_session][row_mask, feat_idx], axis=0)

    return unnormalized_inputs_session_wise, inputs_session_wise, choices_session_wise, masks_session_wise, invalid_idx_session_wise


def save_data_and_models(
    subject_group: str,
    session_ids: list,
    global_fits: dict,
    session_wise_fits: dict,
    inputs: list,
    unnorm_inputs: list,
    choices: list,
    masks: list,
    invalid_idx: list,
):

    # store data and models for session-wise
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

        for i, feat in enumerate(MODEL_FEATURES):
            df[feat] = session_inputs[:, i]

        session_data[session_id] = pd.DataFrame(df)

    models_and_data = {
        "global": global_fits,
        "session_wise": session_wise_fits,
        "data": session_data,
        "config": GLM_HMM_CONFIG,
    }

    with open(Path(glm_hmm_dir, MODEL_NAME, f"{subject_group}.pkl"), "wb") as f:
        pickle.dump(models_and_data, f)


def fit_glm_hmm_for_subject_group(subject_group, session_ids):
    unnorm_inputs, inputs, choices, masks, invalid_idx = process_sessions(data, session_ids)
    global_models, fit_lls = global_fit(observations=choices, inputs=inputs, masks=masks, state_range=STATE_RANGE, n_iters=N_ITER_GLOBAL, n_initializations=N_INITS)

    # get best model out of N_INITS initializations for each state
    init_params = {"glm_weights": {}, "transition_matrices": {}}
    for n_states in np.arange(1, 6):
        best_idx = fit_lls[n_states].index(max(fit_lls[n_states]))
        init_params["glm_weights"][n_states] = global_models[n_states][best_idx].observations.params
        init_params["transition_matrices"][n_states] = global_models[n_states][best_idx].transitions.params

    global_fits = {"models": global_models, "fits_lls": fit_lls, "init_params": init_params}

    models_cv, train_ll, test_ll = session_wise_fit_cv(
        observations=choices,
        inputs=inputs,
        masks=masks,
        n_sessions=len(session_ids),
        init_params=init_params,
        state_range=STATE_RANGE,
        n_iters=N_ITER_CV,
        k_folds=K_FOLDS,
        tolerance=TOLERANCE,
        fitting_method=FITTING_METHOD,
    )

    session_wise_fits = {"models": models_cv, "train_ll": train_ll, "test_ll": test_ll}
    save_data_and_models(
        subject_group=subject_group,
        session_ids=session_ids,
        global_fits=global_fits,
        session_wise_fits=session_wise_fits,
        inputs=inputs,
        unnorm_inputs=unnorm_inputs,
        choices=choices,
        masks=masks,
        invalid_idx=invalid_idx,
    )


if __name__ == "__main__":
    STATE_RANGE = np.arange(1, 6)
    N_INITS = 20
    N_ITER_GLOBAL = 7000
    N_ITER_CV = 2500
    K_FOLDS = 5
    TOLERANCE = 1e-4
    FITTING_METHOD = "em"

    GLM_HMM_CONFIG = {
        "name": "masked_no_bias_1_back_prev_choice_coherence",
        "observed_dimensions": 1,
        "n_categories": 2,
        "add_bias": True,
        "current_trial_features": ["normalized_stimulus"],
        "prev_trial_features": ["prev_choice", "prev_coherence", "prev_choice_coherence"],
        "n_trials_back": 1,
    }

    # Set up project root and import project-specific modules
    PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    # Define directories
    processed_dir = Path(dir_config.data.processed)
    glm_hmm_dir = processed_dir / "glm_hmm"
    glm_hmm_dir.mkdir(parents=True, exist_ok=True)
    model_dir = glm_hmm_dir / GLM_HMM_CONFIG["name"]
    model_dir.mkdir(parents=True, exist_ok=True)

    CURRENT_TRIAL_FEATURES = GLM_HMM_CONFIG["current_trial_features"] + ["bias"] if GLM_HMM_CONFIG["add_bias"] else GLM_HMM_CONFIG["current_trial_features"]
    PREV_TRIAL_FEATURES = GLM_HMM_CONFIG["prev_trial_features"]
    N_TRIALS_BACK = GLM_HMM_CONFIG["n_trials_back"]

    MODEL_FEATURES = CURRENT_TRIAL_FEATURES + [f"{var}_{n + 1}" for n in range(N_TRIALS_BACK) for var in PREV_TRIAL_FEATURES]
    GLM_HMM_CONFIG["model_features"] = MODEL_FEATURES

    INPUT_DIM = len(MODEL_FEATURES)
    MODEL_NAME = GLM_HMM_CONFIG["name"]

    data = pd.read_csv(Path(processed_dir, "processed_all_data_accu_60_all.csv"))
    data.choice = data.choice.fillna(-1).astype(int)
    data.target = data.target.fillna(-1).astype(int)
    data.outcome = data.outcome.fillna(-1).astype(int)

    metadata = pd.read_csv(Path(processed_dir, "processed_metadata_all_data_accu_60.csv"))

    SUBJECT_GROUPS = get_group_session_ids(metadata)
    for group_name, session_ids in SUBJECT_GROUPS.items():
        print(f"Fitting GLM-HMM for {group_name} group with sessions: {session_ids}")
        fit_glm_hmm_for_subject_group(group_name, session_ids)
