import copy

import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn import preprocessing

from .config import GlmHmmConfig


def get_group_session_ids(metadata: pd.DataFrame) -> dict:
    """Map each subject group to its sorted list of session IDs."""

    def session_ids_for(meta, subjects, treatment_label):
        """Sorted session IDs for the given subjects in a given treatment state."""
        selected = meta[(meta["subject_id"].isin(subjects)) & (meta["treatment"].str.upper() == treatment_label.upper())]
        return sorted(selected["session_id"].dropna().unique().tolist())

    # HC sessions (is_pd == 0, single session per subject)
    hc_meta = metadata[metadata["is_pd"] == 0]
    hc_session_ids = sorted(hc_meta["session_id"].dropna().unique().tolist())

    # PD subtype x medication groups
    pd_meta = metadata[metadata["is_pd"] == 1]
    tremor_subjects = sorted(pd_meta[pd_meta["trem_vs_brady_type"] == "tremor"]["subject_id"].unique().tolist())
    brady_subjects = sorted(pd_meta[pd_meta["trem_vs_brady_type"] == "bradykinetic"]["subject_id"].unique().tolist())

    return {
        "asmHC": hc_session_ids,
        "Tremor_OFF": session_ids_for(pd_meta, tremor_subjects, "OFF"),
        "Tremor_ON": session_ids_for(pd_meta, tremor_subjects, "ON"),
        "Brady_OFF": session_ids_for(pd_meta, brady_subjects, "OFF"),
        "Brady_ON": session_ids_for(pd_meta, brady_subjects, "ON"),
    }


# --------------------------------------------------------------------------- #
# Design-matrix construction
# --------------------------------------------------------------------------- #
def extract_previous_trial_data(session_data: pd.DataFrame, valid_idx: np.ndarray, first_trial: int, config: GlmHmmConfig) -> dict:
    """Build the lagged (previous-trial) features for one session.

    Returns a dict mapping each previous-trial feature to an
    (n_trials, n_trials_back) array.
    """
    npr.seed(1)
    n_trials = session_data.shape[0] - first_trial

    signed_coherence = session_data.signed_coherence.values
    choice = session_data.choice.values * 2 - 1  # convert to -1/1
    target = session_data.target.values * 2 - 1  # convert to -1/1

    prev_data = {var: np.empty((n_trials, config.n_trials_back), dtype=float) for var in config.prev_trial_features}

    for i in range(first_trial, session_data.shape[0]):
        valid_before = valid_idx[valid_idx < i][-config.n_trials_back :]
        padded = np.pad(valid_before, (config.n_trials_back - len(valid_before), 0), "constant", constant_values=0)
        for var in config.prev_trial_features:
            var_col = var[5:] if var.startswith("prev_") else var

            if var_col == "coherence":
                vals = signed_coherence
            elif var_col == "target":
                vals = target
            elif var_col == "choice":
                vals = choice
            elif var_col == "choice_outcome":
                vals = choice * session_data.outcome.values
            elif var_col == "choice_target":
                vals = choice * target
            elif var_col == "choice_coherence":
                vals = choice * signed_coherence
            elif var_col == "coherence_choice_outcome":
                vals = choice * session_data.outcome.values * signed_coherence
            else:
                vals = session_data[var_col].values

            prev_data[var][i - first_trial] = vals[padded]
    return prev_data


def prepare_input_data(data: pd.DataFrame, valid_idx: np.ndarray, first_trial: int, config: GlmHmmConfig) -> list:
    """Assemble the (1, n_trials, input_dim) design matrix for one session."""
    n_trials = data.shape[0] - first_trial
    X = np.zeros((1, n_trials, config.input_dim))

    # standardize coherence to [-1, 1] if not already done
    if data.signed_coherence.max() > 1 or data.signed_coherence.min() < -1:
        data.signed_coherence = data.signed_coherence / 100

    # Current-trial features
    for idx, feat in enumerate(config.input_features):
        if feat in ["normalized_stimulus", "standardized_stimulus", "stimulus"]:
            X[0, :, idx] = data.signed_coherence.values[first_trial:]
        elif feat == "color":
            X[0, :, idx] = data.color.values[first_trial:]
        elif feat == "bias":
            X[0, :, idx] = 1
        else:
            X[0, :, idx] = data[feat].values[first_trial:]

    # Previous-trial (lagged) features
    prev_data = extract_previous_trial_data(data, valid_idx, first_trial, config)
    col_idx = len(config.input_features)
    for var in config.prev_trial_features:
        for n in range(config.n_trials_back):
            X[0, :, col_idx] = prev_data[var][:, n]
            col_idx += 1
    return list(X)


def process_sessions(data: pd.DataFrame, session_ids: list, config: GlmHmmConfig, seed: int = 42) -> tuple:
    """Build inputs, choices, masks and invalid-trial indices for every session.

    Returns (unnormalized_inputs, inputs, choices, masks, invalid_idx), each a
    list with one entry per session.
    """
    if not config.is_masked:
        raise NotImplementedError("Only models with 'masked' in their name are supported; invalid trials must be masked.")

    inputs_session_wise = []
    choices_session_wise = []
    invalid_idx_session_wise = []
    masks_session_wise = []

    npr.seed(seed)
    for session_id in session_ids:
        trial_data = data[data["session_id"] == session_id].reset_index(drop=True)

        valid_idx = np.where(trial_data.outcome != -1)[0]
        if len(valid_idx) < config.n_trials_back:
            raise ValueError(f"Session {session_id} has fewer valid trials ({len(valid_idx)}) than n_trials_back ({config.n_trials_back}). Consider reducing n_trials_back or excluding this session.")

        first_trial = valid_idx[config.n_trials_back - 1] + 1

        inputs = prepare_input_data(trial_data, valid_idx, first_trial, config)
        choices = trial_data.choice.values.reshape(-1, 1).astype("int")[first_trial:]

        # Mask invalid trials and replace them with a random label so they don't
        # break fitting (they are excluded from the likelihood via the mask).
        invalid_idx = np.where(choices == -1)[0]
        choices[invalid_idx, 0] = np.random.choice(2, invalid_idx.shape[0])
        mask = np.ones_like(choices, dtype=bool)
        mask[invalid_idx] = 0

        inputs_session_wise += inputs
        choices_session_wise.append(choices)
        masks_session_wise.append(mask)
        invalid_idx_session_wise.append(invalid_idx)

    # Per-session z-scoring of all features except the bias term.
    unnormalized_inputs_session_wise = copy.deepcopy(inputs_session_wise)

    for idx_session in range(len(session_ids)):
        row_mask = masks_session_wise[idx_session][:, 0]
        for feat_idx, feat in enumerate(config.model_features):
            if feat in config.standardize_features:
                inputs_session_wise[idx_session][row_mask, feat_idx] = preprocessing.scale(inputs_session_wise[idx_session][row_mask, feat_idx], axis=0)

    return unnormalized_inputs_session_wise, inputs_session_wise, choices_session_wise, masks_session_wise, invalid_idx_session_wise
