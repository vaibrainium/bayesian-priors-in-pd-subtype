"""Fit GLM-HMM models per subject group (HC, PD subtype x medication state).

For each group this script builds session-wise design matrices, runs a global
fit to obtain initialisations, then runs k-fold cross-validated session-wise
fits, and pickles the models together with the data and config.
"""

import copy
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.random as npr
import pandas as pd
from sklearn import preprocessing

from config import dir_config
from src.glm_hmm.cv_utils import session_wise_fit_cv
from src.glm_hmm.fitting_utils import global_fit


@dataclass(frozen=True)
class GlmHmmConfig:
    """Model specification and fitting hyperparameters for one GLM-HMM run."""

    # --- model specification (serialised into the output pickle) ---
    name: str
    current_trial_features: tuple = ("normalized_stimulus",)
    prev_trial_features: tuple = ("prev_choice", "prev_coherence", "prev_choice_coherence")
    n_trials_back: int = 1
    add_bias: bool = True
    observed_dimensions: int = 1
    n_categories: int = 2

    # --- fitting hyperparameters ---
    state_range: np.ndarray = field(default_factory=lambda: np.arange(1, 6))
    n_inits: int = 20
    n_iter_global: int = 7000
    n_iter_cv: int = 2500
    k_folds: int = 5
    tolerance: float = 1e-4
    fitting_method: str = "em"

    @property
    def is_masked(self) -> bool:
        """Whether invalid trials are masked (required by the current pipeline)."""
        return "masked" in self.name

    @property
    def input_features(self) -> list:
        """Current-trial regressors, with bias appended when requested."""
        feats = list(self.current_trial_features)
        return feats + ["bias"] if self.add_bias else feats

    @property
    def model_features(self) -> list:
        """Full ordered list of design-matrix columns (current + lagged)."""
        prev = [f"{var}_{n + 1}" for n in range(self.n_trials_back) for var in self.prev_trial_features]
        return self.input_features + prev

    @property
    def input_dim(self) -> int:
        return len(self.model_features)

    def to_serializable(self) -> dict:
        """Plain-dict view stored in the output pickle (kept stable for downstream notebooks)."""
        return {
            "name": self.name,
            "observed_dimensions": self.observed_dimensions,
            "n_categories": self.n_categories,
            "add_bias": self.add_bias,
            "current_trial_features": list(self.current_trial_features),
            "prev_trial_features": list(self.prev_trial_features),
            "n_trials_back": self.n_trials_back,
            "model_features": self.model_features,
        }


# --------------------------------------------------------------------------- #
# Session grouping
# --------------------------------------------------------------------------- #
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

    signed_coherence = session_data.signed_coherence.values / 100  # standardize coherence to [-1, 1]
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

    # Current-trial features
    for idx, feat in enumerate(config.input_features):
        if feat == "normalized_stimulus":
            X[0, :, idx] = data.signed_coherence.values[first_trial:] / 100
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


def process_sessions(data: pd.DataFrame, session_ids: list, config: GlmHmmConfig, seed: int = 42):
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
            if feat != "bias":
                inputs_session_wise[idx_session][row_mask, feat_idx] = preprocessing.scale(inputs_session_wise[idx_session][row_mask, feat_idx], axis=0)

    return unnormalized_inputs_session_wise, inputs_session_wise, choices_session_wise, masks_session_wise, invalid_idx_session_wise


# --------------------------------------------------------------------------- #
# Fitting and persistence
# --------------------------------------------------------------------------- #
def save_data_and_models(
    output_dir: Path,
    subject_group: str,
    session_ids: list,
    global_fits: dict,
    session_wise_fits: dict,
    inputs: list,
    unnorm_inputs: list,
    choices: list,
    masks: list,
    invalid_idx: list,
    config: GlmHmmConfig,
):
    """Pickle the fitted models together with the per-session data and config."""
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

    models_and_data = {
        "global": global_fits,
        "session_wise": session_wise_fits,
        "data": session_data,
        "config": config.to_serializable(),
    }

    with open(output_dir / f"{subject_group}.pkl", "wb") as f:
        pickle.dump(models_and_data, f)


def fit_glm_hmm_for_subject_group(data: pd.DataFrame, subject_group: str, session_ids: list, config: GlmHmmConfig, output_dir: Path):
    """Run the global + cross-validated session-wise fits for one subject group."""
    unnorm_inputs, inputs, choices, masks, invalid_idx = process_sessions(data, session_ids, config)

    global_models, fit_lls = global_fit(
        observations=choices,
        inputs=inputs,
        masks=masks,
        state_range=config.state_range,
        n_iters=config.n_iter_global,
        n_initializations=config.n_inits,
    )

    # Pick the best initialisation per state count to seed the session-wise fits.
    init_params = {"glm_weights": {}, "transition_matrices": {}}
    for n_states in config.state_range:
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
        state_range=config.state_range,
        n_iters=config.n_iter_cv,
        k_folds=config.k_folds,
        tolerance=config.tolerance,
        fitting_method=config.fitting_method,
    )
    session_wise_fits = {"models": models_cv, "train_ll": train_ll, "test_ll": test_ll}

    save_data_and_models(
        output_dir=output_dir,
        subject_group=subject_group,
        session_ids=session_ids,
        global_fits=global_fits,
        session_wise_fits=session_wise_fits,
        inputs=inputs,
        unnorm_inputs=unnorm_inputs,
        choices=choices,
        masks=masks,
        invalid_idx=invalid_idx,
        config=config,
    )


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
    config = GlmHmmConfig(name="masked_with_bias_1_back_prev_choice_coherence")

    processed_dir = Path(dir_config.data.processed)
    output_dir = processed_dir / "glm_hmm" / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    data, metadata = load_data(processed_dir)

    subject_groups = get_group_session_ids(metadata)
    for group_name, session_ids in subject_groups.items():
        print(f"Fitting GLM-HMM for {group_name} group with sessions: {session_ids}")
        fit_glm_hmm_for_subject_group(data, group_name, session_ids, config, output_dir)


if __name__ == "__main__":
    main()
