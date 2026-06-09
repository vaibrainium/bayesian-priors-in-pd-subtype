"""Configuration object shared by the GLM-HMM fitting scripts and notebooks."""

from dataclasses import dataclass, field

import numpy as np


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
    n_iter_cv: int = 5000
    k_folds: int = 5
    tolerance: float = 1e-4
    fitting_method: str = "em"

    # --- pooled (group-level) CV only; ignored in session-wise mode ---
    n_inits_cv: int = 5  # random restarts per (state, fold) in pooled CV; best train-LL kept
    selection_rule: str = "1sem"  # "1sem" (within one fold-SEM of peak) or "tol"
    selection_tol: float = 0.005  # bits/trial, used only when selection_rule == "tol"

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
            "state_range": self.state_range.tolist(),
        }
