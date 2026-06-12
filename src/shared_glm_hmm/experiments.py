"""Registry of feature variants to fit on the shared HC+PD basis.

Mirrors the role of ``src.glm_hmm.experiments.CONFIGS`` but is a separate, additive
registry so the original is untouched. Each entry is a
:class:`~src.glm_hmm.config.GlmHmmConfig`; the shared-basis fit script (`fit_shared_basis.py`)
reads ``--config`` from this dict.

Naming scheme ``<curr>__<hist>[_min][_std]``:
  <curr> encodes the current-trial features present, as ordered letters:
      X = stimulus,  Y = color,  Z = color x stimulus interaction (stim_x_color)
    so  X__   = stimulus only (no-color control),
        XY__  = stimulus + color,
        XYZ__ = stimulus + color + interaction.
  <hist> : previous-trial history family --
      ``tgt`` = prev_choice + prev_target + prev_choice_target
      ``coh`` = prev_choice + prev_coherence + prev_choice_coherence
  _min   : reduced history (prev_choice + the choice x {target|coherence} interaction only).
  _std   : history terms are z-scored (per session).
  The ``__`` separates the current-trial part (left) from the previous-trial part (right).
"""

import numpy as np

from src.glm_hmm.config import GlmHmmConfig

CONFIGS: dict[str, GlmHmmConfig] = {
    # ── XY: current = (stimulus, color) ───────────────────────────────────────
    "XY__tgt_std": GlmHmmConfig(
        current_trial_features=("stimulus", "color"),
        prev_trial_features=("prev_choice", "prev_target", "prev_choice_target"),
        standardize_inputs=("prev_choice", "prev_target", "prev_choice_target"),
        n_trials_back=1,
        state_range=np.arange(1, 7),
    ),
    "XY__tgt": GlmHmmConfig(
        current_trial_features=("stimulus", "color"),
        prev_trial_features=("prev_choice", "prev_target", "prev_choice_target"),
        standardize_inputs=(),
        n_trials_back=1,
        state_range=np.arange(1, 7),
    ),
    "XY__tgt_min": GlmHmmConfig(
        current_trial_features=("stimulus", "color"),
        prev_trial_features=("prev_choice", "prev_choice_target"),
        standardize_inputs=(),
        n_trials_back=1,
        state_range=np.arange(1, 7),
    ),
    "XY__coh": GlmHmmConfig(
        current_trial_features=("stimulus", "color"),
        prev_trial_features=("prev_choice", "prev_coherence", "prev_choice_coherence"),
        standardize_inputs=(),
        n_trials_back=1,
        state_range=np.arange(1, 7),
    ),
    "XY__coh_min": GlmHmmConfig(
        current_trial_features=("stimulus", "color"),
        prev_trial_features=("prev_choice", "prev_choice_coherence"),
        standardize_inputs=(),
        n_trials_back=1,
        state_range=np.arange(1, 7),
    ),
    # ── XYZ: current = (stimulus, color, stim_x_color) ────────────────────────
    "XYZ__tgt": GlmHmmConfig(
        current_trial_features=("stimulus", "color", "stim_x_color"),
        prev_trial_features=("prev_choice", "prev_target", "prev_choice_target"),
        standardize_inputs=(),
        n_trials_back=1,
        state_range=np.arange(1, 7),
    ),
    "XYZ__tgt_min": GlmHmmConfig(
        current_trial_features=("stimulus", "color", "stim_x_color"),
        prev_trial_features=("prev_choice", "prev_choice_target"),
        standardize_inputs=(),
        n_trials_back=1,
        state_range=np.arange(1, 7),
    ),
    "XYZ__coh": GlmHmmConfig(
        current_trial_features=("stimulus", "color", "stim_x_color"),
        prev_trial_features=("prev_choice", "prev_coherence", "prev_choice_coherence"),
        standardize_inputs=(),
        n_trials_back=1,
        state_range=np.arange(1, 7),
    ),
    "XYZ__coh_min": GlmHmmConfig(
        current_trial_features=("stimulus", "color", "stim_x_color"),
        prev_trial_features=("prev_choice", "prev_choice_coherence"),
        standardize_inputs=(),
        n_trials_back=1,
        state_range=np.arange(1, 7),
    ),
    # ── X: current = (stimulus,) — no-color control ───────────────────────────
    # CV vs XY__tgt isolates the prior's contribution to held-out fit.
    "X__tgt": GlmHmmConfig(
        current_trial_features=("stimulus",),
        prev_trial_features=("prev_choice", "prev_target", "prev_choice_target"),
        standardize_inputs=(),
        n_trials_back=1,
        state_range=np.arange(1, 7),
    ),
}
