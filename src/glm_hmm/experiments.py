"""Registry of GLM-HMM model variants to fit.

Each entry is a :class:`~glm_hmm.config.GlmHmmConfig` differing only in its model
specification (current/previous-trial features, lags, bias). The alias (dict key)
selects a variant on the ``model_fitting_session_cv.py`` command line; the
config's ``name`` is its output sub-directory (``glm_hmm/<name>/<group>.pkl``).

Importable from both the fitting scripts and the analysis notebooks so the run
list has a single source of truth.

Add an entry to fit a new variant. Every name must contain "masked" (invalid
trials are masked by the pipeline). Only override the fields you are changing;
everything else falls back to the GlmHmmConfig defaults.
"""

from .config import GlmHmmConfig

CONFIGS: dict[str, GlmHmmConfig] = {
    "normalized_stimulus": GlmHmmConfig(
        name="masked_with_bias_1_back_prev_choice_coherence",
        current_trial_features=("normalized_stimulus",),
        prev_trial_features=("prev_choice", "prev_coherence", "prev_choice_coherence"),
        n_trials_back=1,
        add_bias=True,
        observed_dimensions=1,
        n_categories=2,
    ),
    "standardized_stimulus": GlmHmmConfig(
        name="masked_with_bias_1_back_prev_choice_coherence_standardized_stimulus",
        current_trial_features=("standardized_stimulus",),
        prev_trial_features=("prev_choice", "prev_coherence", "prev_choice_coherence"),
        n_trials_back=1,
        add_bias=True,
        observed_dimensions=1,
        n_categories=2,
    ),
    "standardized_stimulus_with_color": GlmHmmConfig(
        name="masked_with_bias_1_back_prev_choice_coherence_standardized_stimulus_with_color",
        current_trial_features=("standardized_stimulus", "color"),
        prev_trial_features=("prev_choice", "prev_coherence", "prev_choice_coherence"),
        n_trials_back=1,
        add_bias=True,
        observed_dimensions=1,
        n_categories=2,
    ),
    "normalized_stimulus_with_color": GlmHmmConfig(
        name="masked_with_bias_1_back_prev_choice_coherence_with_color",
        current_trial_features=("normalized_stimulus", "color"),
        prev_trial_features=("prev_choice", "prev_coherence", "prev_choice_coherence"),
        n_trials_back=1,
        add_bias=True,
        observed_dimensions=1,
        n_categories=2,
    ),
    "2_back": GlmHmmConfig(
        name="masked_with_bias_2_back_prev_choice_coherence",
        n_trials_back=2,
        current_trial_features=("normalized_stimulus",),
        prev_trial_features=("prev_choice", "prev_coherence", "prev_choice_coherence"),
        add_bias=True,
        observed_dimensions=1,
        n_categories=2,
    ),
}
