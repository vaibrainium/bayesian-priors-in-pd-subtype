"""Registry of GLM-HMM model variants to fit.

Each entry is a :class:`~glm_hmm.config.GlmHmmConfig` differing only in its model
specification (current/previous-trial features, lags, bias). The alias (dict key)
both selects a variant on the ``model_fitting.py`` command line and names its output
sub-directory (``glm_hmm/<key>__<cv_mode>/<group>.pkl``).

Importable from both the fitting scripts and the analysis notebooks so the run
list has a single source of truth.

Add an entry to fit a new variant. Configs are always masked (invalid trials are
masked by the pipeline); only override the fields you are changing, everything else
falls back to the GlmHmmConfig defaults.
"""

from .config import GlmHmmConfig

CONFIGS: dict[str, GlmHmmConfig] = {
    "normalized_stimulus": GlmHmmConfig(
        current_trial_features=("normalized_stimulus",),
        prev_trial_features=("prev_choice", "prev_coherence", "prev_choice_coherence"),
        n_trials_back=1,
        add_bias=True,
        observed_dimensions=1,
        n_categories=2,
    ),
    "standardized_stimulus": GlmHmmConfig(
        current_trial_features=("standardized_stimulus",),
        prev_trial_features=("prev_choice", "prev_coherence", "prev_choice_coherence"),
        n_trials_back=1,
        add_bias=True,
        observed_dimensions=1,
        n_categories=2,
    ),
    "standardized_stimulus_with_color": GlmHmmConfig(
        current_trial_features=("standardized_stimulus", "color"),
        prev_trial_features=("prev_choice", "prev_coherence", "prev_choice_coherence"),
        n_trials_back=1,
        add_bias=True,
        observed_dimensions=1,
        n_categories=2,
    ),
    "normalized_stimulus_with_color": GlmHmmConfig(
        current_trial_features=("normalized_stimulus", "color"),
        prev_trial_features=("prev_choice", "prev_coherence", "prev_choice_coherence"),
        n_trials_back=1,
        add_bias=True,
        observed_dimensions=1,
        n_categories=2,
    ),
    "2_back": GlmHmmConfig(
        n_trials_back=2,
        current_trial_features=("normalized_stimulus",),
        prev_trial_features=("prev_choice", "prev_coherence", "prev_choice_coherence"),
        add_bias=True,
        observed_dimensions=1,
        n_categories=2,
    ),
}
