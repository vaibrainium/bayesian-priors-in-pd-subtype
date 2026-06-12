"""Registry of feature variants to fit on the shared HC+PD basis.

Mirrors the role of ``src.glm_hmm.experiments.CONFIGS`` but is a separate, additive
registry so the original is untouched. Each entry is a
:class:`~src.glm_hmm.config.GlmHmmConfig`; the shared-basis fit script (`fit_shared_basis.py`)
reads ``--config`` from this dict.

Start with ``ashwood_color``: the canonical Ashwood et al. (2022) history terms
(prev_choice, prev_target, prev_choice_target) plus the ``color`` prior cue, which the task
shows acts as a criterion/bias shift. Add variations here as needed (minimal, fuller
history, signed-color, ...).
"""

import numpy as np

from src.glm_hmm.config import GlmHmmConfig

CONFIGS: dict[str, GlmHmmConfig] = {
    # Canonical Ashwood history + the prior (color) cue. color is left in its raw 0/1
    # coding (not standardized) so its weight reads directly as prior-integration gain.
    "ashwood_color": GlmHmmConfig(
        current_trial_features=("stimulus", "color"),
        prev_trial_features=("prev_choice", "prev_target", "prev_choice_target"),
        standardize_inputs=("prev_choice", "prev_target", "prev_choice_target"),
        n_trials_back=1,
        add_bias=True,
        observed_dimensions=1,
        n_categories=2,
        state_range=np.arange(1, 7),
    ),
}
