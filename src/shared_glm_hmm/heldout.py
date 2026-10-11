"""Cross-validation of the shared GLM-HMM on held-out participants or whole sessions.

The pooled CV in :func:`src.glm_hmm.cv_utils.group_wise_fit_cv` holds out interleaved trial blocks
*within* every session, so each test block is scored by a model that was trained on the same
session (and the same participant). That can overstate generalisation. Here a fold holds out whole
sessions instead:

* ``"participant"`` -- every session of a held-out participant (a patient's OFF and ON sessions
  together) is in the test fold, so no part of that person's behaviour is seen in training.
* ``"session"`` -- whole sessions are held out without regard to participant, so a patient's other
  session can still be in training.

Folds are stratified by group (HC / tremor / bradykinetic / other PD) so every fold has a similar mix.
Nothing in the initialisation sees the test fold: each fold gets its own one-state GLM fit on the
training sessions, and every K-state restart is a perturbation of that, mirroring
:func:`src.glm_hmm.fitting_utils.global_fit`. Held-out sessions are scored with the trained model's
parameters fixed (initial state distribution, transitions and GLM weights), one session at a time.
"""

import numpy as np
import numpy.random as npr
import ssm

GROUPS = ("HC", "tremor", "bradykinetic")


def session_group(sess_md):
    """HC / tremor / bradykinetic / other_PD label per session (``session_metadata`` table)."""
    subtype = sess_md["subtype"].astype("string")
    return subtype.where(subtype.isin(GROUPS), "other_PD").fillna("other_PD")


def assign_folds(sess_md, scheme, k_folds=5, seed=0):
    """``{session_id: fold}`` with whole participants (or whole sessions) kept together.

    Units (participants or sessions) are shuffled within each group and dealt round-robin into
    folds, continuing the deal across groups so fold sizes stay balanced.
    """
    if scheme not in ("participant", "session"):
        raise ValueError(f"unknown scheme {scheme!r}")
    md = sess_md.assign(group=session_group(sess_md))
    unit = "subject_id" if scheme == "participant" else "session_id"
    units = md.drop_duplicates(unit)[[unit, "group"]]

    rng = np.random.default_rng(seed)
    unit_fold, position = {}, 0
    for group in sorted(units["group"].unique()):
        members = units.loc[units["group"] == group, unit].to_numpy()
        for member in rng.permutation(members):
            unit_fold[member] = position % k_folds
            position += 1
    return dict(zip(md["session_id"], md[unit].map(unit_fold)))


def new_hmm(n_states, input_dim, prior_sigma):
    return ssm.HMM(n_states, 1, input_dim, observations="input_driven_obs", observation_kwargs=dict(C=2, prior_sigma=prior_sigma), transitions="standard")


def fit_glm(observations, inputs, masks, prior_sigma=2.0, n_iters=7000, tolerance=1e-4):
    """One-state GLM on the training sessions; the base every restart is perturbed from."""
    glm = new_hmm(1, inputs[0].shape[1], prior_sigma)
    glm.fit(observations, inputs=inputs, masks=masks, method="em", num_iters=n_iters, tolerance=tolerance, verbose=0)
    return glm


def fit_restart(glm_weights, observations, inputs, masks, n_states, init_num, prior_sigma=2.0, n_iters=5000, tolerance=1e-4):
    """One K-state restart from a perturbation of the fold's GLM weights (``global_fit`` scheme)."""
    input_dim = inputs[0].shape[1]
    hmm = new_hmm(n_states, input_dim, prior_sigma)
    if n_states == 1:
        hmm.observations.params = np.copy(glm_weights)
    else:
        npr.seed(init_num * n_states)
        hmm.observations.params = glm_weights + npr.normal(0, 0.2, (n_states, 1, input_dim))
        transition = 0.9 * np.eye(n_states) + npr.multivariate_normal(np.zeros(n_states), 0.05 * np.eye(n_states), size=n_states)
        transition /= transition.sum(axis=1, keepdims=True)
        # passed as-is (not logged), exactly as global_fit does, so the starting points match the
        # pipeline behind Figure 4; EM re-estimates the transitions either way
        hmm.transitions.params = [transition]
    hmm.fit(observations, inputs=inputs, masks=masks, method="em", num_iters=n_iters, initialize=False, tolerance=tolerance, verbose=0)
    return hmm


def session_log_likelihoods(hmm, observations, inputs, masks):
    """Log-likelihood (nats) of each session separately, parameters fixed."""
    return np.array([hmm.log_likelihood(o, inputs=i, masks=m) for o, i, m in zip(observations, inputs, masks)])
