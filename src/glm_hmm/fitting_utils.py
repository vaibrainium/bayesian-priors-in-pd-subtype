"""GLM-HMM global and session-wise fitting routines.

``global_fit`` dispatches every ``(n_states, init_num)`` fit in a single joblib
pool so all cores stay saturated and the worker pool is created only once;
``session_wise_fit`` runs one fit per session in a single pool. Both pin inner
BLAS/OpenMP threads to 1 so the workers don't oversubscribe the cores, and the
EM (with ``initialize=False`` for the session-wise fit) is fully deterministic.
"""

import numpy as np
import numpy.random as npr
import ssm
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from tqdm import tqdm


def global_fit(observations, inputs, masks, state_range=np.arange(2, 6), n_initializations=20, fitting_method="em", n_iters=200, tolerance=10**-4, n_jobs=-1):
    """
    Global GLM-HMM fitting with all (state, initialization) fits dispatched in one pool.
    """
    print("Fitting GLM globally...")
    obs_dim = observations[0].shape[1]
    input_dim = inputs[0].shape[1]
    C = len(np.unique(observations[0]))
    glm = ssm.HMM(1, obs_dim, input_dim, observations="input_driven_obs", observation_kwargs=dict(C=C), transitions="standard")

    glm.fit(observations, inputs=inputs, masks=masks, method=fitting_method, num_iters=n_iters, tolerance=tolerance)
    glm_weights = glm.observations.params

    def fit_single_initialization(n_states, init_num):
        """
        Fit GLM-HMM with a single initialization.
        """
        npr.seed(init_num * n_states)  # Set seed for reproducibility
        glm_hmm = ssm.HMM(n_states, obs_dim, input_dim, observations="input_driven_obs", observation_kwargs=dict(C=C), transitions="standard")

        # Initialize weights and transition matrix
        glm_hmm.observations.params = glm_weights + np.random.normal(0, 0.2, (n_states, 1, input_dim))
        transition_matrix = 0.9 * np.eye(n_states) + np.random.multivariate_normal(mean=np.zeros(n_states), cov=0.05 * np.eye(n_states), size=n_states)
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
        glm_hmm.transitions.params = [transition_matrix]
        fit_ll = glm_hmm.fit(observations, inputs=inputs, masks=masks, method=fitting_method, num_iters=n_iters, initialize=False, tolerance=tolerance)
        return n_states, init_num, glm_hmm, fit_ll

    # Dispatch every (n_states, init_num) fit in one pool to keep all cores saturated.
    tasks = [(n_states, init_num) for n_states in state_range for init_num in range(n_initializations)]
    with threadpool_limits(limits=1):
        results = Parallel(n_jobs=n_jobs)(delayed(fit_single_initialization)(n_states, init_num) for n_states, init_num in tasks)

    # Regroup by state, preserving the original init_num ordering within each state.
    models_glm_hmm = {n_states: [None] * n_initializations for n_states in state_range}
    fit_lls_glm_hmm = {n_states: [None] * n_initializations for n_states in state_range}
    for n_states, init_num, glm_hmm, fit_ll in results:
        models_glm_hmm[n_states][init_num] = glm_hmm
        fit_lls_glm_hmm[n_states][init_num] = fit_ll
    return models_glm_hmm, fit_lls_glm_hmm


def session_wise_fit(observations, inputs, masks, n_sessions, init_params, n_states, fitting_method="em", n_iters=200, tolerance=10**-4, n_jobs=-1):
    """
    Session-wise GLM-HMM fitting (one fit per session) with BLAS threads pinned.
    """
    masks = [np.ones_like(arr) for arr in observations] if masks is None else masks
    assert len(observations) == n_sessions, "Observations are not compatible with number of sessions!"
    assert len(inputs) == n_sessions, "Inputs are not compatible with number of sessions!"
    assert len(masks) == n_sessions, "Masks are not compatible with number of sessions!"
    assert "transition_matrices" in init_params and "glm_weights" in init_params, "Initial parameters not provided correctly!"

    obs_dim = observations[0].shape[1]
    input_dim = inputs[0].shape[1]
    C = len(np.unique(observations[0]))

    def process_session(idx_session):
        """
        Fit a GLM-HMM for a specific session.
        """
        glm_hmm = ssm.HMM(n_states, obs_dim, input_dim, observations="input_driven_obs", observation_kwargs=dict(C=C), transitions="standard")
        glm_hmm.observations.params = init_params["glm_weights"][idx_session]
        glm_hmm.transitions.params = init_params["transition_matrices"][idx_session]

        fit_ll = glm_hmm.fit(observations[idx_session], inputs=inputs[idx_session], masks=masks[idx_session], method=fitting_method, num_iters=n_iters, initialize=False, tolerance=tolerance)
        return idx_session, glm_hmm, fit_ll

    # Single pool over sessions; pin BLAS so the workers don't oversubscribe the cores.
    with threadpool_limits(limits=1):
        results = Parallel(n_jobs=n_jobs)(delayed(process_session)(idx_session) for idx_session in tqdm(range(n_sessions), desc="Fitting sessions"))

    models_session = {idx_session: model for idx_session, model, _ in results}
    fit_ll_session = {idx_session: fit_ll for idx_session, _, fit_ll in results}

    return models_session, fit_ll_session
