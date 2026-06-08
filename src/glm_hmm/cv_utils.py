"""Session-wise k-fold cross-validated GLM-HMM fitting.

Every ``(session, state, fold)`` fit is dispatched in a single joblib pool so all
cores stay busy and the worker pool is created only once, with inner BLAS/OpenMP
threads pinned to 1 so the workers don't oversubscribe the cores. The per-fold EM
uses ``initialize=False`` and is fully deterministic.
"""

import numpy as np
import ssm
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from tqdm import tqdm


def split_non_continuous(arr):
    arr = np.array(arr)
    # Find where the difference between consecutive elements is not 1
    breaks = np.where(np.diff(arr) != 1)[0] + 1
    # Use np.split to divide the array at those indices
    return np.split(arr, breaks)


def cross_validation_split(session_length, idx_split=0, n_sub_block=4, k_folds=5):
    assert 0 <= idx_split < 5, "Number of splits must be between [0,5)!"
    block_length = session_length // n_sub_block // k_folds

    test_idx = []
    for i in range(n_sub_block):
        if i == n_sub_block - 1 and idx_split == k_folds - 1:
            end_idx = session_length
            start_idx = (i * k_folds + idx_split) * block_length
        else:
            end_idx = (i * k_folds + idx_split + 1) * block_length
            start_idx = (i * k_folds + idx_split) * block_length
        test_idx.append(np.arange(start_idx, end_idx))

    train_idx = np.setdiff1d(np.arange(session_length), np.concatenate(test_idx))
    train_idx = split_non_continuous(train_idx)
    return train_idx, test_idx


def session_wise_fit_cv(observations, inputs, masks, n_sessions, init_params, k_folds=5, state_range=np.arange(2, 6), fitting_method="em", n_iters=200, tolerance=10**-4, n_jobs=-1):
    """
    Session-wise k-fold cross-validated GLM-HMM fitting, all fits dispatched in one pool.
    """

    masks = [np.ones_like(arr) for arr in observations] if masks is None else masks
    assert len(observations) == n_sessions, "Observations are not compatible with number of sessions!"
    assert len(inputs) == n_sessions, "Inputs are not compatible with number of sessions!"
    assert len(masks) == n_sessions, "Masks are not compatible with number of sessions!"
    assert "transition_matrices" in init_params.keys() and "glm_weights" in init_params.keys(), "Initial parameters not provided correctly!"

    obs_dim = observations[0].shape[1]
    input_dim = inputs[0].shape[1]
    C = len(np.unique(observations[0]))

    def fit_model_on_fold(idx_session, state_idx, n_states, idx_split):
        """
        Fit a GLM-HMM on one fold and compute training and testing log-likelihoods.
        """
        glm_hmm = ssm.HMM(n_states, obs_dim, input_dim, observations="input_driven_obs", observation_kwargs=dict(C=C), transitions="standard")
        glm_hmm.observations.params = np.copy(init_params["glm_weights"][n_states])
        glm_hmm.transitions.params = np.copy(init_params["transition_matrices"][n_states])

        session_length = observations[idx_session].shape[0]
        train_idx, test_idx = cross_validation_split(session_length, idx_split, k_folds=k_folds)
        train_obs = [observations[idx_session][train] for train in train_idx]
        test_obs = [observations[idx_session][test] for test in test_idx]
        train_masks = [masks[idx_session][train] for train in train_idx]
        test_masks = [masks[idx_session][test] for test in test_idx]
        train_inputs = [inputs[idx_session][train] for train in train_idx]
        test_inputs = [inputs[idx_session][test] for test in test_idx]

        # Fit the model on the training data
        glm_hmm.fit(train_obs, inputs=train_inputs, masks=train_masks, method=fitting_method, num_iters=n_iters, initialize=False, tolerance=tolerance)
        test_ll = glm_hmm.log_likelihood(test_obs, inputs=test_inputs, masks=test_masks)
        train_ll = glm_hmm.log_likelihood(train_obs, inputs=train_inputs, masks=train_masks)
        return idx_session, state_idx, n_states, idx_split, glm_hmm, train_ll, test_ll

    # Dispatch every (session, state, fold) fit in a single pool so all cores stay busy and the
    # joblib pool is created only once (previously one pool per session x state, 5-way parallel only).
    tasks = [(idx_session, state_idx, n_states, idx_split) for idx_session in range(n_sessions) for state_idx, n_states in enumerate(state_range) for idx_split in range(k_folds)]
    with threadpool_limits(limits=1):
        results = Parallel(n_jobs=n_jobs)(delayed(fit_model_on_fold)(idx_session, state_idx, n_states, idx_split) for idx_session, state_idx, n_states, idx_split in tqdm(tasks, desc="Fitting CV folds"))

    # Scatter results back into the original structures (index-safe via returned keys).
    models_session_state_fold = {idx_session: {n_states: [None] * k_folds for n_states in state_range} for idx_session in range(n_sessions)}
    train_ll = np.full((n_sessions, len(state_range), k_folds), np.nan)
    test_ll = np.full((n_sessions, len(state_range), k_folds), np.nan)
    for idx_session, state_idx, n_states, idx_split, glm_hmm, train_ll_val, test_ll_val in results:
        models_session_state_fold[idx_session][n_states][idx_split] = glm_hmm
        train_ll[idx_session, state_idx, idx_split] = train_ll_val
        test_ll[idx_session, state_idx, idx_split] = test_ll_val

    return models_session_state_fold, train_ll, test_ll
