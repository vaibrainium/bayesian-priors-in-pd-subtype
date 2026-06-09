"""K-fold cross-validated GLM-HMM fitting, both session-wise and pooled.

Two cross-validation designs share the same interleaved-block split
(``cross_validation_split``):

* ``session_wise_fit_cv`` -- one GLM-HMM per session, cross-validated *within* each
  session.
* ``group_wise_fit_cv`` -- all sessions of a subject group pooled into a single
  model per fold (Ashwood et al. 2022 style), with ``n_initializations`` restarts
  per ``(state, fold)``. The pooled selection helpers (``bernoulli_null_loglik``,
  ``pooled_bits_per_trial``, ``select_best_n_states``) score the result in
  held-out bits/trial and pick the most parsimonious state count.

Every ``(session/state/fold[/init])`` fit is dispatched in a single joblib pool so
all cores stay busy and the worker pool is created only once, with inner
BLAS/OpenMP threads pinned to 1 so the workers don't oversubscribe the cores. The
per-fold EM uses ``initialize=False`` and is fully deterministic.
"""

import numpy as np
import numpy.random as npr
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


def _perturb_init(glm_weights, n_states, input_dim, init_num):
    """Return (perturbed glm_weights, transition_matrix) for a restart.

    ``init_num == 0`` returns the unperturbed global init so the original solution is
    always among the candidates; later restarts jitter the weights and randomize the
    transition matrix, mirroring the scheme in :func:`src.glm_hmm.fitting_utils.global_fit`.
    """
    if init_num == 0:
        return np.copy(glm_weights), None
    npr.seed(init_num * n_states)
    weights = glm_weights + npr.normal(0, 0.2, (n_states, 1, input_dim))
    transition_matrix = 0.9 * np.eye(n_states) + npr.multivariate_normal(mean=np.zeros(n_states), cov=0.05 * np.eye(n_states), size=n_states)
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
    return weights, transition_matrix


def group_wise_fit_cv(observations, inputs, masks, init_params, k_folds=5, state_range=np.arange(1, 6), fitting_method="em", n_iters=2500, tolerance=10**-4, n_initializations=5, n_jobs=-1):
    """Pooled (group-level) k-fold cross-validated GLM-HMM fitting.

    Parameters mirror :func:`src.glm_hmm.cv_utils.session_wise_fit_cv`. ``observations``,
    ``inputs`` and ``masks`` are per-session lists; for each ``(n_states, fold)`` a model is
    fit on the pooled training blocks of *all* sessions and scored on the pooled test blocks.
    ``init_params`` is the global (pooled) fit's init. ``n_initializations`` random restarts
    are run per ``(state, fold)`` (restart 0 = the unperturbed global init); the one with the
    highest pooled train log-likelihood is kept, smoothing EM local-optima wiggles in the
    selection curve. Pass ``n_initializations=1`` to recover the single-init behaviour.

    Returns
    -------
    models : dict[int, list]
        ``models[n_states][fold]`` -> the best pooled ``ssm.HMM`` for that fold.
    train_ll, test_ll : np.ndarray, shape (len(state_range), k_folds)
        Pooled summed log-likelihood (nats) on the train / test blocks of the kept restart.
    n_test_trials, n_train_trials : np.ndarray, shape (len(state_range), k_folds)
        Number of *valid* (unmasked) trials in the pooled test / train set, for bits/trial.
    """
    masks = [np.ones_like(arr) for arr in observations] if masks is None else masks
    n_sessions = len(observations)
    assert len(inputs) == n_sessions, "Inputs are not compatible with number of sessions!"
    assert len(masks) == n_sessions, "Masks are not compatible with number of sessions!"
    assert "transition_matrices" in init_params and "glm_weights" in init_params, "Initial parameters not provided correctly!"

    obs_dim = observations[0].shape[1]
    input_dim = inputs[0].shape[1]
    C = len(np.unique(np.concatenate(observations)))

    def fit_fold(state_idx, n_states, idx_split, init_num):
        """Fit one pooled restart on one fold and score pooled train/test log-likelihoods."""
        weights, transition_matrix = _perturb_init(init_params["glm_weights"][n_states], n_states, input_dim, init_num)
        glm_hmm = ssm.HMM(n_states, obs_dim, input_dim, observations="input_driven_obs", observation_kwargs=dict(C=C), transitions="standard")
        glm_hmm.observations.params = weights
        glm_hmm.transitions.params = np.copy(init_params["transition_matrices"][n_states]) if transition_matrix is None else [transition_matrix]

        # Pool train/test blocks across every session using the same interleaved-block split.
        train_obs, train_inputs, train_masks = [], [], []
        test_obs, test_inputs, test_masks = [], [], []
        for idx_session in range(n_sessions):
            session_length = observations[idx_session].shape[0]
            train_idx, test_idx = cross_validation_split(session_length, idx_split, k_folds=k_folds)
            train_obs += [observations[idx_session][tr] for tr in train_idx]
            train_inputs += [inputs[idx_session][tr] for tr in train_idx]
            train_masks += [masks[idx_session][tr] for tr in train_idx]
            test_obs += [observations[idx_session][te] for te in test_idx]
            test_inputs += [inputs[idx_session][te] for te in test_idx]
            test_masks += [masks[idx_session][te] for te in test_idx]

        glm_hmm.fit(train_obs, inputs=train_inputs, masks=train_masks, method=fitting_method, num_iters=n_iters, initialize=False, tolerance=tolerance)
        test_ll = glm_hmm.log_likelihood(test_obs, inputs=test_inputs, masks=test_masks)
        train_ll = glm_hmm.log_likelihood(train_obs, inputs=train_inputs, masks=train_masks)
        n_test = int(sum(m.sum() for m in test_masks))
        n_train = int(sum(m.sum() for m in train_masks))
        return state_idx, n_states, idx_split, glm_hmm, train_ll, test_ll, n_train, n_test

    # Dispatch every (state, fold, init) restart in one pool so all cores stay busy.
    tasks = [(state_idx, n_states, idx_split, init_num) for state_idx, n_states in enumerate(state_range) for idx_split in range(k_folds) for init_num in range(n_initializations)]
    with threadpool_limits(limits=1):
        results = Parallel(n_jobs=n_jobs)(delayed(fit_fold)(state_idx, n_states, idx_split, init_num) for state_idx, n_states, idx_split, init_num in tqdm(tasks, desc="Fitting pooled CV folds"))

    models = {n_states: [None] * k_folds for n_states in state_range}
    train_ll = np.full((len(state_range), k_folds), -np.inf)
    test_ll = np.full((len(state_range), k_folds), np.nan)
    n_test_trials = np.full((len(state_range), k_folds), np.nan)
    n_train_trials = np.full((len(state_range), k_folds), np.nan)
    # Keep, per (state, fold), the restart with the highest pooled train log-likelihood.
    for state_idx, n_states, idx_split, glm_hmm, train_ll_val, test_ll_val, n_train, n_test in results:
        if train_ll_val > train_ll[state_idx, idx_split]:
            models[n_states][idx_split] = glm_hmm
            train_ll[state_idx, idx_split] = train_ll_val
            test_ll[state_idx, idx_split] = test_ll_val
            n_train_trials[state_idx, idx_split] = n_train
            n_test_trials[state_idx, idx_split] = n_test

    return models, train_ll, test_ll, n_test_trials, n_train_trials


def bernoulli_null_loglik(data):
    """Pooled Bernoulli (coin-flip) null log-likelihood and valid-trial count for a group.

    ``data`` is the ``{session_id: DataFrame}`` dict saved with each group (columns
    ``choices`` and ``invalid_idx``). The null predicts every trial at the group/session's
    empirical right-choice rate; it is the reference for bits/trial.
    """
    ll, n = 0.0, 0
    for df in data.values():
        valid = ~df["invalid_idx"].values
        y = df["choices"].values[valid]
        p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        ll += float((y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
        n += int(valid.sum())
    return ll, n


def pooled_bits_per_trial(test_ll, n_test_trials, null_ll, n_valid):
    """Pooled CV test LL converted to bits/trial vs the Bernoulli null.

    Parameters
    ----------
    test_ll, n_test_trials : np.ndarray, shape (n_states, k_folds)
        Pooled summed test log-likelihood (nats) and valid test-trial count per fold.
    null_ll, n_valid : float, int
        Group null log-likelihood (nats) and total valid trials, from ``bernoulli_null_loglik``.

    Returns
    -------
    mean_bits : np.ndarray, shape (n_states,)
        bits/trial over one full held-out pass (folds tile the data, so summing is exact).
    sem_bits : np.ndarray, shape (n_states,)
        Standard error across folds (folds partition the data, so treat this as a spread
        indicator, not a strict independent-sample SEM).
    """
    mean_bits = (test_ll.sum(1) - null_ll) / (n_valid * np.log(2))
    null_per_fold = null_ll * (n_test_trials / n_valid)
    bits_fold = (test_ll - null_per_fold) / (n_test_trials * np.log(2))
    sem_bits = bits_fold.std(1) / np.sqrt(bits_fold.shape[1])
    return mean_bits, sem_bits


def select_best_n_states(pooled_fits, data, rule="1sem", tol=0.005):
    """Select the most parsimonious number of states from a pooled-CV result.

    Scores each state count by held-out bits/trial (vs a Bernoulli null), then applies a
    parsimony rule so a flat plateau is read as "fewest states that are as good as the best"
    rather than the raw argmax.

    Parameters
    ----------
    pooled_fits : dict
        Output saved by ``group_wise_fit_cv`` (keys ``test_ll``, ``n_test_trials``, ``state_range``).
    data : dict
        The group's ``{session_id: DataFrame}`` (for the null model).
    rule : {"1sem", "tol"}
        ``"1sem"`` (default): smallest K whose mean bits/trial is within 1 fold-SEM of the best K.
        ``"tol"``: smallest K within ``tol`` bits/trial of the best K.
    tol : float
        Absolute bits/trial tolerance for ``rule="tol"`` (ignored for ``"1sem"``).

    Returns
    -------
    dict with keys: ``best`` (chosen K), ``best_unconstrained`` (argmax K), ``states``,
    ``mean_bits``, ``sem_bits``, ``marginal_gain`` (bits gained per added state), ``rule``.
    """
    state_range = np.asarray(pooled_fits["state_range"])
    null_ll, n_valid = bernoulli_null_loglik(data)
    mean_bits, sem_bits = pooled_bits_per_trial(pooled_fits["test_ll"], pooled_fits["n_test_trials"], null_ll, n_valid)

    best_idx = int(np.argmax(mean_bits))
    if rule == "1sem":
        cutoff = mean_bits[best_idx] - sem_bits[best_idx]
    elif rule == "tol":
        cutoff = mean_bits[best_idx] - tol
    else:
        msg = f"Unknown rule {rule!r}; use '1sem' or 'tol'."
        raise ValueError(msg)

    # Most parsimonious state count whose mean reaches the cutoff (search up to the peak).
    chosen_idx = next(i for i in range(best_idx + 1) if mean_bits[i] >= cutoff)

    marginal_gain = np.diff(mean_bits, prepend=np.nan)
    return {
        "best": int(state_range[chosen_idx]),
        "best_unconstrained": int(state_range[best_idx]),
        "states": state_range,
        "mean_bits": mean_bits,
        "sem_bits": sem_bits,
        "marginal_gain": marginal_gain,
        "rule": rule,
    }
