"""Principled K (state-count) selection for the shared basis.

Three independent criteria to triangulate, complementing the existing CV bits/trial:

* ``icl_bic``  - BIC and ICL per K from the stored global fits on the full data. BIC
  penalises parameters; ICL adds the posterior assignment entropy, so it specifically
  rewards *well-separated* states (the quantity that matches "how many distinct
  strategies"). Lower is better for both.
* ``bootstrap_state_stability`` - resample sessions with replacement, refit the pooled
  model at each K, Hungarian-align to a reference, and report how reproducibly the state
  weights re-recover. The largest K that stays reproducible is the supported K.

All heavy fitting is reused from ``src.glm_hmm.fitting_utils``; nothing here refits the CV.
"""

import numpy as np
import numpy.random as npr
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from src.glm_hmm.fitting_utils import group_wise_fit
from src.shared_glm_hmm.fitting import session_arrays


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _best_global_model(bundle, n_states):
    """Best global init at ``n_states`` by full-data log-likelihood."""
    models = bundle["global"]["models"][n_states]
    obs, inp, msk = session_arrays(bundle)
    lls = [m.log_likelihood(obs, inputs=inp, masks=msk) for m in models]
    return models[int(np.argmax(lls))], obs, inp, msk


def _n_params(n_states, input_dim):
    """Free parameters: emissions K*input_dim + transitions K*(K-1) + initial (K-1)."""
    return n_states * input_dim + n_states * (n_states - 1) + (n_states - 1)


def _align_perm(weights, ref_weights):
    """Hungarian permutation aligning ``weights`` rows to ``ref_weights`` (both K x M)."""
    K = weights.shape[0]
    cost = np.array([[np.sum((weights[i] - ref_weights[j]) ** 2) for j in range(K)] for i in range(K)])
    _, col = linear_sum_assignment(cost)
    inv = np.empty_like(col)
    inv[col] = np.arange(K)
    return inv


# --------------------------------------------------------------------------- #
# ICL / BIC
# --------------------------------------------------------------------------- #
def icl_bic(bundle):
    """Per-K BIC and ICL from the stored global models, evaluated on the full data.

    Returns a DataFrame indexed by state count with columns
    ``ll, n_params, n_trials, bic, entropy, icl`` (lower bic/icl = better).
    """
    state_range = list(np.asarray(bundle["group_pooled_cv"]["state_range"]))
    input_dim = len(bundle["config"]["model_features"])
    rows = []

    for K in state_range:
        model, obs, inp, msk = _best_global_model(bundle, K)
        ll = float(model.log_likelihood(obs, inputs=inp, masks=msk))
        n_trials = int(sum(m.sum() for m in msk))

        # posterior assignment entropy over valid trials (mean-field ICL penalty)
        entropy = 0.0
        for o, i, m in zip(obs, inp, msk):
            gamma = model.expected_states(data=o, input=i, mask=m)[0]  # (T, K)
            valid = m[:, 0]
            g = np.clip(gamma[valid], 1e-12, 1.0)
            entropy += float(-np.sum(g * np.log(g)))

        p = _n_params(K, input_dim)
        bic = -2 * ll + p * np.log(n_trials)
        icl = bic + 2 * entropy
        rows.append(dict(state=int(K), ll=ll, n_params=p, n_trials=n_trials, bic=bic, entropy=entropy, icl=icl))

    return pd.DataFrame(rows).set_index("state")


# --------------------------------------------------------------------------- #
# Bootstrap state-stability
# --------------------------------------------------------------------------- #
def _one_bootstrap(obs, inp, msk, init_flat, K, n_iters, prior_sigma, ref_w, seed):
    """One bootstrap resample + pooled refit; return mean per-state weight correlation to ref."""
    rng = npr.RandomState(seed)
    idx = rng.choice(len(obs), len(obs), replace=True)
    o_b = [obs[i] for i in idx]
    i_b = [inp[i] for i in idx]
    m_b = [msk[i] for i in idx]
    boot_model, _ = group_wise_fit(o_b, i_b, m_b, init_flat, n_states=K, n_iters=n_iters, prior_sigma=prior_sigma)
    boot_w = -boot_model.observations.params[:, 0, :]
    aligned = boot_w[_align_perm(boot_w, ref_w)]
    return float(np.nanmean([np.corrcoef(ref_w[k], aligned[k])[0, 1] for k in range(K)]))


def bootstrap_state_stability(bundle, state_range=None, B=20, n_iters=300, seed=0, n_jobs=-1):
    """Reproducibility of the K-state solution across session bootstraps.

    For each K: fit a reference pooled model on all sessions, then refit on ``B`` session
    resamples (with replacement), Hungarian-align each to the reference, and score the mean
    per-state Pearson correlation of GLM weights. Higher = more reproducible. The ``B`` refits
    per K run in parallel (joblib, ``n_jobs``; BLAS pinned to 1 thread per worker).

    ``n_iters`` is kept modest (refits are warm-started from the reference init) because
    this is the expensive criterion; restrict ``state_range`` to a window around the
    candidate K to keep runtime bounded.

    Returns a DataFrame indexed by state count with ``mean_stability, sem_stability, B``.
    """
    if state_range is None:
        state_range = list(np.asarray(bundle["group_pooled_cv"]["state_range"]))
    init_all = bundle["global"]["init_params"]
    obs, inp, msk = session_arrays(bundle)
    prior_sigma = bundle["config"].get("prior_sigma", 2.0)
    rows = []

    for K in state_range:
        K = int(K)
        if K < 2:
            rows.append(dict(state=K, mean_stability=np.nan, sem_stability=np.nan, B=0))
            continue

        init_flat = {
            "glm_weights": init_all["glm_weights"][K],
            "transition_matrices": init_all["transition_matrices"][K],
        }
        ref_model, _ = group_wise_fit(obs, inp, msk, init_flat, n_states=K, n_iters=n_iters, prior_sigma=prior_sigma)
        ref_w = -ref_model.observations.params[:, 0, :]  # (K, M)

        with threadpool_limits(limits=1):
            corrs = Parallel(n_jobs=n_jobs)(delayed(_one_bootstrap)(obs, inp, msk, init_flat, K, n_iters, prior_sigma, ref_w, seed + b) for b in range(B))
        corrs = np.array(corrs)
        rows.append(dict(state=K, mean_stability=float(corrs.mean()), sem_stability=float(corrs.std() / np.sqrt(B)), B=B))

    return pd.DataFrame(rows).set_index("state")
