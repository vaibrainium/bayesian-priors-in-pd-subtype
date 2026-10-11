"""Held-out-participant and held-out-session CV for the shared GLM-HMM (review comment 5).

Refits the selected shared-basis model (``XY__tgt``, the Figure 4 model) on the identical design
matrices, but with folds that hold out whole sessions rather than interleaved trial blocks.
See :mod:`src.shared_glm_hmm.heldout` for the fold schemes and the initialisation.

    python3 scripts/shared_glm_hmm/fit_heldout_cv.py --n-jobs 20

For the participant scheme it also refits each held-out session at ``--best-k``, warm-started from
the fold's pooled model, exactly as ``finetune_shared_basis.py`` does with the full-data model. That
gives every session's occupancy-weighted prior effect from a model that never saw the participant.

Every fit is cached as a shard under ``outputs/glm_hmm_heldout_cv/shards`` so an interrupted run
resumes where it stopped. Reads the data directory, never writes to it.
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for `config` / `src`

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from config import dir_config, main_config
from src.glm_hmm.fitting_utils import session_wise_fit
from src.shared_glm_hmm.fitting import session_arrays
from src.shared_glm_hmm.grouping import session_metadata
from src.shared_glm_hmm.heldout import assign_folds, fit_glm, fit_restart, session_group, session_log_likelihoods

SCHEMES = ("participant", "session")


def subset(arrays, index):
    return [[a[i] for i in index] for a in arrays]


def cached(path, compute):
    """Load a shard if present, otherwise compute and write it."""
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            return pickle.load(f)
    result = compute()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(result, f)
    tmp.rename(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=main_config.GLM_HMM.final_model)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--n-inits", type=int, default=6, help="Restarts per (fold, K>1); the best training log-likelihood is kept.")
    parser.add_argument("--best-k", type=int, default=4, help="K of the reported model, for the held-out session refits.")
    parser.add_argument("--seed", type=int, default=0, help="Fold assignment seed.")
    parser.add_argument("--n-jobs", type=int, default=20)
    args = parser.parse_args()

    processed_dir = Path(dir_config.data.processed)
    out_dir = Path(__file__).resolve().parents[2] / "outputs" / "glm_hmm_heldout_cv"
    assert not str(out_dir.resolve()).startswith(str(processed_dir.parents[0])), "never write to the data dir"
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    with open(processed_dir / "shared_glm_hmm" / args.config / "all_subjects.pkl", "rb") as f:
        bundle = pickle.load(f)
    config = bundle["config"]
    prior_sigma = config.get("prior_sigma", 2.0)
    state_range = [int(k) for k in config["state_range"]]
    sessions = list(bundle["data"].keys())
    arrays = session_arrays(bundle)  # observations, inputs, masks, in `sessions` order

    metadata = pd.read_csv(processed_dir / "processed_metadata_all_data_accu_60.csv")
    sess_md = session_metadata(metadata).set_index("session_id").loc[sessions].reset_index()
    sess_md["group"] = session_group(sess_md).to_numpy()
    folds = {scheme: assign_folds(sess_md, scheme, args.k_folds, args.seed) for scheme in SCHEMES}

    def split(scheme, fold):
        train = [i for i, s in enumerate(sessions) if folds[scheme][s] != fold]
        test = [i for i, s in enumerate(sessions) if folds[scheme][s] == fold]
        return train, test

    print(f"config={args.config}  sessions={len(sessions)}  K={state_range}  folds={args.k_folds}  restarts={args.n_inits}")
    for scheme in SCHEMES:
        sizes = pd.Series(folds[scheme]).value_counts().sort_index().tolist()
        print(f"  {scheme}: test sessions per fold {sizes}")

    # --- stage 1: one GLM per (scheme, fold), on the training sessions only ---------------------
    def glm_task(scheme, fold):
        train, _ = split(scheme, fold)
        return cached(shard_dir / f"glm_{scheme}_f{fold}.pkl", lambda: fit_glm(*subset(arrays, train), prior_sigma=prior_sigma).observations.params)

    glm_keys = [(s, f) for s in SCHEMES for f in range(args.k_folds)]
    with threadpool_limits(limits=1):
        glm_weights = dict(zip(glm_keys, Parallel(n_jobs=args.n_jobs)(delayed(glm_task)(s, f) for s, f in glm_keys)))
    print("stage 1 done: fold GLMs")

    # --- stage 2: every (scheme, fold, K, restart) -------------------------------------------------
    def restart_task(scheme, fold, n_states, init_num):
        def compute():
            train, _ = split(scheme, fold)
            obs, inp, msk = subset(arrays, train)
            hmm = fit_restart(glm_weights[(scheme, fold)], obs, inp, msk, n_states, init_num, prior_sigma=prior_sigma)
            return {"model": hmm, "train_ll": hmm.log_likelihood(obs, inputs=inp, masks=msk)}

        return cached(shard_dir / f"hmm_{scheme}_f{fold}_k{n_states}_i{init_num}.pkl", compute)

    tasks = [(s, f, k, i) for k in sorted(state_range, reverse=True) for s in SCHEMES for f in range(args.k_folds) for i in range(1 if k == 1 else args.n_inits)]
    print(f"stage 2: {len(tasks)} restarts")
    with threadpool_limits(limits=1):
        fits = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(restart_task)(*t) for t in tasks)

    best = {}
    for (scheme, fold, n_states, _), fit in zip(tasks, fits):
        key = (scheme, fold, n_states)
        if key not in best or fit["train_ll"] > best[key]["train_ll"]:
            best[key] = fit

    # --- stage 3: score held-out sessions with parameters fixed ------------------------------------
    rows = []
    for (scheme, fold, n_states), fit in best.items():
        _, test = split(scheme, fold)
        lls = session_log_likelihoods(fit["model"], *subset(arrays, test))
        for i, ll in zip(test, lls):
            rows.append({"scheme": scheme, "fold": fold, "n_states": n_states, "session_id": sessions[i], "test_ll": ll, "n_valid": int(arrays[2][i].sum()), "train_ll": fit["train_ll"]})
    scores = pd.DataFrame(rows).merge(sess_md[["session_id", "subject_id", "group"]], on="session_id")
    print("stage 3 done: held-out scores")

    # --- stage 4: held-out-participant session refits at best K ---------------------------------
    def refit_fold(fold):
        def compute():
            _, test = split("participant", fold)
            init = best[("participant", fold, args.best_k)]["model"]
            # one copy per session: session_wise_fit assigns these arrays without copying, so with
            # n_jobs=1 a shared array would be updated in place by every session's EM
            init_params = {
                "glm_weights": {j: np.copy(init.observations.params) for j in range(len(test))},
                "transition_matrices": {j: tuple(np.copy(p) for p in init.transitions.params) for j in range(len(test))},
            }
            models, _ = session_wise_fit(*subset(arrays, test), n_sessions=len(test), init_params=init_params, n_states=args.best_k, n_iters=2500, prior_sigma=prior_sigma, n_jobs=1)
            return {sessions[i]: models[j] for j, i in enumerate(test)}

        return cached(shard_dir / f"sessionfit_participant_f{fold}_k{args.best_k}.pkl", compute)

    with threadpool_limits(limits=1):
        session_models = Parallel(n_jobs=min(args.n_jobs, args.k_folds))(delayed(refit_fold)(f) for f in range(args.k_folds))
    print("stage 4 done: held-out session refits")

    payload = {
        "config": config,
        "config_name": args.config,
        "folds": folds,
        "sessions": sessions,
        "session_metadata": sess_md,
        "scores": scores,
        "fold_models": {key: fit["model"] for key, fit in best.items()},
        "heldout_session_models": {s: m for fold_models in session_models for s, m in fold_models.items()},
        "settings": vars(args),
    }
    with open(out_dir / "heldout_cv.pkl", "wb") as f:
        pickle.dump(payload, f)
    print(f"saved {out_dir / 'heldout_cv.pkl'}")


if __name__ == "__main__":
    main()
