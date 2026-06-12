"""Slurm-parallel shared-basis fit, sharded by state count K.

The shared basis is a single pooled group, so the original ``(config, group)`` array axis
collapses. The parallelism here is across **state count K**: each K is an independent
global fit + pooled CV, and joblib already saturates a node's cores *within* a K (over its
inits and folds). So this launcher fits one K per Slurm array task, writing a self-contained
shard, and a final ``--merge`` step stacks the shards into the same ``all_subjects.pkl``
bundle that ``fit_shared_basis.py`` would have produced in one process.

    # enumerate K work units (sets the --array range)
    python3 scripts/shared_glm_hmm/fit_shared_basis_slurm.py --config XY__tgt --list-jobs
    # one array task: fit a single K
    python3 scripts/shared_glm_hmm/fit_shared_basis_slurm.py --config XY__tgt --job_id $SLURM_ARRAY_TASK_ID
    # after the array finishes: assemble the bundle
    python3 scripts/shared_glm_hmm/fit_shared_basis_slurm.py --config XY__tgt --merge

Shards: ``shared_glm_hmm/<run_name>/shards/k<K>.pkl``  ->  merged: ``.../all_subjects.pkl``.
"""

import argparse
import dataclasses
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for `config` / `src`

import numpy as np

from config import dir_config
from src.glm_hmm.cv_utils import select_best_n_states
from src.shared_glm_hmm.experiments import CONFIGS
from src.shared_glm_hmm.fitting import prepare_group, run_pooled_cv
from src.shared_glm_hmm.grouping import get_pooled_session_ids

# Reuse the loader (with --color-coding remap) from the single-process driver.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_fit_shared_basis", Path(__file__).with_name("fit_shared_basis.py"))
_fsb = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fsb)
load_data = _fsb.load_data


def run_name_for(config_name: str, color_coding: str) -> str:
    return config_name + ("__colorpm1" if color_coding == "pm1" else "")


def run_dir(processed_dir: Path, run_name: str) -> Path:
    return processed_dir / "shared_glm_hmm" / run_name


# --------------------------------------------------------------------------- #
# one shard = one K
# --------------------------------------------------------------------------- #
def fit_one_state(config, run_name, data, session_ids, K):
    """Fit global+CV for a single state count and return the (single-K) payload."""
    config_k = dataclasses.replace(config, state_range=np.array([K]))
    prep = prepare_group(data, session_ids, config_k)
    return run_pooled_cv(prep, config_k, run_name)


# --------------------------------------------------------------------------- #
# merge shards -> full bundle
# --------------------------------------------------------------------------- #
def merge_shards(shard_dir: Path, full_config, run_name: str) -> dict:
    """Stack per-K shard payloads into one bundle with the full state range."""
    shards = {}
    for p in shard_dir.glob("k*.pkl"):
        if p.stat().st_size == 0:
            continue
        s = pickle.load(open(p, "rb"))
        K = int(np.asarray(s["group_pooled_cv"]["state_range"])[0])
        shards[K] = s
    if not shards:
        raise FileNotFoundError(f"no shards found in {shard_dir}")

    Ks = sorted(shards)
    ref = shards[Ks[0]]

    gpc = {
        "models": {K: shards[K]["group_pooled_cv"]["models"][K] for K in Ks},
        "train_ll": np.vstack([shards[K]["group_pooled_cv"]["train_ll"] for K in Ks]),
        "test_ll": np.vstack([shards[K]["group_pooled_cv"]["test_ll"] for K in Ks]),
        "n_test_trials": np.vstack([shards[K]["group_pooled_cv"]["n_test_trials"] for K in Ks]),
        "n_train_trials": np.vstack([shards[K]["group_pooled_cv"]["n_train_trials"] for K in Ks]),
        "state_range": np.array(Ks),
        "n_initializations": ref["group_pooled_cv"]["n_initializations"],
    }
    glob = {
        "models": {K: shards[K]["global"]["models"][K] for K in Ks},
        "fits_lls": {K: shards[K]["global"]["fits_lls"][K] for K in Ks},
        "init_params": {
            "glm_weights": {K: shards[K]["global"]["init_params"]["glm_weights"][K] for K in Ks},
            "transition_matrices": {K: shards[K]["global"]["init_params"]["transition_matrices"][K] for K in Ks},
        },
    }
    return {
        "global": glob,
        "group_pooled_cv": gpc,
        "data": ref["data"],
        "config": full_config.to_serializable(run_name),  # full state range, not the shard's single K
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="XY__tgt", choices=list(CONFIGS))
    parser.add_argument("--color-coding", default="01", choices=["01", "pm1"])
    parser.add_argument("--job_id", type=int, default=None, help="Fit only the K at this index (Slurm array task). See --list-jobs.")
    parser.add_argument("--list-jobs", action="store_true", help="Print K work-unit enumeration and the --array range, then exit.")
    parser.add_argument("--merge", action="store_true", help="Assemble shards into all_subjects.pkl (run after the array finishes).")
    parser.add_argument("--force", action="store_true", help="Recompute even if a shard / merged pkl already exists.")
    args = parser.parse_args()

    config = CONFIGS[args.config]
    run_name = run_name_for(args.config, args.color_coding)
    processed_dir = Path(dir_config.data.processed)
    rdir = run_dir(processed_dir, run_name)
    shard_dir = rdir / "shards"
    states = [int(k) for k in config.state_range]

    if args.list_jobs:
        print(f"{len(states)} work units (set Slurm --array=0-{len(states) - 1}):")
        for i, K in enumerate(states):
            print(f"  {i}: {run_name}  K={K}")
        return

    if args.merge:
        out_path = rdir / "all_subjects.pkl"
        if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
            print(f"skip merge: {out_path} already on disk (use --force)")
            return
        payload = merge_shards(shard_dir, config, run_name)
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        sel = select_best_n_states(payload["group_pooled_cv"], payload["data"], rule=config.selection_rule, tol=config.selection_tol)
        print(f"merged {len(payload['group_pooled_cv']['state_range'])} shards -> {out_path}")
        print(f"1-SEM reference best_k = {sel['best']} (peak {sel['best_unconstrained']}); bits/trial = {[round(b, 4) for b in sel['mean_bits']]}")
        print("Choose final K from the triangulation figure in notebook 6.00.")
        return

    if args.job_id is None:
        parser.error("provide --job_id N (array task), --merge, or --list-jobs")
    if not 0 <= args.job_id < len(states):
        parser.error(f"--job_id {args.job_id} out of range [0, {len(states) - 1}]")

    K = states[args.job_id]
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"k{K}.pkl"
    if shard_path.exists() and shard_path.stat().st_size > 0 and not args.force:
        print(f"skip: shard {shard_path} already on disk (use --force)")
        return

    data, metadata = load_data(processed_dir, args.color_coding)
    session_ids = get_pooled_session_ids(metadata)["all_subjects"]
    print(f"=== shard: run={run_name}  K={K}  n_sessions={len(session_ids)} ===")
    payload = fit_one_state(config, run_name, data, session_ids, K)
    with open(shard_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"saved shard: {shard_path}")


if __name__ == "__main__":
    main()
