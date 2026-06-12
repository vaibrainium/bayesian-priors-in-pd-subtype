"""Precompute the K-selection metrics for a shared-basis run to CSVs.

The ICL/BIC pass and (especially) the bootstrap state-stability are too heavy for a login
node, so compute them on a worker and let notebook 6.00 just load the CSVs. Reads the merged
``all_subjects.pkl`` and writes, into the same run directory:

  selection_iclbic.csv      per-K  ll, n_params, n_trials, bic, entropy, icl
  selection_stability.csv    per-K  mean_stability, sem_stability, B

    python3 scripts/shared_glm_hmm/compute_stability.py --config ashwood_color_non_standardized
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for `config` / `src`

import numpy as np

from config import dir_config
from src.shared_glm_hmm.experiments import CONFIGS
from src.shared_glm_hmm.selection import bootstrap_state_stability, icl_bic


def run_name_for(config_name: str, color_coding: str) -> str:
    return config_name + ("__colorpm1" if color_coding == "pm1" else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="ashwood_color_non_standardized", choices=list(CONFIGS))
    parser.add_argument("--color-coding", default="01", choices=["01", "pm1"])
    parser.add_argument("--B", type=int, default=30, help="Bootstrap resamples per K (default 30).")
    parser.add_argument("--n-iters", type=int, default=300, help="EM iters per bootstrap refit (default 300).")
    parser.add_argument("--min-state", type=int, default=2, help="Skip stability for K below this (default 2).")
    parser.add_argument("--force", action="store_true", help="Recompute even if CSVs already exist.")
    args = parser.parse_args()

    run_name = run_name_for(args.config, args.color_coding)
    rdir = Path(dir_config.data.processed) / "shared_glm_hmm" / run_name
    cv_path = rdir / "all_subjects.pkl"
    if not cv_path.exists() or cv_path.stat().st_size == 0:
        parser.error(f"missing {cv_path}; run the fit + merge first.")

    bundle = pickle.load(open(cv_path, "rb"))
    states = list(np.asarray(bundle["group_pooled_cv"]["state_range"]))
    print(f"=== K-selection metrics: run={run_name}  states={[int(k) for k in states]} ===")

    icl_csv = rdir / "selection_iclbic.csv"
    if icl_csv.exists() and not args.force:
        print(f"skip ICL/BIC: {icl_csv.name} already on disk (use --force)")
    else:
        ib = icl_bic(bundle)
        ib.to_csv(icl_csv)
        print(f"wrote {icl_csv.name}  (BIC-min K={int(ib['bic'].idxmin())}, ICL-min K={int(ib['icl'].idxmin())})")

    stab_csv = rdir / "selection_stability.csv"
    if stab_csv.exists() and not args.force:
        print(f"skip stability: {stab_csv.name} already on disk (use --force)")
    else:
        stab_states = [int(k) for k in states if int(k) >= args.min_state]
        print(f"bootstrap stability: B={args.B}  n_iters={args.n_iters}  states={stab_states}")
        stab = bootstrap_state_stability(bundle, state_range=stab_states, B=args.B, n_iters=args.n_iters)
        stab.to_csv(stab_csv)
        print(f"wrote {stab_csv.name}")
        print(stab.round(3).to_string())


if __name__ == "__main__":
    main()
