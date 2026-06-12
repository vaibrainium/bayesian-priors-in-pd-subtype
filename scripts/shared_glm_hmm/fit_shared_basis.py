"""Fit one shared HC+PD GLM-HMM state basis (pooled CV over all sessions).

Pools every session (HC + all PD, both medication states) into a single group and runs the
existing pooled cross-validation, producing one shared state basis instead of five
independent per-(subtype x medication) models. Output schema matches
``scripts/glm_hmm/model_fitting.py`` (pooled mode), so the analysis notebooks read it
unchanged.

    python3 scripts/shared_glm_hmm/fit_shared_basis.py --config ashwood_color

Output: ``shared_glm_hmm/<run_name>/all_subjects.pkl`` (``run_name`` = config, plus a
``__colorpm1`` suffix when ``--color-coding pm1`` so both codings can coexist).
Heavy (~100 sessions x ~620 trials through 5-fold CV x state range); run on a big box or
via the Slurm wrapper.
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for `config` / `src`

import pandas as pd

from config import dir_config
from src.glm_hmm.cv_utils import select_best_n_states
from src.shared_glm_hmm.experiments import CONFIGS
from src.shared_glm_hmm.fitting import prepare_group, run_pooled_cv
from src.shared_glm_hmm.grouping import get_pooled_session_ids


def load_data(processed_dir: Path, color_coding: str = "01"):
    data = pd.read_csv(processed_dir / "processed_all_data_accu_60_all.csv")
    for col in ("choice", "target", "outcome"):
        data[col] = data[col].fillna(-1).astype(int)
    if color_coding == "pm1":
        # Symmetric coding: 0 (equal) -> -1, 1 (unequal) -> +1; NaN (invalid) preserved.
        data["color"] = data["color"].map({0.0: -1.0, 1.0: 1.0})
    metadata = pd.read_csv(processed_dir / "processed_metadata_all_data_accu_60.csv")
    return data, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="ashwood_color", choices=list(CONFIGS), help="Feature variant to fit (from src.shared_glm_hmm.experiments).")
    parser.add_argument("--color-coding", default="01", choices=["01", "pm1"], help="`color` regressor coding: 01 (equal=0, unequal=1; recommended) or pm1 (equal=-1, unequal=+1).")
    parser.add_argument("--force", action="store_true", help="Refit even if all_subjects.pkl already exists.")
    args = parser.parse_args()

    config = CONFIGS[args.config]
    run_name = args.config + ("__colorpm1" if args.color_coding == "pm1" else "")
    processed_dir = Path(dir_config.data.processed)
    out_dir = processed_dir / "shared_glm_hmm" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_subjects.pkl"

    if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
        print(f"skip: {out_path} already on disk (use --force to refit)")
        return

    data, metadata = load_data(processed_dir, args.color_coding)
    groups = get_pooled_session_ids(metadata)
    session_ids = groups["all_subjects"]
    print(f"=== shared basis: run={run_name}  n_sessions={len(session_ids)}  states={list(config.state_range)} ===")

    prep = prepare_group(data, session_ids, config)
    payload = run_pooled_cv(prep, config, run_name)

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    selection = select_best_n_states(payload["group_pooled_cv"], payload["data"], rule=config.selection_rule, tol=config.selection_tol)
    print(f"saved: {out_path}")
    print(f"1-SEM reference best_k = {selection['best']} (peak {selection['best_unconstrained']}); "
          f"bits/trial = {[round(b, 4) for b in selection['mean_bits']]}")
    print("Final K should be chosen from the triangulation figure in notebook 6.00.")


if __name__ == "__main__":
    main()
