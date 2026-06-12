"""Finetune the shared-basis CV bundle at a chosen K.

Reuses the pooled finetuning logic (per-session models + one shared pooled model) at the K
selected from the triangulation figure (notebook 6.00). Output schema matches
``scripts/glm_hmm/model_finetuning.py`` (``<group>_final.pkl``), so notebooks 5.30/6.01
read it unchanged.

    python3 scripts/shared_glm_hmm/finetune_shared_basis.py --best-k 4

Output: ``shared_glm_hmm/<run_name>/all_subjects_final.pkl``.
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for `config` / `src`

from config import dir_config
from src.shared_glm_hmm.experiments import CONFIGS
from src.shared_glm_hmm.fitting import finetune_pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--best-k", type=int, required=True, help="State count to finetune at (from the 6.00 triangulation figure).")
    parser.add_argument("--config", default="ashwood_color", choices=list(CONFIGS), help="Feature variant (must already be CV-fit).")
    parser.add_argument("--color-coding", default="01", choices=["01", "pm1"], help="Must match the coding used at fit time (selects the run directory).")
    parser.add_argument("--n-iters", type=int, default=2500, help="EM iterations for the finetuning fits (default: 2500).")
    parser.add_argument("--force", action="store_true", help="Refit even if all_subjects_final.pkl exists.")
    args = parser.parse_args()

    run_name = args.config + ("__colorpm1" if args.color_coding == "pm1" else "")
    model_dir = Path(dir_config.data.processed) / "shared_glm_hmm" / run_name
    cv_path = model_dir / "all_subjects.pkl"
    out_path = model_dir / "all_subjects_final.pkl"

    if not cv_path.exists() or cv_path.stat().st_size == 0:
        parser.error(f"CV bundle not found: {cv_path}. Run fit_shared_basis.py first.")
    if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
        print(f"skip: {out_path} already on disk (use --force to refit)")
        return

    with open(cv_path, "rb") as f:
        bundle = pickle.load(f)

    state_range = list(bundle["group_pooled_cv"]["state_range"])
    if args.best_k not in state_range:
        parser.error(f"--best-k {args.best_k} not in fitted state range {state_range}")

    print(f"=== finetuning shared basis: run={run_name}  best_k={args.best_k}  n_iters={args.n_iters} ===")
    finetuned = finetune_pooled(bundle, args.best_k, n_iters=args.n_iters)

    payload = {
        "model": finetuned,
        "best_k": args.best_k,
        "cv_type": "pooled_cv",
        "data": bundle["data"],
        "config": bundle["config"],
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"saved: {out_path}  ({len(finetuned['models'])} session models + pooled)")


if __name__ == "__main__":
    main()
