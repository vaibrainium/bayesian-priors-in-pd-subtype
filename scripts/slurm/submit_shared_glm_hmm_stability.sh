#!/usr/bin/env bash
# Submit the K-selection metric precompute (ICL/BIC + bootstrap stability) -> CSVs that
# notebook 6.00 loads. A single work unit; runs after the fit+merge has produced
# all_subjects.pkl.
#
# Usage: bash scripts/slurm/submit_shared_glm_hmm_stability.sh [--config NAME] [--color 01|pm1] [--B N] [--n-iters N]
#   --config   default ashwood_color_non_standardized; overrides CONFIG env. Must match the fit.
#   --color    01 (default) or pm1.
#   --B        bootstrap resamples per K (default 30).
#   --n-iters  EM iters per bootstrap refit (default 300).
#
# Example:
#   bash scripts/slurm/submit_shared_glm_hmm_stability.sh --config ashwood_color_non_standardized --B 30

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${CONFIG:-ashwood_color_non_standardized}"
COLOR="${COLOR:-01}"
B="${B:-30}"
NITERS="${NITERS:-300}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)               CONFIG="$2"; shift 2 ;;
        --color|--color-coding)  COLOR="$2"; shift 2 ;;
        --B)                    B="$2"; shift 2 ;;
        --n-iters)              NITERS="$2"; shift 2 ;;
        -h|--help)              sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "Config  : ${CONFIG}  (color=${COLOR})"
echo "Stability: B=${B}  n_iters=${NITERS}"
echo

sbatch \
    --export=ALL,CONFIG="${CONFIG}",COLOR="${COLOR}",B="${B}",NITERS="${NITERS}" \
    "${SCRIPT_DIR}/shared_glm_hmm_stability.slurm"
