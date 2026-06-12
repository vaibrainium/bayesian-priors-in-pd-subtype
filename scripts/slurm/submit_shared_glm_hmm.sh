#!/usr/bin/env bash
# Submit the shared-basis pipeline as a dependency chain: a K-sharded fit array -> a merge
# -> the K-selection metric precompute (ICL/BIC + bootstrap stability). Each step runs only
# after the previous succeeds (afterok), so one command takes you end-to-end to the CSVs that
# notebook 6.00 loads. The --array range is computed dynamically from --list-jobs.
#
# Usage: bash scripts/slurm/submit_shared_glm_hmm.sh [--config NAME] [--color 01|pm1]
#                  [--no-stability] [--B N] [--n-iters N] [--list-jobs]
#   --config        feature variant from src.shared_glm_hmm.experiments.CONFIGS
#                   (default XY__tgt); overrides any CONFIG env var.
#   --color         color coding: 01 (recommended) or pm1 (default 01).
#   --no-stability  submit only fit + merge (skip the stability precompute).
#   --B             bootstrap resamples per K for stability (default 30).
#   --n-iters       EM iters per bootstrap refit for stability (default 300).
#   --list-jobs     print the K work units for the chosen config and exit WITHOUT submitting.
#   CONFIG=.. COLOR=.. env vars are also honored (flags win).
#
# Examples:
#   bash scripts/slurm/submit_shared_glm_hmm.sh --config XY__tgt
#   bash scripts/slurm/submit_shared_glm_hmm.sh --config XY__coh --no-stability
#   CONFIG=XY__tgt COLOR=pm1 bash scripts/slurm/submit_shared_glm_hmm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${HOME}/bayesian-priors-in-pd-subtype/.env"

CONFIG="${CONFIG:-XY__tgt}"
COLOR="${COLOR:-01}"
B="${B:-30}"
NITERS="${NITERS:-300}"
LIST_ONLY=0
STABILITY=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)             CONFIG="$2"; shift 2 ;;
        --color|--color-coding) COLOR="$2"; shift 2 ;;
        --no-stability)       STABILITY=0; shift ;;
        --B)                  B="$2"; shift 2 ;;
        --n-iters)            NITERS="$2"; shift 2 ;;
        --list-jobs)          LIST_ONLY=1; shift ;;
        -h|--help)            sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Resolve the K work-unit count from the launcher itself (run in-container so the imports
# match the array tasks). Capture fully to avoid BrokenPipeError.
run_list_jobs() {
    singularity exec \
        --bind ${CLUSTER_DATA_PATH}:/mnt/pd-data \
        --bind ${PROJECT_ROOT}:/src \
        ${SIF_PATH} \
        bash -c "export PYTHONPATH=/src && python3 /src/scripts/shared_glm_hmm/fit_shared_basis_slurm.py --config ${CONFIG} --color-coding ${COLOR} --list-jobs"
}

LIST_OUTPUT=$(run_list_jobs)

if [ "${LIST_ONLY}" -eq 1 ]; then
    echo "config=${CONFIG}  color=${COLOR}"
    echo "${LIST_OUTPUT}"
    echo "(--list-jobs: nothing submitted)"
    exit 0
fi

N=$(echo "${LIST_OUTPUT}" | head -1 | awk '{print $1}')
if [ -z "${N}" ] || [ "${N}" -eq 0 ]; then
    echo "No work units found. Nothing to submit." >&2
    exit 1
fi

ARRAY_RANGE="0-$(( N - 1 ))"
echo "Config  : ${CONFIG}  (color=${COLOR})"
echo "States  : ${N}  (--array=${ARRAY_RANGE})"
echo

FIT_ID=$(sbatch --parsable \
    --array="${ARRAY_RANGE}" \
    --export=ALL,CONFIG="${CONFIG}",COLOR="${COLOR}" \
    "${SCRIPT_DIR}/shared_glm_hmm_fitting.slurm")
echo "Fit array : job ${FIT_ID}"

MERGE_ID=$(sbatch --parsable \
    --dependency=afterok:"${FIT_ID}" \
    --export=ALL,CONFIG="${CONFIG}",COLOR="${COLOR}" \
    "${SCRIPT_DIR}/shared_glm_hmm_merge.slurm")
echo "Merge     : job ${MERGE_ID}  (runs after ${FIT_ID} succeeds)"

LAST_ID="${MERGE_ID}"
if [ "${STABILITY}" -eq 1 ]; then
    STAB_ID=$(sbatch --parsable \
        --dependency=afterok:"${MERGE_ID}" \
        --export=ALL,CONFIG="${CONFIG}",COLOR="${COLOR}",B="${B}",NITERS="${NITERS}" \
        "${SCRIPT_DIR}/shared_glm_hmm_stability.slurm")
    echo "Stability : job ${STAB_ID}  (runs after ${MERGE_ID} succeeds; B=${B}, n_iters=${NITERS})"
    LAST_ID="${STAB_ID}"
fi

echo
echo "Chain submitted. When ${LAST_ID} finishes: open notebooks/6.00 (CV + ICL/BIC + stability"
echo "CSVs are all precomputed) to choose K, then"
echo "  bash ${SCRIPT_DIR}/submit_shared_glm_hmm_finetuning.sh <K> --config ${CONFIG} --color ${COLOR}"
