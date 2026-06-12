#!/usr/bin/env bash
# Submit the shared-basis fit as a K-sharded Slurm array, then a dependent merge.
# The --array range is computed dynamically from --list-jobs so it always matches the
# config's state range. Mirrors submit_glm_hmm.sh / submit_glm_hmm_finetuning.sh.
#
# Usage: bash scripts/slurm/submit_shared_glm_hmm.sh [--config NAME] [--color 01|pm1] [--list-jobs]
#   --config     feature variant from src.shared_glm_hmm.experiments.CONFIGS
#                (default XY__tgt); overrides any CONFIG env var.
#   --color      color coding: 01 (recommended) or pm1 (default 01).
#   --list-jobs  print the K work units for the chosen config and exit WITHOUT submitting.
#   CONFIG=.. COLOR=.. env vars are also honored (flags win).
#
# Examples:
#   bash scripts/slurm/submit_shared_glm_hmm.sh --config XY__tgt --list-jobs
#   bash scripts/slurm/submit_shared_glm_hmm.sh --config XY__tgt
#   CONFIG=XY__tgt COLOR=pm1 bash scripts/slurm/submit_shared_glm_hmm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${HOME}/bayesian-priors-in-pd-subtype/.env"

CONFIG="${CONFIG:-XY__tgt}"
COLOR="${COLOR:-01}"
LIST_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)             CONFIG="$2"; shift 2 ;;
        --color|--color-coding) COLOR="$2"; shift 2 ;;
        --list-jobs)          LIST_ONLY=1; shift ;;
        -h|--help)            sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
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

echo
echo "When ${MERGE_ID} finishes: open notebooks/6.00 to choose K, then"
echo "  bash ${SCRIPT_DIR}/submit_shared_glm_hmm_finetuning.sh <K> --config ${CONFIG} --color ${COLOR}"
