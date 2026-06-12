#!/usr/bin/env bash
# Submit the shared-basis fit as a K-sharded Slurm array, then a dependent merge.
# The --array range is computed dynamically from --list-jobs so it always matches the
# config's state range. Mirrors submit_glm_hmm.sh / submit_glm_hmm_finetuning.sh.
#
# Usage: bash scripts/slurm/submit_shared_glm_hmm.sh
#   CONFIG  feature variant from src.shared_glm_hmm.experiments.CONFIGS (default ashwood_color)
#   COLOR   color coding: 01 (recommended) or pm1                       (default 01)
#
# Examples:
#   bash scripts/slurm/submit_shared_glm_hmm.sh
#   CONFIG=ashwood_color COLOR=pm1 bash scripts/slurm/submit_shared_glm_hmm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${HOME}/bayesian-priors-in-pd-subtype/.env"

CONFIG="${CONFIG:-ashwood_color}"
COLOR="${COLOR:-01}"

# Query the K work-unit count dynamically from the launcher itself (run in-container so the
# imports resolve exactly as the array tasks will). Capture fully to avoid BrokenPipeError.
LIST_OUTPUT=$(singularity exec \
        --bind ${CLUSTER_DATA_PATH}:/mnt/pd-data \
        --bind ${PROJECT_ROOT}:/src \
        ${SIF_PATH} \
        bash -c "export PYTHONPATH=/src && python3 /src/scripts/shared_glm_hmm/fit_shared_basis_slurm.py --config ${CONFIG} --color-coding ${COLOR} --list-jobs")
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
echo "  BESTK=<K> CONFIG=${CONFIG} COLOR=${COLOR} bash ${SCRIPT_DIR}/submit_shared_glm_hmm_finetuning.sh"
