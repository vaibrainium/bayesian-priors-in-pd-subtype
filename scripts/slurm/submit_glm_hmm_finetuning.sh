#!/usr/bin/env bash
# Submit glm_hmm_finetuning.slurm as a Slurm array job. The --array range is
# computed dynamically by calling model_finetuning.py --list-jobs so it always
# matches what's actually on disk.
#
# Usage: bash scripts/slurm/submit_glm_hmm_finetuning.sh [--cv-mode MODE ...]
#   --cv-mode  restrict to "pooled" or "session" (space-separated); omit for both.
#
# Examples:
#   bash scripts/slurm/submit_glm_hmm_finetuning.sh
#   bash scripts/slurm/submit_glm_hmm_finetuning.sh --cv-mode pooled

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${HOME}/bayesian-priors-in-pd-subtype/.env"

# Parse optional --cv-mode argument.
CV_MODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cv-mode)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                CV_MODE="${CV_MODE} $1"
                shift
            done
            CV_MODE="${CV_MODE# }"  # trim leading space
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

CV_MODE_ARG=""
if [ -n "${CV_MODE}" ]; then
    CV_MODE_ARG="--cv-mode ${CV_MODE}"
fi

# Query the number of work units dynamically from the script itself.
N=$(singularity exec \
        --bind ${CLUSTER_DATA_PATH}:/mnt/pd-data \
        --bind ${PROJECT_ROOT}:/src \
        ${SIF_PATH} \
        bash -c "export PYTHONPATH=/src && python3 /src/scripts/glm_hmm/model_finetuning.py ${CV_MODE_ARG} --list-jobs" \
    | head -1 | awk '{print $1}')

if [ -z "${N}" ] || [ "${N}" -eq 0 ]; then
    echo "No work units found (all _final.pkl already exist?). Nothing to submit."
    exit 0
fi

ARRAY_RANGE="0-$(( N - 1 ))"
echo "CV mode : ${CV_MODE:-all}"
echo "Units   : ${N}  (--array=${ARRAY_RANGE})"
echo

EXPORT_VARS="ALL"
if [ -n "${CV_MODE}" ]; then
    EXPORT_VARS="ALL,CV_MODE=${CV_MODE}"
fi

sbatch \
    --array="${ARRAY_RANGE}" \
    --export="${EXPORT_VARS}" \
    "${SCRIPT_DIR}/glm_hmm_finetuning.slurm"
