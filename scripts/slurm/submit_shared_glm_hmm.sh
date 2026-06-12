#!/usr/bin/env bash
# Submit the shared-basis pipeline (K-sharded fit array -> merge -> ICL/BIC + bootstrap
# stability) for one or more configs. Each step runs only after the previous succeeds
# (afterok), so each config self-sequences; multiple configs run in parallel. The --array
# range is computed dynamically from --list-jobs so it always matches the config's state range.
#
# Usage: bash scripts/slurm/submit_shared_glm_hmm.sh [CONFIG ...] [options]
#   CONFIG ...      one or more config names (positional or repeated --config NAME) from
#                   src.shared_glm_hmm.experiments.CONFIGS. Omit for the CONFIG env var or
#                   the default (XY__tgt). Use --all for the whole registry.
#   --all           submit every config in the registry (already-finished ones no-op).
#   --color         color coding: 01 (recommended) or pm1 (default 01).
#   --no-stability  submit only fit + merge per config (skip the stability precompute).
#   --B             bootstrap resamples per K for stability (default 30).
#   --n-iters       EM iters per bootstrap refit for stability (default 300).
#   --list-jobs     print the K work units for each config and exit WITHOUT submitting.
#
# Examples:
#   bash scripts/slurm/submit_shared_glm_hmm.sh XY__coh XYZ__tgt X__tgt
#   bash scripts/slurm/submit_shared_glm_hmm.sh --all
#   bash scripts/slurm/submit_shared_glm_hmm.sh XY__coh --no-stability --color pm1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${HOME}/bayesian-priors-in-pd-subtype/.env"

COLOR="${COLOR:-01}"
B="${B:-30}"
NITERS="${NITERS:-300}"
LIST_ONLY=0
STABILITY=1
ALL=0
CONFIGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)               CONFIGS+=("$2"); shift 2 ;;
        --all)                  ALL=1; shift ;;
        --color|--color-coding)  COLOR="$2"; shift 2 ;;
        --no-stability)         STABILITY=0; shift ;;
        --B)                    B="$2"; shift 2 ;;
        --n-iters)              NITERS="$2"; shift 2 ;;
        --list-jobs)            LIST_ONLY=1; shift ;;
        -h|--help)              sed -n '2,21p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)                     echo "unknown argument: $1" >&2; exit 1 ;;
        *)                      CONFIGS+=("$1"); shift ;;   # positional config name
    esac
done

in_container() {  # run a python snippet inside the singularity image
    singularity exec \
        --bind ${CLUSTER_DATA_PATH}:/mnt/pd-data \
        --bind ${PROJECT_ROOT}:/src \
        ${SIF_PATH} \
        bash -c "export PYTHONPATH=/src && $1"
}

if [ "${ALL}" -eq 1 ]; then
    read -r -a CONFIGS <<< "$(in_container "python3 -c 'from src.shared_glm_hmm.experiments import CONFIGS; print(\" \".join(CONFIGS))'")"
fi
if [ "${#CONFIGS[@]}" -eq 0 ]; then
    CONFIGS=("${CONFIG:-XY__tgt}")
fi

submit_chain() {
    local cfg="$1"
    local list n range fit merge stab last
    list=$(in_container "python3 /src/scripts/shared_glm_hmm/fit_shared_basis_slurm.py --config ${cfg} --color-coding ${COLOR} --list-jobs")

    if [ "${LIST_ONLY}" -eq 1 ]; then
        echo "── ${cfg} (color=${COLOR}) ──"; echo "${list}"; echo; return
    fi

    n=$(echo "${list}" | head -1 | awk '{print $1}')
    if [ -z "${n}" ] || [ "${n}" -eq 0 ]; then
        echo "${cfg}: no work units, skipping." >&2; return
    fi
    range="0-$(( n - 1 ))"

    fit=$(sbatch --parsable --array="${range}" \
        --export=ALL,CONFIG="${cfg}",COLOR="${COLOR}" \
        "${SCRIPT_DIR}/shared_glm_hmm_fitting.slurm")
    merge=$(sbatch --parsable --dependency=afterok:"${fit}" \
        --export=ALL,CONFIG="${cfg}",COLOR="${COLOR}" \
        "${SCRIPT_DIR}/shared_glm_hmm_merge.slurm")
    last="${merge}"
    stab=""
    if [ "${STABILITY}" -eq 1 ]; then
        stab=$(sbatch --parsable --dependency=afterok:"${merge}" \
            --export=ALL,CONFIG="${cfg}",COLOR="${COLOR}",B="${B}",NITERS="${NITERS}" \
            "${SCRIPT_DIR}/shared_glm_hmm_stability.slurm")
        last="${stab}"
    fi
    printf "  %-16s fit=%s  merge=%s%s  (array 0-%s)\n" \
        "${cfg}" "${fit}" "${merge}" "${stab:+  stability=$stab}" "$(( n - 1 ))"
}

echo "configs (${#CONFIGS[@]}): ${CONFIGS[*]}   color=${COLOR}$( [ "${STABILITY}" -eq 1 ] && echo "  stability(B=${B},n_iters=${NITERS})" )"
echo
for cfg in "${CONFIGS[@]}"; do
    submit_chain "${cfg}"
done

if [ "${LIST_ONLY}" -eq 0 ]; then
    echo
    echo "Submitted. When a config's chain finishes: open notebooks/6.00 (CV + ICL/BIC +"
    echo "stability CSVs precomputed) to choose K, then"
    echo "  bash ${SCRIPT_DIR}/submit_shared_glm_hmm_finetuning.sh <K> --config <name> --color ${COLOR}"
fi
