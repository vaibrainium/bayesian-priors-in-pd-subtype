#!/usr/bin/env bash
# Submit glm_hmm_fitting.slurm as a Slurm array job with a dynamically computed
# --array range, so the range can never drift out of sync with the config list.
#
# Usage: bash scripts/slurm/submit_glm_hmm.sh [CONFIG ...]
#   CONFIG  one or more config aliases from glm_hmm.experiments.CONFIGS;
#           omit to use the default set below.
#
# Examples:
#   bash scripts/slurm/submit_glm_hmm.sh
#   bash scripts/slurm/submit_glm_hmm.sh normalized_stimulus standardized_stimulus

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_CONFIGS="normalized_stimulus standardized_stimulus standardized_stimulus_with_color normalized_stimulus_with_color"
CONFIGS="${*:-${DEFAULT_CONFIGS}}"

# Subject groups are fixed by get_group_session_ids in data_preparation.py.
# One work unit per (config, group) pair — mirrors the jobs list in model_fitting_with_slurm.py.
GROUPS="asmHC Tremor_OFF Tremor_ON Brady_OFF Brady_ON"
N_CONFIGS=$(echo ${CONFIGS} | wc -w)
N_GROUPS=$(echo ${GROUPS}   | wc -w)
N=$(( N_CONFIGS * N_GROUPS ))

ARRAY_RANGE="0-$(( N - 1 ))"
echo "Configs : ${CONFIGS}"
echo "Groups  : ${GROUPS}"
echo "Units   : ${N}  (--array=${ARRAY_RANGE})"
echo

sbatch \
    --array="${ARRAY_RANGE}" \
    --export=ALL,CONFIGS="${CONFIGS}" \
    "${SCRIPT_DIR}/glm_hmm_fitting.slurm"
