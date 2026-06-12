#!/usr/bin/env bash
# Submit the shared-basis finetuning at the K chosen from notebook 6.00.
# A single work unit (one pooled basis at one K), so no array — just exports BESTK.
#
# Usage: bash scripts/slurm/submit_shared_glm_hmm_finetuning.sh <BESTK>
#   or:  BESTK=<K> bash scripts/slurm/submit_shared_glm_hmm_finetuning.sh
#   CONFIG / COLOR default to ashwood_color / 01 (must match the fitted run).
#
# Example:
#   bash scripts/slurm/submit_shared_glm_hmm_finetuning.sh 4
#   CONFIG=ashwood_color COLOR=pm1 bash scripts/slurm/submit_shared_glm_hmm_finetuning.sh 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${CONFIG:-ashwood_color}"
COLOR="${COLOR:-01}"
BESTK="${1:-${BESTK:-}}"

if [ -z "${BESTK}" ]; then
    echo "error: provide K — 'submit_shared_glm_hmm_finetuning.sh <K>' or 'BESTK=<K> ...'" >&2
    exit 1
fi

echo "Config  : ${CONFIG}  (color=${COLOR})"
echo "Finetune: K=${BESTK}"
echo

sbatch \
    --export=ALL,CONFIG="${CONFIG}",COLOR="${COLOR}",BESTK="${BESTK}" \
    "${SCRIPT_DIR}/shared_glm_hmm_finetuning.slurm"
