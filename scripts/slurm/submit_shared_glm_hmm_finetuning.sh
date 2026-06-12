#!/usr/bin/env bash
# Submit the shared-basis finetuning at the K chosen from notebook 6.00.
# A single work unit (one pooled basis at one K), so no array — just exports BESTK.
#
# Usage: bash scripts/slurm/submit_shared_glm_hmm_finetuning.sh <K> [--config NAME] [--color 01|pm1]
#   <K>        state count chosen from notebook 6.00 (or --best-k K, or BESTK=K env).
#   --config   default ashwood_color_non_standardized; overrides CONFIG env. Must match the fit.
#   --color    01 (default) or pm1. Must match the fit.
#
# Examples:
#   bash scripts/slurm/submit_shared_glm_hmm_finetuning.sh 4 --config ashwood_color_non_standardized
#   BESTK=4 bash scripts/slurm/submit_shared_glm_hmm_finetuning.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${CONFIG:-ashwood_color_non_standardized}"
COLOR="${COLOR:-01}"
BESTK="${BESTK:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)               CONFIG="$2"; shift 2 ;;
        --color|--color-coding)  COLOR="$2"; shift 2 ;;
        --best-k)               BESTK="$2"; shift 2 ;;
        -h|--help)              sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*) echo "unknown argument: $1" >&2; exit 1 ;;
        *)  BESTK="$1"; shift ;;   # positional K
    esac
done

if [ -z "${BESTK}" ]; then
    echo "error: provide K — 'submit_shared_glm_hmm_finetuning.sh <K>' or '--best-k K' or 'BESTK=K ...'" >&2
    exit 1
fi

echo "Config  : ${CONFIG}  (color=${COLOR})"
echo "Finetune: K=${BESTK}"
echo

sbatch \
    --export=ALL,CONFIG="${CONFIG}",COLOR="${COLOR}",BESTK="${BESTK}" \
    "${SCRIPT_DIR}/shared_glm_hmm_finetuning.slurm"
