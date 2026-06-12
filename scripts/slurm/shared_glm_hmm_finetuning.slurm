#!/bin/bash

#SBATCH --job-name=shared_glm_hmm_finetuning
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue

#SBATCH -o /gscratch/walkerlab/vaibhav/logs/shared_glm_hmm_finetuning_%j.out
#SBATCH -e /gscratch/walkerlab/vaibhav/logs/shared_glm_hmm_finetuning_%j.err

# Finetune the shared basis at the K chosen from notebook 6.00 (triangulation figure):
# per-session models + one shared pooled model -> all_subjects_final.pkl. CPU-bound
# (session_wise_fit uses joblib n_jobs=-1), so no GPU. BESTK is required and passed from
# submit_shared_glm_hmm_finetuning.sh via --export.
#
# Submit via submit_shared_glm_hmm_finetuning.sh. Do NOT submit directly.

source "${HOME}/bayesian-priors-in-pd-subtype/.env"

module load singularity

CONFIG="${CONFIG:-ashwood_color}"
COLOR="${COLOR:-01}"
: "${BESTK:?BESTK not set — choose K from notebook 6.00 and submit via submit_shared_glm_hmm_finetuning.sh}"

singularity exec \
    --bind ${CLUSTER_DATA_PATH}:/mnt/pd-data \
    --bind ${PROJECT_ROOT}:/src \
    ${SIF_PATH} \
    bash -c "
export PYTHONPATH=/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 &&
python3 /src/scripts/shared_glm_hmm/finetune_shared_basis.py --config $CONFIG --color-coding $COLOR --best-k $BESTK
"
