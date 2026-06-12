#!/bin/bash

#SBATCH --job-name=shared_glm_hmm_merge
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --requeue

#SBATCH -o /gscratch/walkerlab/vaibhav/logs/shared_glm_hmm_merge_%j.out
#SBATCH -e /gscratch/walkerlab/vaibhav/logs/shared_glm_hmm_merge_%j.err

# Stack the per-K shards from shared_glm_hmm_fitting.slurm into one all_subjects.pkl
# (same schema model_fitting.py pooled mode produces). Cheap: just concatenates arrays.
# Submitted with --dependency=afterok by submit_shared_glm_hmm.sh, so it runs only once
# every K shard has completed. Do NOT submit directly.

source "${HOME}/bayesian-priors-in-pd-subtype/.env"

module load singularity

CONFIG="${CONFIG:-ashwood_color_non_standardized}"
COLOR="${COLOR:-01}"

singularity exec \
    --bind ${CLUSTER_DATA_PATH}:/mnt/pd-data \
    --bind ${PROJECT_ROOT}:/src \
    ${SIF_PATH} \
    bash -c "
export PYTHONPATH=/src &&
python3 /src/scripts/shared_glm_hmm/fit_shared_basis_slurm.py --config $CONFIG --color-coding $COLOR --merge
"
