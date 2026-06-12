#!/bin/bash

#SBATCH --job-name=shared_glm_hmm_fitting
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue

#SBATCH -o /gscratch/walkerlab/vaibhav/logs/shared_glm_hmm_%A_%a.out
#SBATCH -e /gscratch/walkerlab/vaibhav/logs/shared_glm_hmm_%A_%a.err

# Shared-basis GLM-HMM fit: one pooled HC+PD state basis instead of five per-group models.
# The single pooled group collapses the old (config, group) array axis, so we shard by
# STATE COUNT K instead: one array task = one K (its global inits + pooled CV). joblib
# (n_jobs=-1) saturates this node's cores across that K's inits/folds, so more cpus-per-task
# = more within-K parallelism. CPU-bound, so no GPU / --nv. After the array, the shards are
# stacked by shared_glm_hmm_merge.slurm.
#
# Submit via submit_shared_glm_hmm.sh — it computes --array from --list-jobs and chains the
# merge. Do NOT submit this file directly with sbatch.

source "${HOME}/bayesian-priors-in-pd-subtype/.env"

module load singularity

# Passed from submit_shared_glm_hmm.sh via --export; defaults are a fallback only.
CONFIG="${CONFIG:-ashwood_color_non_standardized}"
COLOR="${COLOR:-01}"

singularity exec \
    --bind ${CLUSTER_DATA_PATH}:/mnt/pd-data \
    --bind ${PROJECT_ROOT}:/src \
    ${SIF_PATH} \
    bash -c "
export PYTHONPATH=/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 &&
python3 /src/scripts/shared_glm_hmm/fit_shared_basis_slurm.py --config $CONFIG --color-coding $COLOR --job_id \$SLURM_ARRAY_TASK_ID
"
