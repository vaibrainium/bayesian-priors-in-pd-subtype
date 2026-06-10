#!/bin/bash

#SBATCH --job-name=glm_hmm_fitting
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue

#SBATCH -o /gscratch/walkerlab/vaibhav/logs/glm_hmm_%A_%a.out
#SBATCH -e /gscratch/walkerlab/vaibhav/logs/glm_hmm_%A_%a.err

# GLM-HMM fitting is CPU-bound (joblib n_jobs=-1 in global_fit / CV), so no GPU is
# requested and --nv is omitted. One array task = one (config, group) work unit;
# both CV modes share the global fit.
#
# Submit via submit_glm_hmm.sh — it computes --array dynamically so the range
# stays in sync with the config list. Do NOT submit this file directly with sbatch.

source "${HOME}/bayesian-priors-in-pd-subtype/.env"

module load singularity

# CONFIGS is passed from submit_glm_hmm.sh via --export; the default below is a
# fallback only. Order is load-bearing: --job_id indexes into (configs x groups).
CONFIGS="${CONFIGS:-normalized_stimulus standardized_stimulus standardized_stimulus_with_color normalized_stimulus_with_color}"

singularity exec \
    --bind ${CLUSTER_DATA_PATH}:/mnt/pd-data \
    --bind ${PROJECT_ROOT}:/src \
    ${SIF_PATH} \
    bash -c "
export PYTHONPATH=/src &&
python3 /src/scripts/glm_hmm/model_fitting_with_slurm.py --config $CONFIGS --job_id \$SLURM_ARRAY_TASK_ID
"
