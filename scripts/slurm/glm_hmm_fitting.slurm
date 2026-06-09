#!/bin/bash

#SBATCH --job-name=glm_hmm_fitting
#SBATCH --array=0-19
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
# both CV modes share the global fit. --array range must equal (#configs x #groups).
# Here: 4 configs x 5 groups = 20  ->  --array=0-19.
# Confirm the count with:
#   python3 scripts/glm_hmm/model_fitting_with_slurm.py --config $CONFIGS --list-jobs

source "${HOME}/bayesian-priors-in-pd-subtype/.env"

module load singularity

# Order must stay fixed: --job_id indexes into (configs x groups) in this order.
CONFIGS="normalized_stimulus standardized_stimulus standardized_stimulus_with_color normalized_stimulus_with_color"

singularity exec \
    --bind ${CLUSTER_DATA_PATH}:/mnt/pd-data \
    --bind ${PROJECT_ROOT}:/src \
    ${SIF_PATH} \
    bash -c "
export PYTHONPATH=/src &&
python3 /src/scripts/glm_hmm/model_fitting_with_slurm.py --config $CONFIGS --job_id \$SLURM_ARRAY_TASK_ID
"
