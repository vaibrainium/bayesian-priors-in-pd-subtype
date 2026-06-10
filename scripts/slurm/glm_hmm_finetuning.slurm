#!/bin/bash

#SBATCH --job-name=glm_hmm_finetuning
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue

#SBATCH -o /gscratch/walkerlab/vaibhav/logs/glm_hmm_finetuning_%A_%a.out
#SBATCH -e /gscratch/walkerlab/vaibhav/logs/glm_hmm_finetuning_%A_%a.err

# GLM-HMM finetuning is CPU-bound (session_wise_fit uses joblib n_jobs=-1), so no
# GPU is requested and --nv is omitted. One array task = one (model_dir, group)
# work unit. CV_MODE (optional) is passed from submit_glm_hmm_finetuning.sh via
# --export to restrict which CV types are processed.
#
# Submit via submit_glm_hmm_finetuning.sh — it computes --array dynamically from
# --list-jobs so the range stays in sync with what's on disk. Do NOT submit this
# file directly with sbatch.

source "${HOME}/bayesian-priors-in-pd-subtype/.env"

module load singularity

CV_MODE_ARG=""
if [ -n "${CV_MODE:-}" ]; then
    CV_MODE_ARG="--cv-mode ${CV_MODE}"
fi

singularity exec \
    --bind ${CLUSTER_DATA_PATH}:/mnt/pd-data \
    --bind ${PROJECT_ROOT}:/src \
    ${SIF_PATH} \
    bash -c "
export PYTHONPATH=/src &&
python3 /src/scripts/glm_hmm/model_finetuning.py ${CV_MODE_ARG} --job_id \$SLURM_ARRAY_TASK_ID
"
