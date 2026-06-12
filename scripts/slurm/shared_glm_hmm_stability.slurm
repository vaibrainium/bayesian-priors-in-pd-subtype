#!/bin/bash

#SBATCH --job-name=shared_glm_hmm_stability
#SBATCH -p ckpt-all
#SBATCH -A walkerlab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue

#SBATCH -o /gscratch/walkerlab/vaibhav/logs/shared_glm_hmm_stability_%j.out
#SBATCH -e /gscratch/walkerlab/vaibhav/logs/shared_glm_hmm_stability_%j.err

# Precompute the K-selection metrics (ICL/BIC + bootstrap state-stability) for a shared-basis
# run to CSVs, so notebook 6.00 just loads them instead of refitting on a login node. The B
# bootstrap refits per K run in parallel (joblib n_jobs=-1, BLAS pinned), so more
# cpus-per-task = faster. Requires the merged all_subjects.pkl to exist.
#
# Submit via submit_shared_glm_hmm_stability.sh. Do NOT submit directly.

source "${HOME}/bayesian-priors-in-pd-subtype/.env"

module load singularity

CONFIG="${CONFIG:-ashwood_color_non_standardized}"
COLOR="${COLOR:-01}"
B="${B:-30}"
NITERS="${NITERS:-300}"

singularity exec \
    --bind ${CLUSTER_DATA_PATH}:/mnt/pd-data \
    --bind ${PROJECT_ROOT}:/src \
    ${SIF_PATH} \
    bash -c "
export PYTHONPATH=/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 &&
python3 /src/scripts/shared_glm_hmm/compute_stability.py --config $CONFIG --color-coding $COLOR --B $B --n-iters $NITERS
"
