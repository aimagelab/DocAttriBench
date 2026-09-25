#!/bin/bash --login

#SBATCH --job-name=answerbox_inference
#SBATCH --output=logs/inference/answer_box/%x_%A_%a.out
#SBATCH --error=logs/inference/answer_box/%x_%A_%a.err

#SBATCH --array=0-2                             # One job per model
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --cpus-per-task=2                       # Number of CPUs per task
#SBATCH --gres=gpu:1                            # Number of GPUs per node
#SBATCH --mem=100G                              # RAM
#SBATCH --time=24:00:00                         # Maximum runtime (hh:mm:ss)

set -euo pipefail

# ================================
# REPOSITORY PATHS
# ================================

# Directory from which `sbatch` was executed
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

SCRIPT_PATH="${REPO_ROOT}/scripts/inference/answer_inference.py"
DATASET_DIR="${REPO_ROOT}/data"
MODEL_DIR="${REPO_ROOT}/models/Mappet"
OUTPUT_DIR="${REPO_ROOT}/output/inference"
LOG_DIR="${REPO_ROOT}/logs/inference/answer_box"

# ================================
# MODELS TO PROCESS
# ================================

# Each model is processed by a separate job in the Slurm array
BASE_MODELS=(
    "Mappet-7B"
    "Mappet-8B"
    "Mappet-I-8B"
)

MODEL="${BASE_MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "Job:   ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL}"

# ================================
# ENVIRONMENT
# ================================

# Activate the Conda environment
eval "$(conda shell.bash hook)"
conda activate dab

# ================================
# OUTPUT DIRECTORIES
# ================================

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# ================================
# RUN
# ================================

srun python "${SCRIPT_PATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --model_name "${MODEL_DIR}/${MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --log_file "${LOG_DIR}/answer_box_inference_${MODEL}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"