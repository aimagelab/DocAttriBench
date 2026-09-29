#!/bin/bash --login

#SBATCH --job-name=boxFromAnswer_inference
#SBATCH --output=logs/inference/boxFromAnswer/%x_%A_%a.out
#SBATCH --error=logs/inference/boxFromAnswer/%x_%A_%a.err

#SBATCH --array=0-12                            # One job per model
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
MODEL_DIR="${REPO_ROOT}/models"
OUTPUT_DIR="${REPO_ROOT}/output/inference"
LOG_DIR="${REPO_ROOT}/logs/inference/boxFromAnswer"

# ================================
# MODELS TO PROCESS
# ================================

# Each model is processed by a separate job in the SLURM array.
# Entries correspond to local paths under /models and preserve
# the Hugging Face organization/model directory structure.
MODELS=(
    "MrLight/visa-7B-single-fulldata-merged"
    "OpenGVLab/InternVL2_5-2B"
    "OpenGVLab/InternVL2_5-8B"
    "OpenGVLab/InternVL2_5-38B"
    "OpenGVLab/InternVL3-2B-Instruct"
    "OpenGVLab/InternVL3-8B-Instruct"
    "OpenGVLab/InternVL3-38B-Instruct"
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "Qwen/Qwen2.5-VL-32B-Instruct"
    "Qwen/Qwen3-VL-2B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
    "Qwen/Qwen3-VL-32B-Instruct"
)

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
MODEL_PATH="${MODEL_DIR}/${MODEL}"

# Safe model name for log files
MODEL_TAG="${MODEL//\//_}"

echo "========================================"
echo "Job:        ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Array ID:   ${SLURM_ARRAY_TASK_ID}"
echo "Model:      ${MODEL}"
echo "Model path: ${MODEL_PATH}"
echo "Task:  boxFromAnswer"
echo "========================================"

# ================================
# ENVIRONMENT
# ================================

# Activate the Conda environment
eval "$(conda shell.bash hook)"
conda activate dab

# ================================
# CHECKS
# ================================

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "ERROR: inference script not found: ${SCRIPT_PATH}"
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: model directory not found: ${MODEL_PATH}"
    exit 1
fi

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
    --model_name "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --task "boxFromAnswer" \
    --log_file "${LOG_DIR}/boxFromAnswer_inference_${MODEL_TAG}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"