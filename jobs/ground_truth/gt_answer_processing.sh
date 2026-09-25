#!/usr/bin/env bash

#SBATCH --job-name=gt_answer_processing
#SBATCH --output=logs/gt_processing/%x_%j.out
#SBATCH --error=logs/gt_processing/%x_%j.err

#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=2                 # Number of CPUs per task
#SBATCH --gres=gpu:2                      # Number of GPUs per node
#SBATCH --mem=100G                        # RAM
#SBATCH --time=02:00:00                   # Maximum runtime (hh:mm:ss)

set -euo pipefail

# Directory from which `sbatch` was executed
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"

SCRIPT_PATH="${REPO_ROOT}/scripts/ground_truth/gt_answer_processing.py"
DATASET_DIR="${REPO_ROOT}/data"
MODEL_PATH="${REPO_ROOT}/models/Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
LOG_DIR="${REPO_ROOT}/logs/gt_processing"

# NOTE: this directory must already exist for --output/--error
mkdir -p "${LOG_DIR}"

# Activate the Conda environment
eval "$(conda shell.bash hook)"
conda activate dab

# Print the resolved paths for reproducibility/debugging
echo "Repository root: ${REPO_ROOT}"
echo "Python script:   ${SCRIPT_PATH}"
echo "Dataset:         ${DATASET_DIR}"
echo "Model:           ${MODEL_PATH}"
echo "Log directory:   ${LOG_DIR}"

# Run the ground-truth answer processing script
srun python "${SCRIPT_PATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --model_name "${MODEL_PATH}" \
    --log_file "${LOG_DIR}/gt_answer_processing.log"