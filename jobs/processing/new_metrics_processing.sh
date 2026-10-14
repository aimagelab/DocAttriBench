#!/bin/bash --login

#SBATCH --job-name=metrics_processing
#SBATCH --output=logs/processing/%x_%j.out
#SBATCH --error=logs/processing/%x_%j.err

#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks=1                              # Number of tasks
#SBATCH --cpus-per-task=2                       # Number of CPUs per task
#SBATCH --mem=100G                              # RAM
#SBATCH --time=24:00:00                         # Maximum runtime (hh:mm:ss)

set -euo pipefail

# ================================
# REPOSITORY PATHS
# ================================

# Resolve the repository root from this script location:
# jobs/processing/metrics_processing.sh -> repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SCRIPT_PATH="${REPO_ROOT}/scripts/processing/metrics_processing.py"
DATASET_DIR="${REPO_ROOT}/data"
INFERENCE_DIR="${REPO_ROOT}/output/inference"
METRICS_DIR="${REPO_ROOT}/output/metrics"
LOG_DIR="${REPO_ROOT}/logs/processing"

# ================================
# ENVIRONMENT
# ================================

# Activate the Conda environment
eval "$(conda shell.bash hook)"
conda activate dab

# ================================
# OUTPUT DIRECTORIES
# ================================

mkdir -p "${LOG_DIR}"

# ================================
# RUN
# ================================

echo "Repository root: ${REPO_ROOT}"
echo "Dataset:         ${DATASET_DIR}"
echo "Inference dir:   ${INFERENCE_DIR}"
echo "Metrics dir:     ${METRICS_DIR}"

srun python "${SCRIPT_PATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --inference_dir "${INFERENCE_DIR}" \
    --metrics_dir "${METRICS_DIR}" \
    --log_file "${LOG_DIR}/metrics_processing.log"