#!/usr/bin/env bash

#SBATCH --job-name="VISA_model_merge"
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1                               # number of nodes
#SBATCH --ntasks=1                              # number of tasks
#SBATCH --cpus-per-task=2                       # number of CPUs per task
#SBATCH --gres=gpu:1                            # number of gpus per node max 1-4 for each node
#SBATCH --mem=100G	                            # RAM
#SBATCH --partition=boost_usr_prod              
#SBATCH --time=4:00:00                         # MAX 24 h, hh:mm:ss

set -Eeuo pipefail

readonly JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT="$(cd -- "${JOB_DIR}/../.." && pwd)"
readonly SCRIPT="model_merge"
readonly CONDA_ENV="${ROOT}/envs/dab"
readonly LOG_DIR="${ROOT}/logs/model_merge"
readonly PYTHON_SCRIPT="${ROOT}/scripts/model_merge/${SCRIPT}.py"
readonly BASE_MODEL_DIR="${ROOT}/models/Qwen/Qwen2-VL-7B-Instruct"
readonly CHECKPOINT_DIR="${ROOT}/models/MrLight/visa-7B-single-fulldata"
readonly OUTPUT_DIR="${ROOT}/models/MrLight/visa-7B-single-fulldata-merged"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

mkdir -p -- "${LOG_DIR}"

# SBATCH paths cannot expand shell variables. Redirect after ROOT has been
# resolved so logs are always written inside the repository, independently of
# the directory from which sbatch is invoked.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    readonly SLURM_LOG_BASENAME="${SLURM_JOB_NAME:-VISA_model_merge}_${SLURM_JOB_ID}"
    exec >"${LOG_DIR}/${SLURM_LOG_BASENAME}.out" \
         2>"${LOG_DIR}/${SLURM_LOG_BASENAME}.err"
fi

command -v conda >/dev/null 2>&1 || die "conda is not available in PATH."

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '$1 == "dab" {found = 1} END {exit !found}'; then
    conda activate dab
elif [[ -d "${CONDA_ENV}/conda-meta" ]]; then
    conda activate "${CONDA_ENV}"
else
    die "Conda environment 'dab' was not found, and ${CONDA_ENV} is not a valid Conda environment."
fi

export PYTHONNOUSERSITE=1

[[ -f "${PYTHON_SCRIPT}" ]] || die "Missing training script: ${PYTHON_SCRIPT}"
[[ -d "${BASE_MODEL_DIR}" ]] || die "Missing base model directory: ${BASE_MODEL_DIR}"
[[ -d "${CHECKPOINT_DIR}" ]] || die "Missing checkpoint directory: ${CHECKPOINT_DIR}"

python "${PYTHON_SCRIPT}" \
    --base_model "${BASE_MODEL_DIR}" \
    --adapter_path "${CHECKPOINT_DIR}"\
    --output_model_path "${OUTPUT_DIR}" \
|| echo "Python script failed..."
