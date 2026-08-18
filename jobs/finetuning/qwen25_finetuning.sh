#!/usr/bin/env bash

#SBATCH --job-name="qwen25_finetuning"
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1                               # number of nodes
#SBATCH --ntasks=1                              # number of tasks
#SBATCH --cpus-per-task=2                       # number of CPUs per task
#SBATCH --gres=gpu:1                            # number of gpus per node max 1-4 for each node
#SBATCH --mem=100G	                            # RAM
#SBATCH --partition=boost_usr_prod              
#SBATCH --time=24:00:00                         # MAX 24 h, hh:mm:ss

set -Eeuo pipefail

readonly JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT="$(cd -- "${JOB_DIR}/../.." && pwd)"
readonly SCRIPT="finetuning_qwen"
readonly CONDA_ENV="${ROOT}/envs/dab"
readonly LOG_DIR="${ROOT}/logs/finetuning"
readonly PYTHON_SCRIPT="${ROOT}/scripts/finetuning/${SCRIPT}.py"
readonly DATASET_DIR="${ROOT}/data/dataset"
readonly BASE_MODEL_DIR="${ROOT}/models/Qwen/Qwen2.5-VL-7B-Instruct"
readonly OUTPUT_DIR="${ROOT}/models_checkpoints/Mappet/Mappet2.5-VL-7B-Instruct"
readonly TRAIN_LOG="${LOG_DIR}/qwen25_finetuning.log"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

mkdir -p -- "${LOG_DIR}"

# SBATCH paths cannot expand shell variables. Redirect after ROOT has been
# resolved so logs are always written inside the repository, independently of
# the directory from which sbatch is invoked.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    readonly SLURM_LOG_BASENAME="${SLURM_JOB_NAME:-qwen25_finetuning}_${SLURM_JOB_ID}"
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
[[ -d "${DATASET_DIR}" ]] || die "Missing dataset directory: ${DATASET_DIR}"
[[ -d "${BASE_MODEL_DIR}" ]] || die "Missing base model directory: ${BASE_MODEL_DIR}"

python "${PYTHON_SCRIPT}" \
    --dataset_dir "${DATASET_DIR}" \
    --base_model_name "${BASE_MODEL_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --image_dir "${DATASET_DIR}" \
    --batch_size 4 \
    --gradient_accumulation 8 \
    --dataloader_seed 117 \
    --epochs 1 \
    --log_file "${TRAIN_LOG}" \
    --lr 1e-4 \
    --max_tokens 4500 \
    --num_warmup_steps 25 \
    --temperature_for_training 1.0 \
    --image_max_size 3000 \
    --image_max_pixel_count 14000000 \
    --num_save_steps 50 \
    --use_peft \
    --restore_checkpoint
