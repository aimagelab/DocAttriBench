#!/usr/bin/env bash
set -Eeuo pipefail

readonly ENV_NAME="dab"
readonly EXPECTED_PYTHON="3.10.19"
readonly EXPECTED_PACKAGE_COUNT="311"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly CONDA_SPEC="${SCRIPT_DIR}/conda-explicit-spec.txt"
readonly PIP_REQUIREMENTS="${SCRIPT_DIR}/requirements-pip.txt"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

command -v conda >/dev/null 2>&1 || die "conda is not available in PATH."
[[ -f "${CONDA_SPEC}" ]] || die "Missing file: ${CONDA_SPEC}"
[[ -f "${PIP_REQUIREMENTS}" ]] || die "Missing file: ${PIP_REQUIREMENTS}"

if conda env list | awk -v name="${ENV_NAME}" '$1 == name {found = 1} END {exit !found}'; then
    die "The '${ENV_NAME}' environment already exists; it was not modified."
fi

printf 'Creating the %s environment from the Conda package specification...\n' "${ENV_NAME}"
conda create --yes --name "${ENV_NAME}" --file "${CONDA_SPEC}"

printf 'Installing pinned pip packages...\n'
conda run --name "${ENV_NAME}" python -m pip install \
    --disable-pip-version-check \
    --no-deps \
    --requirement "${PIP_REQUIREMENTS}"

actual_python="$(conda run --name "${ENV_NAME}" python -c 'import platform; print(platform.python_version())')"
actual_package_count="$(conda list --name "${ENV_NAME}" | awk 'NR > 3 && NF >= 3 {count++} END {print count + 0}')"

[[ "${actual_python}" == "${EXPECTED_PYTHON}" ]] || \
    die "Unexpected Python version: ${actual_python} (expected: ${EXPECTED_PYTHON})."
[[ "${actual_package_count}" == "${EXPECTED_PACKAGE_COUNT}" ]] || \
    die "Unexpected package count: ${actual_package_count} (expected: ${EXPECTED_PACKAGE_COUNT})."

printf '\nEnvironment %s created and verified.\n' "${ENV_NAME}"
printf 'To activate it, run: conda activate %s\n' "${ENV_NAME}"
