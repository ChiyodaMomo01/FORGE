#!/usr/bin/env bash
# Distributed 8-GPU closed-loop training launcher for the 3D POM double-gyre benchmark.
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CLI_PROTOCOL=""
CLI_CONFIG=""
CLI_STAGE=""
HAS_DATASET_ROOT=0
HAS_OUTPUT_ROOT=0
HAS_HELP=0

ARGS=("$@")
idx=0
while [[ "${idx}" -lt "${#ARGS[@]}" ]]; do
  arg="${ARGS[${idx}]}"
  case "${arg}" in
    -h|--help)
      HAS_HELP=1
      ;;
    --protocol)
      idx=$((idx + 1))
      CLI_PROTOCOL="${ARGS[${idx}]:-}"
      ;;
    --protocol=*)
      CLI_PROTOCOL="${arg#--protocol=}"
      ;;
    --config)
      idx=$((idx + 1))
      CLI_CONFIG="${ARGS[${idx}]:-}"
      ;;
    --config=*)
      CLI_CONFIG="${arg#--config=}"
      ;;
    --stages)
      idx=$((idx + 1))
      CLI_STAGE="${ARGS[${idx}]:-}"
      ;;
    --stages=*)
      CLI_STAGE="${arg#--stages=}"
      ;;
    --dataset-root|--dataset-root=*)
      HAS_DATASET_ROOT=1
      ;;
    --output-root|--output-root=*)
      HAS_OUTPUT_ROOT=1
      ;;
  esac
  idx=$((idx + 1))
done

if [[ "${HAS_HELP}" -eq 1 ]]; then
  exec "${PYTHON_BIN}" -m forge_pom3d "$@"
fi

PROTOCOL="${CLI_PROTOCOL:-${FORGE_POM3D_PROTOCOL:-pure_4h}}"
STAGE="${CLI_STAGE:-${FORGE_POM3D_STAGE:-forge_main}}"

if [[ "${PROTOCOL}" != "pure_4h" ]]; then
  echo "Unsupported FORGE_POM3D_PROTOCOL=${PROTOCOL} (expected pure_4h)" >&2
  exit 2
fi
DEFAULT_CFG="${REPO_ROOT}/configs/pom3d_pure4h.json"
CONFIG_PATH="${CLI_CONFIG:-${FORGE_CONFIG:-${DEFAULT_CFG}}}"

if [[ "${HAS_DATASET_ROOT}" -eq 0 ]]; then
  : "${FORGE_POM3D_DATASET_ROOT:?Set FORGE_POM3D_DATASET_ROOT or pass --dataset-root to specify the 3D POM ensemble directory}"
fi
if [[ "${HAS_OUTPUT_ROOT}" -eq 0 ]]; then
  : "${FORGE_POM3D_OUTPUT_ROOT:?Set FORGE_POM3D_OUTPUT_ROOT or pass --output-root to specify the target run output directory}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

LAUNCH_ARGS=()
if [[ -z "${CLI_PROTOCOL}" ]]; then
  LAUNCH_ARGS+=(--protocol "${PROTOCOL}")
fi
if [[ -z "${CLI_CONFIG}" ]]; then
  LAUNCH_ARGS+=(--config "${CONFIG_PATH}")
fi
if [[ "${HAS_DATASET_ROOT}" -eq 0 ]]; then
  LAUNCH_ARGS+=(--dataset-root "${FORGE_POM3D_DATASET_ROOT}")
fi
if [[ "${HAS_OUTPUT_ROOT}" -eq 0 ]]; then
  LAUNCH_ARGS+=(--output-root "${FORGE_POM3D_OUTPUT_ROOT}")
fi
if [[ -z "${CLI_STAGE}" ]]; then
  LAUNCH_ARGS+=(--stages "${STAGE}")
fi

exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m forge_pom3d \
  "${LAUNCH_ARGS[@]}" \
  "$@"
