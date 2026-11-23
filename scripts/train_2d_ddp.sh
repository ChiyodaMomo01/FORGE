#!/usr/bin/env bash
# Distributed 8-GPU closed-loop training launcher for 2D hybrid PDE benchmarks.
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CLI_CASE=""
CLI_METHOD=""
CLI_CONFIG=""
CLI_DEVICE=""
HAS_OUTPUT=0
HAS_PAIRS=0
HAS_ORIGINAL=0
HAS_HELP=0

ARGS=("$@")
idx=0
while [[ "${idx}" -lt "${#ARGS[@]}" ]]; do
  arg="${ARGS[${idx}]}"
  case "${arg}" in
    -h|--help)
      HAS_HELP=1
      ;;
    --case)
      idx=$((idx + 1))
      CLI_CASE="${ARGS[${idx}]:-}"
      ;;
    --case=*)
      CLI_CASE="${arg#--case=}"
      ;;
    --method)
      idx=$((idx + 1))
      CLI_METHOD="${ARGS[${idx}]:-}"
      ;;
    --method=*)
      CLI_METHOD="${arg#--method=}"
      ;;
    --config)
      idx=$((idx + 1))
      CLI_CONFIG="${ARGS[${idx}]:-}"
      ;;
    --config=*)
      CLI_CONFIG="${arg#--config=}"
      ;;
    --device)
      idx=$((idx + 1))
      CLI_DEVICE="${ARGS[${idx}]:-}"
      ;;
    --device=*)
      CLI_DEVICE="${arg#--device=}"
      ;;
    --output|--output=*)
      HAS_OUTPUT=1
      ;;
    --pairs|--pairs=*)
      HAS_PAIRS=1
      ;;
    --original|--original=*)
      HAS_ORIGINAL=1
      ;;
  esac
  idx=$((idx + 1))
done

if [[ "${HAS_HELP}" -eq 1 ]]; then
  exec "${PYTHON_BIN}" -m forge.run "$@"
fi

CASE="${CLI_CASE:-${FORGE_CASE:-cahn_hilliard}}"
METHOD="${CLI_METHOD:-${FORGE_METHOD:-main}}"
DEVICE="${CLI_DEVICE:-${FORGE_DEVICE:-cuda}}"
CONFIG_PATH="${CLI_CONFIG:-${FORGE_CONFIG:-${REPO_ROOT}/configs/${CASE}.json}}"

if [[ "${HAS_OUTPUT}" -eq 0 ]]; then
  : "${FORGE_OUTPUT_ROOT:?Set FORGE_OUTPUT_ROOT or pass --output to specify the experiment output directory}"
fi
if [[ "${METHOD}" == "prepare" || "${METHOD}" == "physics_start" || "${METHOD}" == "pretrain" ]]; then
  if [[ "${HAS_PAIRS}" -eq 0 ]]; then
    : "${FORGE_PAIRS_ROOT:?Set FORGE_PAIRS_ROOT or pass --pairs to specify the restricted G-pairs directory}"
  fi
fi
if [[ "${METHOD}" == "prepare" ]]; then
  if [[ "${HAS_ORIGINAL}" -eq 0 ]]; then
    : "${FORGE_ORIGINAL_ROOT:?Set FORGE_ORIGINAL_ROOT or pass --original to specify the reference trajectories directory}"
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

LAUNCH_ARGS=()
if [[ -z "${CLI_CONFIG}" ]]; then
  LAUNCH_ARGS+=(--config "${CONFIG_PATH}")
fi
if [[ -z "${CLI_CASE}" ]]; then
  LAUNCH_ARGS+=(--case "${CASE}")
fi
if [[ -z "${CLI_METHOD}" ]]; then
  LAUNCH_ARGS+=(--method "${METHOD}")
fi
if [[ -z "${CLI_DEVICE}" ]]; then
  LAUNCH_ARGS+=(--device "${DEVICE}")
fi
if [[ -n "${FORGE_REFERENCE_ROOT:-}" ]]; then
  HAS_REF=0
  for arg in "$@"; do
    if [[ "${arg}" == "--reference" || "${arg}" == --reference=* ]]; then
      HAS_REF=1
      break
    fi
  done
  if [[ "${HAS_REF}" -eq 0 ]]; then
    LAUNCH_ARGS+=(--reference "${FORGE_REFERENCE_ROOT}")
  fi
fi

if [[ "${METHOD}" == "prepare" || "${METHOD}" == "physics_start" || "${METHOD}" == "freeze_common" || "${METHOD}" == "validate_gate" || "${NPROC_PER_NODE}" -le 1 ]]; then
  export FORGE_DDP_WORLD_SIZE=1
  exec "${PYTHON_BIN}" -m forge.run \
    "${LAUNCH_ARGS[@]}" \
    "$@"
fi

export FORGE_DDP_WORLD_SIZE="${NPROC_PER_NODE}"
exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m forge.run \
  "${LAUNCH_ARGS[@]}" \
  "$@"
