#!/usr/bin/env bash
set -euo pipefail

ROBOMIMIC_ROOT="${ROBOMIMIC_ROOT:-/mnt/users/hejunhao-20251119/work/robomimic}"
AGENT="${1:?usage: eval_one_model.sh /path/to/model.pth [n_rollouts] [suite_or_env]}"
N_ROLLOUTS="${2:-50}"
HORIZON="${HORIZON:-400}"
SEED="${SEED:-1}"
SUITE_OR_ENV="${3:-all}"
VIDEO_DIR="${4:-}"

case "${SUITE_OR_ENV}" in
  clean)
    ENV_NAMES=(NutAssemblySquare)
    ;;
  object)
    ENV_NAMES=(NutAssemblySquare NutAssemblySquareObjectPerturb)
    ;;
  color)
    ENV_NAMES=(NutAssemblySquare NutAssemblySquareColorPerturb)
    ;;
  camera)
    ENV_NAMES=(NutAssemblySquare NutAssemblySquareCameraPerturb)
    ;;
  visual)
    ENV_NAMES=(NutAssemblySquare NutAssemblySquareVisualOOD)
    ;;
  task|dynamics)
    ENV_NAMES=(NutAssemblySquare NutAssemblySquareTaskDynamicsHard)
    ;;
  ood)
    ENV_NAMES=(NutAssemblySquare NutAssemblySquareVisualOOD NutAssemblySquareTaskDynamicsHard)
    ;;
  legacy)
    ENV_NAMES=(NutAssemblySquare NutAssemblySquareObjectPerturb NutAssemblySquareColorPerturb NutAssemblySquareCameraPerturb)
    ;;
  all)
    ENV_NAMES=(NutAssemblySquare NutAssemblySquareObjectPerturb NutAssemblySquareColorPerturb NutAssemblySquareCameraPerturb NutAssemblySquareVisualOOD NutAssemblySquareTaskDynamicsHard)
    ;;
  *)
    ENV_NAMES=("${SUITE_OR_ENV}")
    ;;
esac

for ENV_NAME in "${ENV_NAMES[@]}"; do
  echo "===== ${ENV_NAME} ====="
  EXTRA_ARGS=()
  if [[ -n "${VIDEO_DIR}" ]]; then
    mkdir -p "${VIDEO_DIR}"
    EXTRA_ARGS+=(--video_path "${VIDEO_DIR}/${ENV_NAME}.mp4")
  fi
  python "${ROBOMIMIC_ROOT}/robomimic/scripts/run_trained_agent.py" \
    --agent "${AGENT}" \
    --env "${ENV_NAME}" \
    --n_rollouts "${N_ROLLOUTS}" \
    --horizon "${HORIZON}" \
    --seed "${SEED}" \
    --camera_names agentview robot0_eye_in_hand \
    "${EXTRA_ARGS[@]}"
done
