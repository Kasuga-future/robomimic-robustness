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
    ENV_NAMES=(Lift)
    ;;
  visual)
    ENV_NAMES=(Lift LiftVisualOOD)
    ;;
  object)
    ENV_NAMES=(Lift LiftObjectPerturb)
    ;;
  color)
    ENV_NAMES=(Lift LiftColorPerturb)
    ;;
  camera)
    ENV_NAMES=(Lift LiftCameraPerturb)
    ;;
  task|dynamics)
    ENV_NAMES=(Lift LiftTaskDynamicsHard)
    ;;
  ood)
    ENV_NAMES=(Lift LiftVisualOOD LiftTaskDynamicsHard)
    ;;
  legacy)
    ENV_NAMES=(Lift LiftObjectPerturb LiftColorPerturb LiftCameraPerturb)
    ;;
  all)
    ENV_NAMES=(Lift LiftObjectPerturb LiftColorPerturb LiftCameraPerturb LiftVisualOOD LiftTaskDynamicsHard)
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
