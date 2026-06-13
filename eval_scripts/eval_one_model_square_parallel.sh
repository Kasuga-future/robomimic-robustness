#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Parallel evaluation for Square environments
#
# Usage:
#   bash eval_one_model_square_parallel.sh \
#       /path/to/model.pth \
#       50 \
#       all \
#       /path/to/output_dir \
#       4
#
# Arguments:
#   1. checkpoint path
#   2. total rollouts per environment, default 50
#   3. suite or environment name, default all
#   4. output directory, default ./parallel_eval
#   5. number of parallel processes, default 4
#
# Environment variables:
#   ROBOMIMIC_ROOT: robomimic project root
#   HORIZON: rollout horizon, default 400
#   SEED: base random seed, default 1
# ============================================================

ROBOMIMIC_ROOT="${ROBOMIMIC_ROOT:-/media/datasets/yumi/hjh/robo/robomimic}"

AGENT="${1:?Usage: eval_one_model_square_parallel.sh MODEL.pth [N_ROLLOUTS] [SUITE_OR_ENV] [OUTPUT_DIR] [NUM_WORKERS]}"
N_ROLLOUTS="${2:-50}"
SUITE_OR_ENV="${3:-all}"
OUTPUT_DIR="${4:-./parallel_eval}"
NUM_WORKERS="${5:-4}"

HORIZON="${HORIZON:-400}"
BASE_SEED="${SEED:-1}"

if [[ ! -f "${AGENT}" ]]; then
    echo "ERROR: checkpoint does not exist: ${AGENT}" >&2
    exit 1
fi

if (( N_ROLLOUTS <= 0 )); then
    echo "ERROR: N_ROLLOUTS must be positive." >&2
    exit 1
fi

if (( NUM_WORKERS <= 0 )); then
    echo "ERROR: NUM_WORKERS must be positive." >&2
    exit 1
fi

RUN_AGENT="${ROBOMIMIC_ROOT}/robomimic/scripts/run_trained_agent.py"

if [[ ! -f "${RUN_AGENT}" ]]; then
    echo "ERROR: run_trained_agent.py not found: ${RUN_AGENT}" >&2
    exit 1
fi

case "${SUITE_OR_ENV}" in
    clean)
        ENV_NAMES=(
            NutAssemblySquare
        )
        ;;
    object)
        ENV_NAMES=(
            NutAssemblySquare
            NutAssemblySquareObjectPerturb
        )
        ;;
    color)
        ENV_NAMES=(
            NutAssemblySquare
            NutAssemblySquareColorPerturb
        )
        ;;
    camera)
        ENV_NAMES=(
            NutAssemblySquare
            NutAssemblySquareCameraPerturb
        )
        ;;
    visual)
        ENV_NAMES=(
            NutAssemblySquare
            NutAssemblySquareVisualOOD
        )
        ;;
    task|dynamics)
        ENV_NAMES=(
            NutAssemblySquare
            NutAssemblySquareTaskDynamicsHard
        )
        ;;
    ood)
        ENV_NAMES=(
            NutAssemblySquare
            NutAssemblySquareVisualOOD
            NutAssemblySquareTaskDynamicsHard
        )
        ;;
    legacy)
        ENV_NAMES=(
            NutAssemblySquare
            NutAssemblySquareObjectPerturb
            NutAssemblySquareColorPerturb
            NutAssemblySquareCameraPerturb
        )
        ;;
    all)
        ENV_NAMES=(
            NutAssemblySquare
            NutAssemblySquareObjectPerturb
            NutAssemblySquareColorPerturb
            NutAssemblySquareCameraPerturb
            NutAssemblySquareVisualOOD
            NutAssemblySquareTaskDynamicsHard
        )
        ;;
    *)
        ENV_NAMES=(
            "${SUITE_OR_ENV}"
        )
        ;;
esac

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "Checkpoint       : ${AGENT}"
echo "Rollouts / env   : ${N_ROLLOUTS}"
echo "Parallel workers : ${NUM_WORKERS}"
echo "Horizon          : ${HORIZON}"
echo "Base seed        : ${BASE_SEED}"
echo "Output directory : ${OUTPUT_DIR}"
echo "Environments     : ${ENV_NAMES[*]}"
echo "============================================================"

GLOBAL_SUMMARY="${OUTPUT_DIR}/summary_all.json"
GLOBAL_TEMP="${OUTPUT_DIR}/.summary_records.jsonl"
: > "${GLOBAL_TEMP}"

for ENV_NAME in "${ENV_NAMES[@]}"; do
    echo
    echo "============================================================"
    echo "Evaluating environment: ${ENV_NAME}"
    echo "============================================================"

    ENV_DIR="${OUTPUT_DIR}/${ENV_NAME}"
    rm -rf "${ENV_DIR}"
    mkdir -p "${ENV_DIR}"

    BASE_COUNT=$((N_ROLLOUTS / NUM_WORKERS))
    REMAINDER=$((N_ROLLOUTS % NUM_WORKERS))

    PIDS=()
    WORKER_IDS=()

    for ((WORKER_ID=0; WORKER_ID<NUM_WORKERS; WORKER_ID++)); do
        COUNT="${BASE_COUNT}"

        if (( WORKER_ID < REMAINDER )); then
            COUNT=$((COUNT + 1))
        fi

        if (( COUNT <= 0 )); then
            continue
        fi

        WORKER_SEED=$((BASE_SEED + WORKER_ID * 100003))
        LOG_PATH="${ENV_DIR}/worker_${WORKER_ID}.log"

        echo "Starting worker ${WORKER_ID}: rollouts=${COUNT}, seed=${WORKER_SEED}"

        (
            echo "PARALLEL_EVAL_ENV=${ENV_NAME}"
            echo "PARALLEL_EVAL_WORKER_ID=${WORKER_ID}"
            echo "PARALLEL_EVAL_ROLLOUT_COUNT=${COUNT}"
            echo "PARALLEL_EVAL_SEED=${WORKER_SEED}"

            python "${RUN_AGENT}" \
                --agent "${AGENT}" \
                --env "${ENV_NAME}" \
                --n_rollouts "${COUNT}" \
                --horizon "${HORIZON}" \
                --seed "${WORKER_SEED}" \
                --camera_names agentview robot0_eye_in_hand
        ) > "${LOG_PATH}" 2>&1 &

        PIDS+=("$!")
        WORKER_IDS+=("${WORKER_ID}")
    done

    FAILED=0

    for INDEX in "${!PIDS[@]}"; do
        PID="${PIDS[$INDEX]}"
        WORKER_ID="${WORKER_IDS[$INDEX]}"

        if wait "${PID}"; then
            echo "Worker ${WORKER_ID} finished successfully."
        else
            echo "ERROR: worker ${WORKER_ID} failed." >&2
            echo "Log: ${ENV_DIR}/worker_${WORKER_ID}.log" >&2
            FAILED=1
        fi
    done

    if (( FAILED != 0 )); then
        echo
        echo "One or more workers failed for ${ENV_NAME}." >&2
        echo "Last 50 lines from failed worker logs:" >&2

        for LOG_PATH in "${ENV_DIR}"/worker_*.log; do
            if [[ -f "${LOG_PATH}" ]]; then
                echo "---------------- ${LOG_PATH} ----------------" >&2
                tail -n 50 "${LOG_PATH}" >&2
            fi
        done

        exit 1
    fi

    python - "${ENV_DIR}" "${N_ROLLOUTS}" "${ENV_NAME}" "${GLOBAL_TEMP}" <<'PY'
import json
import re
import sys
from pathlib import Path

env_dir = Path(sys.argv[1])
expected_rollouts = int(sys.argv[2])
env_name = sys.argv[3]
global_temp = Path(sys.argv[4])

worker_results = []
total_rollouts = 0
total_success = 0.0
weighted_return_sum = 0.0
weighted_horizon_sum = 0.0

for log_path in sorted(env_dir.glob("worker_*.log")):
    text = log_path.read_text(encoding="utf-8", errors="replace")

    count_matches = re.findall(
        r"PARALLEL_EVAL_ROLLOUT_COUNT=(\d+)",
        text,
    )
    if not count_matches:
        raise RuntimeError(
            f"Cannot find PARALLEL_EVAL_ROLLOUT_COUNT in {log_path}"
        )
    rollout_count = int(count_matches[-1])

    seed_matches = re.findall(
        r"PARALLEL_EVAL_SEED=(\d+)",
        text,
    )
    seed = int(seed_matches[-1]) if seed_matches else None

    success_matches = re.findall(
        r'"Num_Success"\s*:\s*([-+0-9.eE]+)',
        text,
    )
    if not success_matches:
        raise RuntimeError(
            f"Cannot find Num_Success in {log_path}"
        )
    num_success = float(success_matches[-1])

    return_matches = re.findall(
        r'"Return"\s*:\s*([-+0-9.eE]+)',
        text,
    )
    mean_return = (
        float(return_matches[-1])
        if return_matches
        else None
    )

    horizon_matches = re.findall(
        r'"Horizon"\s*:\s*([-+0-9.eE]+)',
        text,
    )
    mean_horizon = (
        float(horizon_matches[-1])
        if horizon_matches
        else None
    )

    success_rate = num_success / rollout_count

    worker_result = {
        "worker_log": log_path.name,
        "seed": seed,
        "num_rollouts": rollout_count,
        "num_success": num_success,
        "success_rate": success_rate,
        "mean_return": mean_return,
        "mean_horizon": mean_horizon,
    }
    worker_results.append(worker_result)

    total_rollouts += rollout_count
    total_success += num_success

    if mean_return is not None:
        weighted_return_sum += mean_return * rollout_count

    if mean_horizon is not None:
        weighted_horizon_sum += mean_horizon * rollout_count

if total_rollouts != expected_rollouts:
    raise RuntimeError(
        f"Expected {expected_rollouts} rollouts, "
        f"but parsed {total_rollouts}"
    )

summary = {
    "environment": env_name,
    "num_rollouts": total_rollouts,
    "num_success": total_success,
    "success_rate": total_success / total_rollouts,
    "mean_return": weighted_return_sum / total_rollouts,
    "mean_horizon": weighted_horizon_sum / total_rollouts,
    "workers": worker_results,
}

summary_path = env_dir / "summary.json"
summary_path.write_text(
    json.dumps(summary, indent=4),
    encoding="utf-8",
)

with global_temp.open("a", encoding="utf-8") as f:
    f.write(json.dumps(summary) + "\n")

print()
print("Evaluation summary:")
print(json.dumps(summary, indent=4))
print(f"Saved summary to: {summary_path}")
PY

done

python - "${GLOBAL_TEMP}" "${GLOBAL_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

temp_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])

results = []

for line in temp_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line:
        results.append(json.loads(line))

global_summary = {
    "environments": results,
}

summary_path.write_text(
    json.dumps(global_summary, indent=4),
    encoding="utf-8",
)

print()
print("============================================================")
print("All evaluations completed.")
print("============================================================")

for result in results:
    print(
        f"{result['environment']}: "
        f"{result['num_success']:.0f}/{result['num_rollouts']} "
        f"success, rate={result['success_rate']:.4f}, "
        f"mean_horizon={result['mean_horizon']:.2f}"
    )

print(f"Global summary saved to: {summary_path}")
PY

rm -f "${GLOBAL_TEMP}"