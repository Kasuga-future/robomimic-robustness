#!/usr/bin/env bash
# flow env evaluation
#SBATCH -J flow_env_eval
#SBATCH -p a100_global
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o /home/hejunhao-20251119/mnt/work/robomimic/logs/%x-%j.out
#SBATCH -e /home/hejunhao-20251119/mnt/work/robomimic/logs/%x-%j.err

# 不用 set -e，否则一个 eval 挂了整个脚本就退出
set -uo pipefail

echo "========================================="
echo "作业启动时间: $(date)"
echo "运行节点: $(hostname)"
echo "分配的 GPU: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "========================================="

nvidia-smi -L || true

ROOT_DIR="/home/hejunhao-20251119/mnt/work/robomimic"
cd "${ROOT_DIR}" || {
    echo "[FATAL] cd ${ROOT_DIR} failed"
    exit 1
}

mkdir -p "${ROOT_DIR}/logs"

source /mnt/public/apps/miniconda3/etc/profile.d/conda.sh
conda activate robomimic || {
    echo "[FATAL] conda activate robomimic failed"
    exit 1
}

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
export ROBOMIMIC_ROOT="${ROOT_DIR}"

echo "Python: $(which python)"
echo "Conda env: ${CONDA_DEFAULT_ENV:-<not set>}"
echo "MUJOCO_GL: ${MUJOCO_GL}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "ROBOMIMIC_ROOT: ${ROBOMIMIC_ROOT}"

python - <<'PY' || echo "[WARN] torch/cuda check failed, continue anyway"
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
PY

LIFT_EVAL_SCRIPT="${ROOT_DIR}/cpj_cbh/eval_one_model.sh"
SQUARE_EVAL_SCRIPT="${ROOT_DIR}/cpj_cbh/eval_one_model_square.sh"
OUT_ROOT="${ROOT_DIR}/env_log"

FAILED_TASKS=()
SUCCEEDED_TASKS=()

run_eval() {
    local name="$1"
    local eval_script="$2"
    local ckpt="$3"
    local rollouts="$4"
    local mode="$5"
    local out_dir="$6"
    local log_file="$7"

    mkdir -p "${out_dir}/videos"

    echo "========================================="
    echo "[RUN] ${name}"
    echo "开始时间: $(date)"
    echo "eval_script: ${eval_script}"
    echo "ckpt: ${ckpt}"
    echo "rollouts: ${rollouts}"
    echo "mode: ${mode}"
    echo "video_dir: ${out_dir}/videos"
    echo "log_file: ${log_file}"
    echo "========================================="

    bash "${eval_script}" \
        "${ckpt}" \
        "${rollouts}" \
        "${mode}" \
        "${out_dir}/videos" \
        2>&1 | tee "${log_file}"

    local ret=${PIPESTATUS[0]}

    echo "========================================="
    echo "[END] ${name}"
    echo "结束时间: $(date)"
    echo "返回码: ${ret}"
    echo "========================================="

    if [ "${ret}" -eq 0 ]; then
        SUCCEEDED_TASKS+=("${name}")
    else
        FAILED_TASKS+=("${name}(exit=${ret})")
    fi

    return 0
}

# =========================
# Square evals
# =========================

run_eval "square_flow_matching_epoch250" \
    "${SQUARE_EVAL_SCRIPT}" \
    "${ROOT_DIR}/robomimic/square_flow_matching_image_eval_logs/square_image_flow_matching_video_eval/20260529002639/models/model_epoch_250.pth" \
    25 \
    all \
    "${OUT_ROOT}/square/flow_matching" \
    "${OUT_ROOT}/square/flow_matching/eval_epoch250_all_25rollout.log"

run_eval "square_flow_matching_x_epoch300" \
    "${SQUARE_EVAL_SCRIPT}" \
    "${ROOT_DIR}/robomimic/square_flow_matching_x_image_eval_logs/square_image_flow_matching_x_video_eval/20260529020951/models/model_epoch_300.pth" \
    25 \
    all \
    "${OUT_ROOT}/square/flow_matching_x" \
    "${OUT_ROOT}/square/flow_matching_x/eval_epoch300_all_25rollout.log"

# =========================
# Lift evals
# =========================

run_eval "lift_flow_matching_epoch25" \
    "${LIFT_EVAL_SCRIPT}" \
    "${ROOT_DIR}/robomimic/lift_flow_matching_image_eval_logs/lift_image_flow_matching_video_eval/20260528234758/models/model_epoch_25.pth" \
    25 \
    all \
    "${OUT_ROOT}/lift/flow_matching" \
    "${OUT_ROOT}/lift/flow_matching/eval_epoch25_all.log"

run_eval "lift_flow_matching_x_epoch25" \
    "${LIFT_EVAL_SCRIPT}" \
    "${ROOT_DIR}/robomimic/lift_flow_matching_x_image_eval_logs/lift_image_flow_matching_x_video_eval/20260529000703/models/model_epoch_25.pth" \
    25 \
    all \
    "${OUT_ROOT}/lift/flow_matching_x" \
    "${OUT_ROOT}/lift/flow_matching_x/eval_epoch25_all.log"

echo "========================================="
echo "[SUMMARY]"
echo "作业结束时间: $(date)"
echo "成功任务数: ${#SUCCEEDED_TASKS[@]}"
printf '  %s\n' "${SUCCEEDED_TASKS[@]:-<none>}"

echo "失败任务数: ${#FAILED_TASKS[@]}"
printf '  %s\n' "${FAILED_TASKS[@]:-<none>}"
echo "========================================="

exit 0
