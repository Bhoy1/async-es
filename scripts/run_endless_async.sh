#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${MODEL_NAME:=Qwen/Qwen2.5-7B-Instruct}"
: "${TASK_ADAPTER:=tasks.endless_terminals.adapter:EndlessTerminalsAdapter}"
: "${NUM_ENGINES:=4}"
: "${CUDA_DEVICES:=0,1,2,3}"
: "${POPULATION_SIZE:=30}"
: "${TRAIN_BATCH_SIZE:=256}"
: "${NUM_ITERATIONS:=100}"
: "${MAX_POLICY_STALENESS:=1}"
: "${PARAMETERIZATION:=full}"
: "${SIGMA:=}"
: "${ALPHA:=}"
: "${LORA_R:=32}"
: "${LORA_ALPHA:=64}"
: "${LORA_TARGET_MODULES:=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
: "${MAX_TOKENS:=2048}"
: "${MAX_MODEL_LEN:=32768}"
: "${ENDLESS_MAX_INPUT_TOKENS:=16384}"
: "${GPU_MEMORY_UTILIZATION:=0.9}"
: "${ENDLESS_MAX_TURNS:=16}"
: "${ENDLESS_MAX_TIME:=300}"
: "${ENDLESS_ENV_BATCH_SIZE:=32}"
: "${ENDLESS_ENV_WORKERS:=32}"
: "${EVAL_MAX_SAMPLES:=100}"
: "${EVAL_INTERVAL:=10}"
: "${CHECKPOINT_INTERVAL:=5}"
: "${CHECKPOINT_KEEP_LAST:=3}"
: "${CHECKPOINT_KEEP_ITERATIONS:=25,50,75,100}"
: "${WANDB_PROJECT:=async-es}"
: "${WANDB_ENTITY:=}"
: "${WANDB_MODE:=online}"
: "${EXPERIMENT_DIR:=${ROOT}/outputs}"
: "${RUN_NAME:=endless_natural_async_$(date +%Y%m%d_%H%M%S)}"
: "${GLOBAL_SEED:=42}"
: "${TRACK_TOKEN_ENTROPY:=0}"
: "${CENTER_ENTROPY_INTERVAL:=10}"
: "${RESUME_CHECKPOINT:=}"
: "${START_ITERATION:=0}"
: "${SKIP_INITIAL_EVAL:=0}"

: "${ENDLESS_OFFICIAL_REPO:?Set ENDLESS_OFFICIAL_REPO to the pinned Endless Terminals checkout}"
: "${ENDLESS_TRAIN_DATA_PATH:?Set ENDLESS_TRAIN_DATA_PATH to train.parquet}"
: "${ENDLESS_EVAL_DATA_PATH:?Set ENDLESS_EVAL_DATA_PATH to validation.parquet}"

COMMAND=(python "${ROOT}/train.py" \
  --task_adapter "${TASK_ADAPTER}" \
  --model_name "${MODEL_NAME}" \
  --max_policy_staleness "${MAX_POLICY_STALENESS}" \
  --num_engines "${NUM_ENGINES}" \
  --cuda_devices "${CUDA_DEVICES}" \
  --population_size "${POPULATION_SIZE}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --num_iterations "${NUM_ITERATIONS}" \
  --parameterization "${PARAMETERIZATION}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --max_tokens "${MAX_TOKENS}" \
  --max_model_len "${MAX_MODEL_LEN}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --train_temperature 0.6 \
  --train_top_p 1.0 \
  --train_top_k -1 \
  --eval_temperature 0.6 \
  --eval_top_p 1.0 \
  --eval_top_k -1 \
  --endless_official_repo "${ENDLESS_OFFICIAL_REPO}" \
  --endless_train_data_path "${ENDLESS_TRAIN_DATA_PATH}" \
  --endless_eval_data_path "${ENDLESS_EVAL_DATA_PATH}" \
  --endless_rollout_scheduler continuous \
  --endless_max_turns "${ENDLESS_MAX_TURNS}" \
  --endless_max_time "${ENDLESS_MAX_TIME}" \
  --endless_max_input_tokens "${ENDLESS_MAX_INPUT_TOKENS}" \
  --endless_env_batch_size "${ENDLESS_ENV_BATCH_SIZE}" \
  --endless_env_workers "${ENDLESS_ENV_WORKERS}" \
  --eval_max_samples "${EVAL_MAX_SAMPLES}" \
  --eval_interval "${EVAL_INTERVAL}" \
  --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
  --checkpoint_keep_last "${CHECKPOINT_KEEP_LAST}" \
  --checkpoint_keep_iterations "${CHECKPOINT_KEEP_ITERATIONS}" \
  --experiment_dir "${EXPERIMENT_DIR}" \
  --run_name "${RUN_NAME}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_mode "${WANDB_MODE}" \
  --global_seed "${GLOBAL_SEED}" \
  --center_entropy_interval "${CENTER_ENTROPY_INTERVAL}" \
  --precision bfloat16)

if [[ -n "${SIGMA}" ]]; then
  COMMAND+=(--sigma "${SIGMA}")
fi
if [[ -n "${ALPHA}" ]]; then
  COMMAND+=(--alpha "${ALPHA}")
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  COMMAND+=(--wandb_entity "${WANDB_ENTITY}")
fi
if [[ "${TRACK_TOKEN_ENTROPY}" == "1" ]]; then
  COMMAND+=(--track_token_entropy)
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  if [[ "${START_ITERATION}" == "0" ]]; then
    echo "START_ITERATION must be set when RESUME_CHECKPOINT is set" >&2
    exit 2
  fi
  COMMAND+=(
    --resume_checkpoint "${RESUME_CHECKPOINT}"
    --start_iteration "${START_ITERATION}"
  )
fi
if [[ "${SKIP_INITIAL_EVAL}" == "1" ]]; then
  COMMAND+=(--skip_initial_eval)
fi

exec "${COMMAND[@]}"
