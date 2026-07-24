#!/usr/bin/bash

set -euo pipefail

export AXRL_OUTPUT_DIR_NAME="${AXRL_OUTPUT_DIR_NAME:-grpo_gsm8k}"
export AXRL_OUTPUT_DIR="${AXRL_OUTPUT_DIR:-${HOME}/axrl-data/outputs}"
export AXRL_LOG_DIR="${AXRL_LOG_DIR:-${AXRL_OUTPUT_DIR}/${AXRL_OUTPUT_DIR_NAME}/logs}"
mkdir -p "${AXRL_OUTPUT_DIR}/${AXRL_OUTPUT_DIR_NAME}" "${AXRL_LOG_DIR}"

python axis_recipe/grpo_gsm8k/generate_configs.py
project_name="2026-04-13-GRPO-GSM8K"

python -u axis_recipe/grpo_gsm8k/train_pipeline.py \
    --config_path="axis_recipe/grpo_gsm8k/pipeline_config.yaml" \
    --controller.output_dir_name="${AXRL_OUTPUT_DIR_NAME}" \
    --online_rl_train.checkpoint_every_n_global_updates=1000000000 \
    --logger.group_name="${AXRL_OUTPUT_DIR_NAME}" \
    --logger.project_name="${project_name}" \
    "$@" \
    2>&1 | tee "${AXRL_OUTPUT_DIR}/${AXRL_OUTPUT_DIR_NAME}/run-train-pipeline-grpo-GSM8K.log"
