#!/usr/bin/bash

set -euo pipefail

export AXRL_DATA_DIR="${AXRL_DATA_DIR:-${HOME}/axrl-data/datasets}"
export AXRL_OUTPUT_DIR_NAME="${AXRL_OUTPUT_DIR_NAME:-search_r1}"
export AXRL_OUTPUT_DIR="${AXRL_OUTPUT_DIR:-${HOME}/axrl-data/outputs/${AXRL_OUTPUT_DIR_NAME}}"
mkdir -p "${AXRL_OUTPUT_DIR}"

bash axis_recipe/search_r1/start_retriever.sh
python axis_recipe/search_r1/search_r1_config.py
project_name="2026-04-13-Search-R1"

python -u axis_recipe/search_r1/train_search_r1.py \
    --config_path="axis_recipe/search_r1/search-r1-config.yaml" \
    --logger.group_name="${AXRL_OUTPUT_DIR_NAME}" \
    --logger.project_name="${project_name}" \
    2>&1 | tee "${AXRL_OUTPUT_DIR}/run-train-pipeline-search-r1.log"
