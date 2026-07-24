#!/usr/bin/env bash

set -euo pipefail

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
GROUP_NAME="colocated-qwen3-32b_${RUN_TIMESTAMP}"
PROJECT_NAME="2026-04-13-Weight-Update"

mkdir -p tmp
python "benchmark/weight_update/run_weight_update_benchmark.py" \
	--logger.group_name="${GROUP_NAME}" \
	--logger.project_name="${PROJECT_NAME}" \
	2>&1 | tee "${AXRL_OUTPUT_DIR}/weight-update-${GROUP_NAME}.log"

python "benchmark/weight_update/run_fp8_weight_sync_benchmark.py" \
	--logger.group_name="${GROUP_NAME}" \
	--logger.project_name="${PROJECT_NAME}" \
	2>&1 | tee "${AXRL_OUTPUT_DIR}/weight-update-${GROUP_NAME}-fp8.log"
