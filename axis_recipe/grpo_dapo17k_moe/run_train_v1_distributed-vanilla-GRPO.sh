#!/usr/bin/bash
set -euo pipefail

bash scripts/start_ray.sh

num_workers=$((2 * AXRL_NODE_COUNT))
dp_size=$((1 * AXRL_NODE_COUNT))
project_name="MOE-Math"

if [[ "${AXRL_NODE_RANK}" -eq 0 ]]; then
    echo "INFO: Head node — launching controller"
    export RAY_ADDRESS="${AXRL_RAY_ADDR}:${AXRL_RAY_PORT}"

    python axis_recipe/grpo_dapo17k_moe/generate_job_config.py

    python -u axis_recipe/grpo_dapo17k_moe/train_pipeline.py \
        --config_path="axis_recipe/grpo_dapo17k_moe/pipeline_config.yaml" \
        --controller.run_mode=online_rl_train \
        --controller.output_dir_name="${AXRL_OUTPUT_DIR_NAME}" \
        --rollout_worker.num_workers=${num_workers} \
        --megatron_worker.dp_size=${dp_size} \
        --logger.group_name="${AXRL_OUTPUT_DIR_NAME}" \
        --logger.project_name="${project_name}" \
        2>&1 | tee "${AXRL_OUTPUT_DIR}/run-pipeline-dapo17k-moe.log"

    ray stop --force
else
    echo "INFO: Worker node (rank ${AXRL_NODE_RANK}) — waiting for Ray to stop"
    # Block until the Ray process exits
    while ray status &>/dev/null; do
        sleep 120
        echo "ray worker waiting for ray to stop."
    done
    echo "INFO: Ray stopped, worker node exiting"
fi
