#!/usr/bin/bash
set -euo pipefail

# ------------------------------------------------------------------
# Distributed training script for Search-R1
#
# Expected environment variables (set by the job launcher):
#   AXRL_NODE_RANK   — 0 for head, 1+ for workers
#   AXRL_NODE_COUNT  — total number of nodes
#   AXRL_RAY_ADDR    — IP of the Ray head node
#   AXRL_RAY_PORT    — Ray GCS port
#   AXRL_SEARCH_PORT — port for the retrieval server (same on every node)
#   AXRL_OUTPUT_DIR  — directory for logs and checkpoints
#   AXRL_OUTPUT_DIR_NAME — human-readable run name (used as logger group)
# ------------------------------------------------------------------

# 1) Start Ray on this node (head or worker)
bash scripts/start_ray.sh

# 2) Start the retrieval server on *every* node so queries are served locally
bash axis_recipe/search_r1/start_retriever.sh

# 3) Compute total rollout workers (4 per node)
num_workers=$((4 * AXRL_NODE_COUNT))
# Megatron is colocated with rollout workers, so its world size must match the
# full rollout placement group: num_workers * rollout_gpus_per_worker.
# This recipe uses rollout tp=2 and megatron (tp=2, cp=2, pp=1), therefore
# megatron dp must scale with node count to consume the same total GPUs.
dp_size=$((2 * AXRL_NODE_COUNT))
project_name="2026-04-13-Search-R1"

if [[ "${AXRL_NODE_RANK}" -eq 0 ]]; then
    echo "INFO: Head node (rank 0) — launching Search-R1 controller"
    export RAY_ADDRESS="${AXRL_RAY_ADDR}:${AXRL_RAY_PORT}"

    # Generate the default config YAML
    python axis_recipe/search_r1/search_r1_config.py

    python -u axis_recipe/search_r1/train_search_r1.py \
        --config_path="axis_recipe/search_r1/search-r1-config.yaml" \
        --controller.run_mode=online_rl_train \
        --controller.output_dir_name="${AXRL_OUTPUT_DIR_NAME}" \
        --rollout_worker.num_workers=${num_workers} \
        --megatron_worker.dp_size=${dp_size} \
        --logger.group_name="${AXRL_OUTPUT_DIR_NAME}" \
        --logger.project_name="${project_name}" \
        2>&1 | tee "${AXRL_OUTPUT_DIR}/run-train-search-r1.log"

    ray stop --force
else
    echo "INFO: Worker node (rank ${AXRL_NODE_RANK}) — waiting for Ray to stop"
    # Block until the head node finishes and Ray shuts down
    while ray status &>/dev/null; do
        sleep 120
        echo "Ray worker (rank ${AXRL_NODE_RANK}) waiting for head to finish."
    done
    echo "INFO: Ray stopped, worker node (rank ${AXRL_NODE_RANK}) exiting"
fi
