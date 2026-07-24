#!/usr/bin/bash
set -euo pipefail

# ------------------------------------------------------------------
# Distributed rollout smoke test for the OpenHands black-box RL recipe.
#
# Expected environment variables from the job launcher:
#   AXRL_NODE_RANK   - 0 for head, 1+ for workers
#   AXRL_NODE_COUNT  - total number of nodes
#   AXRL_RAY_ADDR    - IP of the Ray head node
#   AXRL_RAY_PORT    - Ray GCS port
#   AXRL_OUTPUT_DIR  - output root directory for logs and generated case reports
#   AXRL_OUTPUT_DIR_NAME - human-readable run name (used as logger group)
#
# Optional smoke-test knobs:
#   AXRL_ROLLOUT_TEST_NUM_CASES       - number of OpenHands cases (default: 2)
#   AXRL_ROLLOUT_WORKERS_TOTAL        - total rollout workers across the Ray cluster (default: 4)
#   AXRL_MEGATRON_DP_SIZE             - Megatron DP size (default: 2)
#   AXRL_MEGATRON_CP_SIZE             - Megatron context parallel size (default: 2)
#   AXRL_OPENHANDS_TEST_FILE_DIR      - solution file dir in the E2B sandbox (default: tmp/openhands-case-study)
#   AXRL_ROLLOUT_MAX_NEW_TOKENS       - optional per-model-call generation cap
#   AXRL_OPENHANDS_MAX_MODEL_CALLS    - optional OpenHands model-call cap
#   AXRL_MODEL_NAME                   - optional Megatron/rollout model name override
#   AXRL_OPENAI_REASONING_PARSER      - optional OpenAI response reasoning parser, e.g. qwen3
#   AXRL_OPENAI_TOOL_CALL_PARSER      - optional OpenAI response tool-call parser, e.g. qwen
# ------------------------------------------------------------------

AXRL_NODE_RANK="${AXRL_NODE_RANK:-0}"
AXRL_NODE_COUNT="${AXRL_NODE_COUNT:-1}"
AXRL_RAY_ADDR="${AXRL_RAY_ADDR:-127.0.0.1}"
AXRL_RAY_PORT="${AXRL_RAY_PORT:-6379}"
AXRL_OUTPUT_DIR_NAME="${AXRL_OUTPUT_DIR_NAME:-blackbox-rl-rollout-test}"
AXRL_ROLLOUT_TEST_NUM_CASES="${AXRL_ROLLOUT_TEST_NUM_CASES:-2}"
AXRL_ROLLOUT_WORKERS_TOTAL="${AXRL_ROLLOUT_WORKERS_TOTAL:-4}"
AXRL_MEGATRON_DP_SIZE="${AXRL_MEGATRON_DP_SIZE:-2}"
AXRL_MEGATRON_CP_SIZE="${AXRL_MEGATRON_CP_SIZE:-2}"
AXRL_ROLLOUT_TP_SIZE="${AXRL_ROLLOUT_TP_SIZE:-4}"
AXRL_ROLLOUT_ACTORS="${AXRL_ROLLOUT_ACTORS:-16}"
AXRL_CPUS_PER_ROLLOUT_ACTOR="${AXRL_CPUS_PER_ROLLOUT_ACTOR:-4}"
AXRL_RAY_CLUSTER_WAIT_SECONDS="${AXRL_RAY_CLUSTER_WAIT_SECONDS:-600}"
AXRL_OPENHANDS_TEST_FILE_DIR="${AXRL_OPENHANDS_TEST_FILE_DIR:-tmp/openhands-case-study}"
AXRL_ROLLOUT_MAX_NEW_TOKENS="${AXRL_ROLLOUT_MAX_NEW_TOKENS:-}"
AXRL_OPENHANDS_MAX_MODEL_CALLS="${AXRL_OPENHANDS_MAX_MODEL_CALLS:-}"
AXRL_MODEL_NAME="${AXRL_MODEL_NAME:-}"
AXRL_OPENAI_REASONING_PARSER="${AXRL_OPENAI_REASONING_PARSER:-}"
AXRL_OPENAI_TOOL_CALL_PARSER="${AXRL_OPENAI_TOOL_CALL_PARSER:-}"
AXRL_BLACKBOX_CONFIG="${AXRL_BLACKBOX_CONFIG:-axis_recipe/blackbox_rl/blackbox-rl-config.yaml}"
project_name="${AXRL_PROJECT_NAME:-BlackBoxRL}"
launcher_axrl_output_dir="${AXRL_OUTPUT_DIR:-}"

if [[ -f "${HOME}/.axrl_env.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/.axrl_env.sh"
fi

if [[ -f .axrl/user.env.sh ]]; then
    # shellcheck source=/dev/null
    source .axrl/user.env.sh
fi

# shellcheck source=axis_recipe/blackbox_rl/launcher_env.sh
source axis_recipe/blackbox_rl/launcher_env.sh

AXRL_NODE_RANK="${AXRL_BLACKBOX_NODE_RANK:-${AXRL_NODE_RANK}}"
AXRL_NODE_COUNT="${AXRL_BLACKBOX_NODE_COUNT:-${AXRL_NODE_COUNT}}"
AXRL_RAY_ADDR="${AXRL_BLACKBOX_RAY_ADDR:-${AXRL_RAY_ADDR}}"
AXRL_RAY_PORT="${AXRL_BLACKBOX_RAY_PORT:-${AXRL_RAY_PORT}}"
export AXRL_NODE_RANK AXRL_NODE_COUNT AXRL_RAY_ADDR AXRL_RAY_PORT
default_output_root="${AXRL_SHM_ROOT:-${HOME}/axrl-data}/outputs"
AXRL_OUTPUT_DIR="${AXRL_BLACKBOX_OUTPUT_DIR:-${launcher_axrl_output_dir:-${default_output_root}}}"
export AXRL_OUTPUT_DIR
export AXRL_LOG_DIR="${AXRL_OUTPUT_DIR}/${AXRL_OUTPUT_DIR_NAME}/logs"

mkdir -p "${AXRL_OUTPUT_DIR}/${AXRL_OUTPUT_DIR_NAME}" "${AXRL_LOG_DIR}"

# 1) Start Ray on this node (head or worker).
bash scripts/start_ray.sh

# 2) Keep worker sizes as total cluster sizes. Defaults consume 16 GPUs total:
#    rollout: 4 workers * tp=4, Megatron: dp=2 * tp=4 * cp=2.
wait_for_ray_cluster() {
    if [[ "${AXRL_NODE_RANK}" -ne 0 ]]; then
        return
    fi
    AXRL_EXPECTED_GPUS=$((AXRL_ROLLOUT_WORKERS_TOTAL * AXRL_ROLLOUT_TP_SIZE)) \
    AXRL_RAY_CLUSTER_WAIT_SECONDS="${AXRL_RAY_CLUSTER_WAIT_SECONDS}" \
        python - <<'PY'
import os
import time

import ray

address = f"{os.environ['AXRL_RAY_ADDR']}:{os.environ['AXRL_RAY_PORT']}"
target_nodes = int(os.environ["AXRL_NODE_COUNT"])
target_gpus = float(os.environ["AXRL_EXPECTED_GPUS"])
deadline = time.time() + int(os.environ["AXRL_RAY_CLUSTER_WAIT_SECONDS"])
ray.init(address=address, ignore_reinit_error=True, logging_level="ERROR")
while time.time() < deadline:
    alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
    gpus = float(ray.cluster_resources().get("GPU", 0))
    if len(alive_nodes) >= target_nodes and gpus >= target_gpus:
        print(f"INFO: Ray cluster ready: nodes={len(alive_nodes)}, gpus={gpus}.", flush=True)
        raise SystemExit(0)
    print(f"INFO: Waiting for Ray cluster: nodes={len(alive_nodes)}/{target_nodes}, gpus={gpus}/{target_gpus}.", flush=True)
    time.sleep(10)
raise SystemExit(f"Timed out waiting for Ray cluster nodes={target_nodes}, gpus={target_gpus}.")
PY
}

wait_for_ray_cluster

max_running_requests=$((AXRL_NODE_COUNT * AXRL_ROLLOUT_TEST_NUM_CASES))
if [[ "${max_running_requests}" -lt "${AXRL_ROLLOUT_ACTORS}" ]]; then
    max_running_requests="${AXRL_ROLLOUT_ACTORS}"
fi

if [[ "${AXRL_NODE_RANK}" -eq 0 ]]; then
    echo "INFO: Head node (rank 0) - launching OpenHands black-box RL rollout smoke"
    export RAY_ADDRESS="${AXRL_RAY_ADDR}:${AXRL_RAY_PORT}"

    # Generate the default config YAML when the caller did not provide one.
    if [[ "${AXRL_BLACKBOX_CONFIG}" == "axis_recipe/blackbox_rl/blackbox-rl-config.yaml" && ! -f "${AXRL_BLACKBOX_CONFIG}" ]]; then
        python axis_recipe/blackbox_rl/config.py
    fi

    rollout_args=(
        --num_cases="${AXRL_ROLLOUT_TEST_NUM_CASES}"
        --config_path="${AXRL_BLACKBOX_CONFIG}"
        --controller.run_mode=eval_only
        --controller.output_dir_name="${AXRL_OUTPUT_DIR_NAME}"
        --controller.max_running_requests="${max_running_requests}"
        --controller.num_rollout_actors="${AXRL_ROLLOUT_ACTORS}"
        --controller.num_cpus_per_actor="${AXRL_CPUS_PER_ROLLOUT_ACTOR}"
        --rollout_worker.num_workers="${AXRL_ROLLOUT_WORKERS_TOTAL}"
        --rollout_worker.tp_size="${AXRL_ROLLOUT_TP_SIZE}"
        --megatron_worker.dp_size="${AXRL_MEGATRON_DP_SIZE}"
        --megatron_worker.cp_size="${AXRL_MEGATRON_CP_SIZE}"
        --logger.group_name="${AXRL_OUTPUT_DIR_NAME}"
        --logger.project_name="${project_name}"
        --openhands_env.test_file_dir="${AXRL_OPENHANDS_TEST_FILE_DIR}"
    )

    if [[ -n "${AXRL_ROLLOUT_MAX_NEW_TOKENS}" ]]; then
        rollout_args+=(
            --eval_sampling_config.max_new_tokens="${AXRL_ROLLOUT_MAX_NEW_TOKENS}"
            --train_sampling_config.max_new_tokens="${AXRL_ROLLOUT_MAX_NEW_TOKENS}"
            --rollout_worker.sampling_config.max_new_tokens="${AXRL_ROLLOUT_MAX_NEW_TOKENS}"
        )
    fi

    if [[ -n "${AXRL_OPENHANDS_MAX_MODEL_CALLS}" ]]; then
        rollout_args+=(--openhands_env.max_model_calls="${AXRL_OPENHANDS_MAX_MODEL_CALLS}")
    fi

    if [[ -n "${AXRL_MODEL_NAME}" ]]; then
        rollout_args+=(
            --megatron_worker.model.name="${AXRL_MODEL_NAME}"
            --rollout_worker.model.name="${AXRL_MODEL_NAME}"
        )
    fi

    if [[ -n "${AXRL_OPENAI_REASONING_PARSER}" ]]; then
        rollout_args+=(--openai_proxy.reasoning_parser="${AXRL_OPENAI_REASONING_PARSER}")
    fi

    if [[ -n "${AXRL_OPENAI_TOOL_CALL_PARSER}" ]]; then
        rollout_args+=(--openai_proxy.tool_call_parser="${AXRL_OPENAI_TOOL_CALL_PARSER}")
    fi

    status=0
    python -u axis_recipe/blackbox_rl/run_rollout_test.py "${rollout_args[@]}" \
        2>&1 | tee "${AXRL_OUTPUT_DIR}/${AXRL_OUTPUT_DIR_NAME}/run-rollout-test-blackbox-rl.log" || status=$?

    ray stop --force
    exit "${status}"
else
    echo "INFO: Worker node (rank ${AXRL_NODE_RANK}) - waiting for Ray to stop"
    while timeout 20 ray status &>/dev/null; do
        sleep 120
        echo "Ray worker (rank ${AXRL_NODE_RANK}) waiting for head to finish."
    done
    ray stop --force
    echo "INFO: Ray stopped, worker node (rank ${AXRL_NODE_RANK}) exiting"
fi
