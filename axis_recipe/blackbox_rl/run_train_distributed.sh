#!/usr/bin/bash
set -euo pipefail

AXRL_NODE_RANK="${AXRL_NODE_RANK:-0}"
AXRL_NODE_COUNT="${AXRL_NODE_COUNT:-1}"
AXRL_RAY_ADDR="${AXRL_RAY_ADDR:-127.0.0.1}"
AXRL_RAY_PORT="${AXRL_RAY_PORT:-6379}"
AXRL_OUTPUT_DIR_NAME="${AXRL_OUTPUT_DIR_NAME:-blackbox-rl-train}"
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

bash scripts/start_ray.sh

num_workers="${AXRL_ROLLOUT_WORKERS_TOTAL:-$((2 * AXRL_NODE_COUNT))}"
dp_size="${AXRL_MEGATRON_DP_SIZE:-$((1 * AXRL_NODE_COUNT))}"

if [[ "${AXRL_NODE_RANK}" -eq 0 ]]; then
    echo "INFO: Head node - launching OpenHands black-box RL train"
    export RAY_ADDRESS="${AXRL_RAY_ADDR}:${AXRL_RAY_PORT}"

    if [[ ! -f "${AXRL_BLACKBOX_CONFIG}" ]]; then
        python axis_recipe/blackbox_rl/config.py
    fi

    train_args=(
        --config_path="${AXRL_BLACKBOX_CONFIG}"
        --controller.run_mode=online_rl_train
        --controller.output_dir_name="${AXRL_OUTPUT_DIR_NAME}"
        --rollout_worker.num_workers="${num_workers}"
        --megatron_worker.dp_size="${dp_size}"
        --logger.group_name="${AXRL_OUTPUT_DIR_NAME}"
        --logger.project_name="${project_name}"
    )

    status=0
    python -u axis_recipe/blackbox_rl/train_blackbox_rl.py "${train_args[@]}" "$@" \
        2>&1 | tee "${AXRL_OUTPUT_DIR}/${AXRL_OUTPUT_DIR_NAME}/run-train-blackbox-rl.log" || status=$?

    ray stop --force
    exit "${status}"
else
    echo "INFO: Worker node (rank ${AXRL_NODE_RANK}) - waiting for Ray to stop"
    while timeout 20 ray status &>/dev/null; do
        sleep 120
        echo "ray worker waiting for ray to stop."
    done
    ray stop --force
    echo "INFO: Ray stopped, worker node exiting"
fi
