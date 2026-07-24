#!/usr/bin/env bash
# Per-node bootstrap for the magi-forward pressure test.
#
# Invoked once per node (rank 0 = head, others = workers) by
# ``run_pressure_test.sh``. Reads the following env vars set by the
# orchestrator:
#   AXRL_NODE_RANK   — 0..N-1
#   AXRL_NODE_COUNT  — total nodes
#   AXRL_RAY_ADDR    — head node IP
#   AXRL_RAY_PORT    — head node ray port
#   AXRL_DATA_DIR    — for ModelConfig.get_full_path resolution
#   AXRL_MODEL_DIR   — same
#   AXRL_LOG_DIR
#   AXRL_OUTPUT_DIR
#   PROJECT_DIR      — absolute path to this repo on the shared filesystem
#
# Behaviour:
#   - rank 0: starts ray head, runs the python driver, then ray-stops.
#   - rank >0: starts ray worker, blocks until ray stops on rank 0.

set -euo pipefail

cd "${PROJECT_DIR}"

bash scripts/start_ray.sh

if [[ "${AXRL_NODE_RANK}" -eq 0 ]]; then
    echo "INFO: head node — running pressure-test driver"
    export RAY_ADDRESS="${AXRL_RAY_ADDR}:${AXRL_RAY_PORT}"
    LOG_FILE="${PROJECT_DIR}/tmp/magi-bench-logs/pressure.log"
    mkdir -p "$(dirname "${LOG_FILE}")"
    set +e
    stdbuf -oL -eL python -u -m benchmark.magi_forward.pressure_test \
        --output-dir "${PROJECT_DIR}/tmp/magi-bench-pressure" \
        --measured-passes "${PRESSURE_MEASURED_PASSES:-5}" \
        --phase "${PRESSURE_PHASE:-all}" \
        2>&1 | tee -a "${LOG_FILE}"
    DRIVER_EXIT=${PIPESTATUS[0]}
    set -e
    echo "INFO: driver exited with ${DRIVER_EXIT}; ray stop"
    ray stop --force
    exit "${DRIVER_EXIT}"
else
    echo "INFO: worker node rank ${AXRL_NODE_RANK} — waiting for ray to stop"
    while ray status &>/dev/null; do
        sleep 60
    done
    echo "INFO: ray stopped, worker exiting"
fi
