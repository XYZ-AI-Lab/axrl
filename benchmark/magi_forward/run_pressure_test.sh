#!/usr/bin/env bash
# Multi-node launcher for the magi-forward pressure test.
#
# Reads SSH destinations from a node file, one command per line.
# Each node should be able to source ``/root/.axrl_env.sh`` or otherwise provide
# AXRL_NODE_RANK, AXRL_NODE_COUNT, AXRL_RAY_ADDR, AXRL_RAY_PORT, and AXRL_*_DIR.
#
# This script:
#   - SSH-launches ``pressure_node_bootstrap.sh`` on every node inside a
#     detached tmux session (``magi-pressure-rank<N>``).
#   - Waits for the rank-0 driver session to exit.
#   - On exit/interrupt, ``ray stop --force`` + tmux kill on every node.
#
# Usage:
#   bash benchmark/magi_forward/run_pressure_test.sh
#
# Optional env overrides:
#   NODE_FILE (default: tmp/magi-pressure-nodes.txt)
#   PROJECT_DIR (default: pwd)
#   PRESSURE_MEASURED_PASSES (default: 5)
#   PRESSURE_PHASE (default: all; 'inference' or 'training' to run one phase)

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd -P)}"
NODE_FILE="${NODE_FILE:-${PROJECT_DIR}/tmp/magi-pressure-nodes.txt}"
SESSION_PREFIX="${SESSION_PREFIX:-magi-pressure-rank}"

if [[ ! -f "${NODE_FILE}" ]]; then
    echo "ERROR: ${NODE_FILE} not found" >&2
    exit 1
fi

mapfile -t SSH_LINES < "${NODE_FILE}"
NODE_COUNT="${#SSH_LINES[@]}"
echo "[pressure] ${NODE_COUNT} nodes from ${NODE_FILE}"

mkdir -p "${PROJECT_DIR}/tmp/magi-bench-logs"

cleanup() {
    echo "[pressure] cleanup: ray stop on all nodes"
    for ssh_cmd in "${SSH_LINES[@]}"; do
        local_ssh="${ssh_cmd/ssh /ssh -o StrictHostKeyChecking=no }"
        ${local_ssh} "ray stop --force >/dev/null 2>&1 || true; tmux kill-server 2>/dev/null || true" &
    done
    wait
}
trap cleanup EXIT INT TERM

# Launch each node in a detached tmux session. The bootstrap script reads
# AXRL_NODE_RANK / AXRL_RAY_ADDR / AXRL_RAY_PORT from /root/.axrl_env.sh.
for rank in "${!SSH_LINES[@]}"; do
    ssh_cmd="${SSH_LINES[$rank]}"
    SESSION="${SESSION_PREFIX}${rank}"
    REMOTE_SCRIPT=$(cat <<EOF
set -e
source /root/.axrl_env.sh
cd ${PROJECT_DIR}
export PROJECT_DIR='${PROJECT_DIR}'
export PRESSURE_MEASURED_PASSES='${PRESSURE_MEASURED_PASSES:-5}'
export PRESSURE_PHASE='${PRESSURE_PHASE:-all}'
tmux kill-session -t ${SESSION} 2>/dev/null || true
tmux new-session -d -s ${SESSION} \
    'source /root/.axrl_env.sh && export PROJECT_DIR=${PROJECT_DIR} && export PRESSURE_MEASURED_PASSES=${PRESSURE_MEASURED_PASSES:-5} && export PRESSURE_PHASE=${PRESSURE_PHASE:-all} && \
     bash ${PROJECT_DIR}/benchmark/magi_forward/pressure_node_bootstrap.sh 2>&1 \
       | tee -a ${PROJECT_DIR}/tmp/magi-bench-logs/pressure-rank${rank}.log'
EOF
)
    echo "[pressure] launching rank ${rank} via: ${ssh_cmd}"
    ${ssh_cmd/ssh /ssh -o StrictHostKeyChecking=no } "${REMOTE_SCRIPT}"
done

echo "[pressure] all ${NODE_COUNT} nodes launched."
echo "[pressure] driver log: tail -f ${PROJECT_DIR}/tmp/magi-bench-logs/pressure.log"
echo "[pressure] per-rank: tail -f ${PROJECT_DIR}/tmp/magi-bench-logs/pressure-rank<N>.log"
echo "[pressure] head session attach:"
echo "  ${SSH_LINES[0]/ssh /ssh -t -o StrictHostKeyChecking=no } tmux attach -t ${SESSION_PREFIX}0"

# Block until rank-0 tmux exits (= driver finished, ray was stopped).
echo "[pressure] waiting for rank-0 driver to finish..."
HEAD_SSH_RELAXED="${SSH_LINES[0]/ssh /ssh -o StrictHostKeyChecking=no }"
while ${HEAD_SSH_RELAXED} "tmux has-session -t ${SESSION_PREFIX}0 2>/dev/null"; do
    sleep 60
done
echo "[pressure] driver finished"
