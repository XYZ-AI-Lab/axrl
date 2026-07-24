#!/usr/bin/env bash
set -euo pipefail

# NIXL / UCX transports for the TensorStore (see axrl/utils/tensor_store.py).
# Drops TCP intentionally: a NIC regression fails loudly instead of silently
# falling back to TCP. Override in the environment for local tests without RDMA.
export UCX_TLS="${UCX_TLS:-rc,ud,sm,cma,self}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"

# Raise Raylet's inherited fd limit; GPU SSH sessions may default to 1024.
NOFILE_LIMIT="${AXRL_NOFILE_LIMIT:-65535}"
ulimit -n "${NOFILE_LIMIT}" || echo "WARN: failed to set NOFILE=${NOFILE_LIMIT}" >&2
echo "INFO: NOFILE soft/hard $(ulimit -Sn)/$(ulimit -Hn)"

WORKER_PORT_MIN="${AXRL_RAY_WORKER_PORT_MIN:-60000}"
WORKER_PORT_MAX="${AXRL_RAY_WORKER_PORT_MAX:-61000}"
CONTROL_PORT_BASE="${AXRL_RAY_CONTROL_PORT_BASE:-62000}"

GCS_SERVER_PORT="${AXRL_RAY_GCS_SERVER_PORT:-${CONTROL_PORT_BASE}}"
NODE_MANAGER_PORT="${AXRL_RAY_NODE_MANAGER_PORT:-$((CONTROL_PORT_BASE + 1))}"
OBJECT_MANAGER_PORT="${AXRL_RAY_OBJECT_MANAGER_PORT:-$((CONTROL_PORT_BASE + 2))}"
RAY_CLIENT_SERVER_PORT="${AXRL_RAY_CLIENT_SERVER_PORT:-$((CONTROL_PORT_BASE + 3))}"
DASHBOARD_PORT="${AXRL_RAY_DASHBOARD_PORT:-$((CONTROL_PORT_BASE + 4))}"
DASHBOARD_AGENT_LISTEN_PORT="${AXRL_RAY_DASHBOARD_AGENT_LISTEN_PORT:-$((CONTROL_PORT_BASE + 5))}"
DASHBOARD_AGENT_GRPC_PORT="${AXRL_RAY_DASHBOARD_AGENT_GRPC_PORT:-$((CONTROL_PORT_BASE + 6))}"
RUNTIME_ENV_AGENT_PORT="${AXRL_RAY_RUNTIME_ENV_AGENT_PORT:-$((CONTROL_PORT_BASE + 7))}"
METRICS_EXPORT_PORT="${AXRL_RAY_METRICS_EXPORT_PORT:-$((CONTROL_PORT_BASE + 8))}"

OBJECT_STORE_MEMORY_BYTES="${AXRL_RAY_OBJECT_STORE_MEMORY_BYTES:-}"
OBJECT_SPILLING_DIRECTORY="${AXRL_RAY_OBJECT_SPILLING_DIRECTORY:-}"
OBJECT_SPILLING_THRESHOLD="${AXRL_RAY_OBJECT_SPILLING_THRESHOLD:-0.95}"
RAY_SYSTEM_CONFIG_JSON="${AXRL_RAY_SYSTEM_CONFIG_JSON:-}"


COMMON_RAY_ARGS=(
	"--min-worker-port=${WORKER_PORT_MIN}"
	"--max-worker-port=${WORKER_PORT_MAX}"
	"--node-manager-port=${NODE_MANAGER_PORT}"
	"--object-manager-port=${OBJECT_MANAGER_PORT}"
	"--dashboard-agent-listen-port=${DASHBOARD_AGENT_LISTEN_PORT}"
	"--dashboard-agent-grpc-port=${DASHBOARD_AGENT_GRPC_PORT}"
	"--runtime-env-agent-port=${RUNTIME_ENV_AGENT_PORT}"
	"--metrics-export-port=${METRICS_EXPORT_PORT}"
)

HEAD_RAY_ARGS=()

if [[ -n "${AXRL_RAY_NUM_CPUS:-}" ]]; then
	COMMON_RAY_ARGS+=("--num-cpus=${AXRL_RAY_NUM_CPUS}")
fi

if [[ -n "${OBJECT_STORE_MEMORY_BYTES}" ]]; then
	COMMON_RAY_ARGS+=("--object-store-memory=${OBJECT_STORE_MEMORY_BYTES}")
fi

if [[ -n "${OBJECT_SPILLING_DIRECTORY}" ]]; then
	mkdir -p "${OBJECT_SPILLING_DIRECTORY}"
	COMMON_RAY_ARGS+=("--object-spilling-directory=${OBJECT_SPILLING_DIRECTORY}")
fi

if [[ -n "${RAY_SYSTEM_CONFIG_JSON}" ]]; then
	HEAD_RAY_ARGS+=("--system-config=${RAY_SYSTEM_CONFIG_JSON}")
elif [[ -n "${OBJECT_SPILLING_THRESHOLD}" ]]; then
	HEAD_RAY_ARGS+=("--system-config={\"object_spilling_threshold\":${OBJECT_SPILLING_THRESHOLD}}")
fi

wait_for_ray_cluster_ready() {
	if [[ "${AXRL_RAY_WAIT_FOR_CLUSTER:-true}" == "false" ]]; then
		echo "INFO: Skipping Ray readiness wait because AXRL_RAY_WAIT_FOR_CLUSTER=false"
		return
	fi

	local expected_nodes expected_gpus timeout_seconds poll_seconds
	expected_nodes="${AXRL_EXPECTED_RAY_NODES:-${AXRL_RAY_EXPECTED_NODES:-${AXRL_NODE_COUNT:-1}}}"
	timeout_seconds="${AXRL_RAY_READY_TIMEOUT_SECONDS:-1800}"
	poll_seconds="${AXRL_RAY_READY_POLL_SECONDS:-5}"

	if [[ -n "${AXRL_EXPECTED_RAY_GPUS:-}" ]]; then
		expected_gpus="${AXRL_EXPECTED_RAY_GPUS}"
	elif [[ -n "${AXRL_RAY_EXPECTED_GPUS:-}" ]]; then
		expected_gpus="${AXRL_RAY_EXPECTED_GPUS}"
	elif [[ -n "${GPU_PER_NODE_COUNT:-}" ]]; then
		expected_gpus="$((expected_nodes * GPU_PER_NODE_COUNT))"
	elif [[ -n "${AXRL_GPUS_PER_NODE:-}" ]]; then
		expected_gpus="$((expected_nodes * AXRL_GPUS_PER_NODE))"
	else
		expected_gpus=0
	fi

	echo "INFO: Waiting for Ray cluster readiness: nodes=${expected_nodes}, gpus=${expected_gpus}, timeout=${timeout_seconds}s"
	AXRL_RAY_READY_ADDRESS="${AXRL_RAY_ADDR}:${AXRL_RAY_PORT}" \
	AXRL_RAY_READY_EXPECTED_NODES="${expected_nodes}" \
	AXRL_RAY_READY_EXPECTED_GPUS="${expected_gpus}" \
	AXRL_RAY_READY_TIMEOUT_SECONDS="${timeout_seconds}" \
	AXRL_RAY_READY_POLL_SECONDS="${poll_seconds}" \
	"${AXRL_PYTHON_BIN:-python}" - <<'PY'
import os
import sys
import time

import ray


def _node_addr(node: dict) -> str:
    return str(node.get("NodeManagerAddress") or node.get("NodeManagerHostname") or node.get("NodeID") or "unknown")


address = os.environ["AXRL_RAY_READY_ADDRESS"]
expected_nodes = int(os.environ["AXRL_RAY_READY_EXPECTED_NODES"])
expected_gpus = int(os.environ["AXRL_RAY_READY_EXPECTED_GPUS"])
timeout_seconds = float(os.environ["AXRL_RAY_READY_TIMEOUT_SECONDS"])
poll_seconds = float(os.environ["AXRL_RAY_READY_POLL_SECONDS"])

deadline = time.monotonic() + timeout_seconds
last_message = ""
ray.init(address=address, ignore_reinit_error=True, namespace="_axrl_start_ray_wait", log_to_driver=False)
try:
    while True:
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        alive_addrs = sorted(_node_addr(node) for node in alive_nodes)
        gpus = int(ray.cluster_resources().get("GPU", 0))
        message = f"alive_nodes={len(alive_nodes)}/{expected_nodes} gpus={gpus}/{expected_gpus} addrs={alive_addrs}"
        if message != last_message:
            print(f"INFO: Ray readiness: {message}", flush=True)
            last_message = message

        if len(alive_nodes) >= expected_nodes and gpus >= expected_gpus:
            print("INFO: Ray readiness check passed.", flush=True)
            raise SystemExit(0)

        if time.monotonic() >= deadline:
            print(f"ERROR: Ray readiness timed out after {timeout_seconds:.0f}s: {message}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        time.sleep(poll_seconds)
finally:
    ray.shutdown()
PY
}

if [[ "${AXRL_NODE_RANK}" -eq 0 ]]; then
	export RAY_GCS_SERVER_PORT="${GCS_SERVER_PORT}"
	echo "INFO: Starting Ray head node on port ${AXRL_RAY_PORT}"
	echo "INFO: Ray worker ports ${WORKER_PORT_MIN}-${WORKER_PORT_MAX}; control ports base ${CONTROL_PORT_BASE}"
	if [[ -n "${OBJECT_STORE_MEMORY_BYTES}" ]]; then
		echo "INFO: Ray object store memory bytes ${OBJECT_STORE_MEMORY_BYTES}"
	fi
	if [[ -n "${OBJECT_SPILLING_DIRECTORY}" ]]; then
		echo "INFO: Ray object spilling directory ${OBJECT_SPILLING_DIRECTORY}"
	fi
	if [[ -n "${RAY_SYSTEM_CONFIG_JSON}" ]]; then
		echo "INFO: Ray system config ${RAY_SYSTEM_CONFIG_JSON}"
	else
		echo "INFO: Ray object spilling threshold ${OBJECT_SPILLING_THRESHOLD}"
	fi
	ray start \
		--head \
		--port="${AXRL_RAY_PORT}" \
		"--ray-client-server-port=${RAY_CLIENT_SERVER_PORT}" \
		"--dashboard-port=${DASHBOARD_PORT}" \
		"${COMMON_RAY_ARGS[@]}" \
		"${HEAD_RAY_ARGS[@]}"
	wait_for_ray_cluster_ready
else
	echo "INFO: Starting Ray worker node, connecting to ${AXRL_RAY_ADDR}:${AXRL_RAY_PORT}"
	RETRY_INTERVAL=10
	MAX_RETRIES=180 # Wait up to 30 minutes for the head node to be available
	for ((i = 1; i <= MAX_RETRIES; i++)); do
		if ray start --address="${AXRL_RAY_ADDR}:${AXRL_RAY_PORT}" "${COMMON_RAY_ARGS[@]}"; then
			echo "INFO: Successfully connected to Ray head node"
			break
		fi
		echo "WARN: Ray head not available yet, retrying in ${RETRY_INTERVAL}s (attempt ${i}/${MAX_RETRIES})"
		sleep "${RETRY_INTERVAL}"
	done
	if ((i > MAX_RETRIES)); then
		echo "ERROR: Failed to connect to Ray head after ${MAX_RETRIES} attempts" >&2
		exit 1
	fi
fi
