#!/usr/bin/env bash
# Tiny end-to-end smoke test for the magi-forward benchmark.
#
# Runs 4 cases (flat / merged x magi / gptmodel) at trajectories=4, single
# measured pass each, on a single 8-GPU node (cp=8). Verifies both forward
# paths execute end-to-end before kicking off the multi-hour full sweep.
#
# Runs inside a detached tmux session ``magi-bench-smoke`` so progress is
# observable while the wrapper exits immediately. Attach with:
#   tmux attach -t magi-bench-smoke
#
# Usage:
#   bash benchmark/magi_forward/run_smoke.sh
#   # remote node:
#   ssh <host> "bash -lc 'cd /path/to/axrl && bash benchmark/magi_forward/run_smoke.sh'"

set -euo pipefail

SESSION="${SESSION:-magi-bench-smoke}"
LOG_DIR="${LOG_DIR:-tmp/magi-bench-logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/smoke.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[smoke] tmux session ${SESSION} already exists; killing it first"
    tmux kill-session -t "${SESSION}"
fi

CMD="stdbuf -oL -eL python -u -m benchmark.magi_forward.run_magi_forward_benchmark --smoke 2>&1 | tee ${LOG_FILE}"
tmux new-session -d -s "${SESSION}" "${CMD}"
echo "[smoke] launched in tmux session: ${SESSION}"
echo "[smoke] log file: ${LOG_FILE}"
echo "[smoke] attach:   tmux attach -t ${SESSION}"
echo "[smoke] follow:   tail -f ${LOG_FILE}"
