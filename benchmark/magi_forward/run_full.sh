#!/usr/bin/env bash
# Full sweep for the magi-forward benchmark.
#
# Runs both sweeps on a single 8-GPU node (cp=8):
#   flat   x in {262144, 131072, 65536, 32768, 8192, 4096, 1024} seq lengths   (Fig 1, 2)
#   merged x in {256, 128, 64, 16, 4} turns                                    (Fig 3, 4)
#
# Each (sweep, size, method) case: 1 warmup + 5 measured compute_logprobs
# passes; CSV/JSON saved incrementally so a partial run is recoverable.
#
# After the sweep finishes, plots are regenerated from the CSV.
#
# Runs inside a detached tmux session ``magi-bench-full`` so progress is
# observable while the wrapper exits immediately. Attach with:
#   tmux attach -t magi-bench-full
#
# Usage:
#   bash benchmark/magi_forward/run_full.sh
#   # remote node:
#   ssh <host> "bash -lc 'cd /path/to/axrl && bash benchmark/magi_forward/run_full.sh'"

set -euo pipefail

SESSION="${SESSION:-magi-bench-full}"
OUTPUT_DIR="${OUTPUT_DIR:-tmp/magi-bench}"
LOG_DIR="${LOG_DIR:-tmp/magi-bench-logs}"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/full.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[full] tmux session ${SESSION} already exists; killing it first"
    tmux kill-session -t "${SESSION}"
fi

CMD="set -eo pipefail; \
stdbuf -oL -eL python -u -m benchmark.magi_forward.run_magi_forward_benchmark --sweep both --resume --output-dir ${OUTPUT_DIR} 2>&1 | tee -a ${LOG_FILE}; \
echo '[full] regenerating plots'; \
python -m benchmark.magi_forward.plot_results --input ${OUTPUT_DIR}/results.csv --output-dir ${OUTPUT_DIR}"
tmux new-session -d -s "${SESSION}" "${CMD}"
echo "[full] launched in tmux session: ${SESSION}"
echo "[full] log file: ${LOG_FILE}"
echo "[full] attach:   tmux attach -t ${SESSION}"
echo "[full] follow:   tail -f ${LOG_FILE}"
