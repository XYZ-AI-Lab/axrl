#!/usr/bin/bash
# Optimized Memory Benchmark for Qwen3-30B-A3B MoE on 8x H200
# Uses max_global_updates=2 to reduce per-experiment time
# Each experiment: ~7min startup + ~2min training = ~9min
#
# Reference: Megatron-LM activation checkpointing
#   https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_block.py
# Reference: verl McoreEngineConfig memory optimization
#   verl/verl/workers/config/engine.py (lines 121-181)
# Reference: slime selective recompute for MoE
#   slime uses recompute_modules=["mla_up_proj"] for DeepSeek-like architectures

set -uo pipefail

BENCHMARK_DIR="${AXRL_OUTPUT_DIR}/benchmark/memory-saving"
mkdir -p "${BENCHMARK_DIR}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY_FILE="${BENCHMARK_DIR}/summary-${TIMESTAMP}.txt"
CONFIG_PATH="axis_recipe/moe/grpo_config.yaml"
ROLLOUT_SAVE_FILENAME="valid_rollouts-memory-saving-debug.zst"
ROLLOUT_PATH="${AXRL_OUTPUT_DIR}/grpo/${ROLLOUT_SAVE_FILENAME}"
GLOBAL_BATCH_SIZE=128
RUN_TIMEOUT_SECONDS=1800
ROLLOUT_NUM_ITERS_PER_WEIGHT_SYNC=4
ROLLOUT_BATCH_ROLLOUTS=4
ROLLOUT_BOOTSTRAP_MAX_UPDATES=4
ROLLOUT_FILTER_ZERO_STD=false

python axis_recipe/moe/generate_job_config.py

COMMON_ARGS=(
    --config_path="${CONFIG_PATH}"
    --eval_on_start=false
    --grpo.rollout_save_filename="${ROLLOUT_SAVE_FILENAME}"
    --grpo.model_sync_every_n_global_updates="${ROLLOUT_NUM_ITERS_PER_WEIGHT_SYNC}"
    --grpo.batch_rollout_for_n_global_updates="${ROLLOUT_BATCH_ROLLOUTS}"
    --grpo.filter_zero_std="${ROLLOUT_FILTER_ZERO_STD}"
    --megatron_worker.global_batch_size="${GLOBAL_BATCH_SIZE}"
    --megatron_worker.enable_fp32_lm_head=false
    --megatron_worker.enable_routing_replay=false
    --megatron_worker.lr_scheduler.lr_warmup_steps=0
)

echo "=== Memory Saving Benchmark Started at $(date) ===" | tee "${SUMMARY_FILE}"
echo "Model: Qwen3-30B-A3B-Instruct-2507 (48 layers, 128 experts, top-8)" | tee -a "${SUMMARY_FILE}"
echo "Hardware: 8x NVIDIA H200 (143GB)" | tee -a "${SUMMARY_FILE}"
echo "Parallelism: TP=1, PP=1, DP=2, CP=4, EP=4, ETP=1" | tee -a "${SUMMARY_FILE}"
echo "Global batch size: ${GLOBAL_BATCH_SIZE}" | tee -a "${SUMMARY_FILE}"
echo "Benchmark timeout: ${RUN_TIMEOUT_SECONDS}s" | tee -a "${SUMMARY_FILE}"
echo "Rollout snapshot: ${ROLLOUT_PATH}" | tee -a "${SUMMARY_FILE}"
echo "Rollout iters per weight sync: ${ROLLOUT_NUM_ITERS_PER_WEIGHT_SYNC}" | tee -a "${SUMMARY_FILE}"
echo "Rollout batch windows: ${ROLLOUT_BATCH_ROLLOUTS}" | tee -a "${SUMMARY_FILE}"
echo "Rollout bootstrap max updates: ${ROLLOUT_BOOTSTRAP_MAX_UPDATES}" | tee -a "${SUMMARY_FILE}"
echo "Rollout filter_zero_std: ${ROLLOUT_FILTER_ZERO_STD}" | tee -a "${SUMMARY_FILE}"
echo "" | tee -a "${SUMMARY_FILE}"

ensure_rollout_snapshot() {
    if [[ -s "${ROLLOUT_PATH}" ]]; then
        return
    fi

    local bootstrap_log="${BENCHMARK_DIR}/bootstrap-rollouts-${TIMESTAMP}.log"
    ray stop --force 2>/dev/null || true
    sleep 3

    python -u axrl/controller/run_grpo_controller.py \
        "${COMMON_ARGS[@]}" \
        --debug_train=false \
        --grpo.max_global_updates="${ROLLOUT_BOOTSTRAP_MAX_UPDATES}" \
        > "${bootstrap_log}" 2>&1

    if [[ ! -s "${ROLLOUT_PATH}" ]]; then
        echo "Snapshot generation failed. Log: ${bootstrap_log}" | tee -a "${SUMMARY_FILE}"
        return 1
    fi

    echo "Snapshot ready: ${ROLLOUT_PATH}" | tee -a "${SUMMARY_FILE}"
    echo "Log: ${bootstrap_log}" | tee -a "${SUMMARY_FILE}"
    echo "" | tee -a "${SUMMARY_FILE}"
}

run_benchmark() {
    local name="$1"
    shift
    local extra_args=("$@")

    local log_file="${BENCHMARK_DIR}/${name}-${TIMESTAMP}.log"
    echo "--- [$(date +%H:%M:%S)] Running: ${name} ---" | tee -a "${SUMMARY_FILE}"
    echo "  Args: ${extra_args[*]}" | tee -a "${SUMMARY_FILE}"

    ray stop --force 2>/dev/null || true
    sleep 3

    timeout "${RUN_TIMEOUT_SECONDS}" python -u axrl/controller/run_grpo_controller.py \
        "${COMMON_ARGS[@]}" \
        --debug_train=true \
        --grpo.max_global_updates="${ROLLOUT_BOOTSTRAP_MAX_UPDATES}" \
        "${extra_args[@]}" \
        > "${log_file}" 2>&1
    local exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        echo "  Status: SUCCESS" | tee -a "${SUMMARY_FILE}"
    elif [ ${exit_code} -eq 124 ]; then
        echo "  Status: TIMEOUT" | tee -a "${SUMMARY_FILE}"
    else
        echo "  Status: FAILED (exit ${exit_code})" | tee -a "${SUMMARY_FILE}"
    fi

    # Extract peak memory (max across ranks, step 1 = warmup excluded if possible)
    local max_peak_alloc
    max_peak_alloc=$(grep "peak_mem_gbs" "${log_file}" | grep -oP '[0-9.]+' | sort -rn | head -1 2>/dev/null || echo "N/A")
    local max_peak_reserved
    max_peak_reserved=$(grep "peak_mem_reserved_gbs" "${log_file}" | grep -oP '[0-9.]+' | sort -rn | head -1 2>/dev/null || echo "N/A")

    # Extract step time (first complete train step from rank 0)
    local step_time
    step_time=$(grep "rk0_8.*cpu_time_s" "${log_file}" | grep -oP '[0-9.]+' | head -1 2>/dev/null || echo "N/A")

    echo "  Max Peak Alloc: ${max_peak_alloc} GB" | tee -a "${SUMMARY_FILE}"
    echo "  Max Peak Reserved: ${max_peak_reserved} GB" | tee -a "${SUMMARY_FILE}"
    echo "  Step time (rk0): ${step_time}s" | tee -a "${SUMMARY_FILE}"
    echo "  Log: ${log_file}" | tee -a "${SUMMARY_FILE}"
    echo "" | tee -a "${SUMMARY_FILE}"

    return ${exit_code}
}

ensure_rollout_snapshot

# =====================================================================
# Phase 1: Memory config comparison (mbs=1)
# =====================================================================

# Exp 1: Baseline - full recompute 1 layer, optimizer offload
run_benchmark "exp01-baseline-mbs1" \
    --megatron_worker.train_micro_batch_size=1

# Exp 2: No recompute at all
run_benchmark "exp02-no-recompute-mbs1" \
    --megatron_worker.train_micro_batch_size=1 \
    --megatron_worker.recompute_granularity=null \
    --megatron_worker.recompute_method=null \
    --megatron_worker.recompute_num_layers=null

# Exp 3: Full recompute ALL 48 layers (max activation memory saving)
# Reference: uniform recompute divides layers into equal groups
run_benchmark "exp03-full-recompute-all48-mbs1" \
    --megatron_worker.train_micro_batch_size=1 \
    --megatron_worker.recompute_granularity=full \
    --megatron_worker.recompute_method=uniform \
    --megatron_worker.recompute_num_layers=48

# Exp 4: Selective recompute - core_attn only
# Reference: verl default, good balance of memory vs speed
run_benchmark "exp04-selective-core-attn-mbs1" \
    --megatron_worker.train_micro_batch_size=1 \
    --megatron_worker.recompute_granularity=selective \
    --megatron_worker.recompute_method=null \
    --megatron_worker.recompute_num_layers=null \
    '--megatron_worker.recompute_modules=["core_attn"]'

# Exp 5: Selective recompute - core_attn + moe_act
run_benchmark "exp05-selective-attn-moeact-mbs1" \
    --megatron_worker.train_micro_batch_size=1 \
    --megatron_worker.recompute_granularity=selective \
    --megatron_worker.recompute_method=null \
    --megatron_worker.recompute_num_layers=null \
    '--megatron_worker.recompute_modules=["core_attn","moe_act"]'

# Exp 6: No optimizer offload (shows optimizer offload memory impact)
run_benchmark "exp06-no-optim-offload-mbs1" \
    --megatron_worker.train_micro_batch_size=1 \
    --megatron_worker.optimizer.optimizer_cpu_offload=false \
    --megatron_worker.optimizer.optimizer_offload_fraction=0.0

# Exp 7: CPU activation offloading
run_benchmark "exp07-cpu-offload-mbs1" \
    --megatron_worker.train_micro_batch_size=1 \
    --megatron_worker.cpu_offloading=true \
    --megatron_worker.cpu_offloading_num_layers=0

# Exp 8: CPU activation offloading + double buffering
run_benchmark "exp08-cpu-offload-doublebuf-mbs1" \
    --megatron_worker.train_micro_batch_size=1 \
    --megatron_worker.cpu_offloading=true \
    --megatron_worker.cpu_offloading_num_layers=0 \
    --megatron_worker.cpu_offloading_double_buffering=true

echo "=== Phase 1 Complete. Starting Phase 2: Max MBS Testing ===" | tee -a "${SUMMARY_FILE}"

# =====================================================================
# Phase 2: Increase micro batch size
# =====================================================================

# Baseline configs with increasing mbs
run_benchmark "exp09-baseline-mbs2" \
    --megatron_worker.train_micro_batch_size=2

run_benchmark "exp10-baseline-mbs4" \
    --megatron_worker.train_micro_batch_size=4

run_benchmark "exp11-baseline-mbs8" \
    --megatron_worker.train_micro_batch_size=8

# Full recompute all 48 layers with increasing mbs (expected to handle more)
run_benchmark "exp12-full-recompute-all48-mbs2" \
    --megatron_worker.train_micro_batch_size=2 \
    --megatron_worker.recompute_granularity=full \
    --megatron_worker.recompute_method=uniform \
    --megatron_worker.recompute_num_layers=48

run_benchmark "exp13-full-recompute-all48-mbs4" \
    --megatron_worker.train_micro_batch_size=4 \
    --megatron_worker.recompute_granularity=full \
    --megatron_worker.recompute_method=uniform \
    --megatron_worker.recompute_num_layers=48

run_benchmark "exp14-full-recompute-all48-mbs8" \
    --megatron_worker.train_micro_batch_size=8 \
    --megatron_worker.recompute_granularity=full \
    --megatron_worker.recompute_method=uniform \
    --megatron_worker.recompute_num_layers=48

echo "=== All Benchmarks Complete at $(date) ===" | tee -a "${SUMMARY_FILE}"
