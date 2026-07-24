#!/usr/bin/bash
# Parallelism Benchmark for Qwen3-30B-A3B MoE on 8x H200
# Tests 23 configurations varying TP, PP, VPP, CP, EP, ETP
#
# Fixed params (from memory benchmark results):
#   train_micro_batch_size=4, recompute_granularity=full, recompute_method=uniform,
#   recompute_num_layers=1, optimizer_cpu_offload=true, optimizer_offload_fraction=1.0
#
# Constraints (from Megatron-LM parallel_state.py):
#   - DP * TP * PP * CP = 8 (world size = 8 GPUs)
#   - 8 % (ETP * EP * PP) == 0 (expert world size divisibility)
#   - ETP defaults to TP if etp_size=None; set ETP=1 explicitly to decouple
#   - VPP requires PP > 1; 48 / (PP * VPP) must be integer
#   - 128 % EP == 0 (num_experts divisible by EP)

set -uo pipefail

BENCHMARK_DIR="${AXRL_OUTPUT_DIR}/benchmark/parallelism-results"
mkdir -p "${BENCHMARK_DIR}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SUMMARY_FILE="${BENCHMARK_DIR}/summary-${TIMESTAMP}.txt"
CONFIG_PATH="axis_recipe/moe/grpo_config.yaml"
ROLLOUT_SAVE_FILENAME="valid_rollouts-moe-parallel-debug.zst"
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
    --megatron_worker.train_micro_batch_size=4
    --megatron_worker.recompute_granularity=full
    --megatron_worker.recompute_method=uniform
    --megatron_worker.recompute_num_layers=1
    --megatron_worker.optimizer.optimizer_cpu_offload=true
    --megatron_worker.optimizer.optimizer_offload_fraction=1.0
)

echo "=== Parallelism Benchmark Started at $(date) ===" | tee "${SUMMARY_FILE}"
echo "Model: Qwen3-30B-A3B-Instruct-2507 (48 layers, 128 experts, top-8)" | tee -a "${SUMMARY_FILE}"
echo "Hardware: 8x NVIDIA H200 (140GB)" | tee -a "${SUMMARY_FILE}"
echo "Fixed: MBS=4, full recompute 1 layer, optimizer offload 100%" | tee -a "${SUMMARY_FILE}"
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

    # Extract metrics from log
    local max_peak_alloc
    max_peak_alloc=$(grep "peak_mem_gbs:" "${log_file}" 2>/dev/null | sed 's/.*peak_mem_gbs: //' | sort -rn | head -1)
    local max_peak_reserved
    max_peak_reserved=$(grep "peak_mem_reserved_gbs:" "${log_file}" 2>/dev/null | sed 's/.*peak_mem_reserved_gbs: //' | sort -rn | head -1)
    local step_time
    step_time=$(grep "cpu_time_s:" "${log_file}" 2>/dev/null | sed 's/.*cpu_time_s: //' | tail -1)

    echo "  Peak Alloc: ${max_peak_alloc:-N/A} GB" | tee -a "${SUMMARY_FILE}"
    echo "  Peak Reserved: ${max_peak_reserved:-N/A} GB" | tee -a "${SUMMARY_FILE}"
    echo "  Step time: ${step_time:-N/A}s" | tee -a "${SUMMARY_FILE}"
    echo "  Log: ${log_file}" | tee -a "${SUMMARY_FILE}"
    echo "" | tee -a "${SUMMARY_FILE}"

    return ${exit_code}
}

ensure_rollout_snapshot

# =====================================================================
# Group 1: Varying TP (tensor parallel)
# =====================================================================

# Exp 01: Baseline — DP=2, TP=1, PP=1, CP=4, EP=4, ETP=1
run_benchmark "exp01-baseline-dp2-tp1-cp4-ep4" \
    --megatron_worker.dp_size=2 \
    --megatron_worker.tp_size=1 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=4 \
    --megatron_worker.ep_size=4 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# Exp 02: TP=2, high DP — DP=4, TP=2, PP=1, CP=1, EP=4, ETP=1
run_benchmark "exp02-dp4-tp2-cp1-ep4" \
    --megatron_worker.dp_size=4 \
    --megatron_worker.tp_size=2 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=1 \
    --megatron_worker.ep_size=4 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# Exp 03: TP=2 + CP=2 — DP=2, TP=2, PP=1, CP=2, EP=4, ETP=1
run_benchmark "exp03-dp2-tp2-cp2-ep4" \
    --megatron_worker.dp_size=2 \
    --megatron_worker.tp_size=2 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=2 \
    --megatron_worker.ep_size=4 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# Exp 04: TP=4 — DP=2, TP=4, PP=1, CP=1, EP=4, ETP=1
run_benchmark "exp04-dp2-tp4-cp1-ep4" \
    --megatron_worker.dp_size=2 \
    --megatron_worker.tp_size=4 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=1 \
    --megatron_worker.ep_size=4 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# Exp 05: TP=4 + CP=2 — DP=1, TP=4, PP=1, CP=2, EP=4, ETP=1
run_benchmark "exp05-dp1-tp4-cp2-ep4" \
    --megatron_worker.dp_size=1 \
    --megatron_worker.tp_size=4 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=2 \
    --megatron_worker.ep_size=4 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# Exp 06: TP=8 — DP=1, TP=8, PP=1, CP=1, EP=8, ETP=1
run_benchmark "exp06-dp1-tp8-cp1-ep8" \
    --megatron_worker.dp_size=1 \
    --megatron_worker.tp_size=8 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=1 \
    --megatron_worker.ep_size=8 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# =====================================================================
# Group 2: Varying CP (context parallel)
# =====================================================================

# Exp 07: Pure DP — DP=8, TP=1, PP=1, CP=1, EP=8, ETP=1
run_benchmark "exp07-dp8-tp1-cp1-ep8" \
    --megatron_worker.dp_size=8 \
    --megatron_worker.tp_size=1 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=1 \
    --megatron_worker.ep_size=8 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# Exp 08: CP=2, high DP — DP=4, TP=1, PP=1, CP=2, EP=4, ETP=1
run_benchmark "exp08-dp4-tp1-cp2-ep4" \
    --megatron_worker.dp_size=4 \
    --megatron_worker.tp_size=1 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=2 \
    --megatron_worker.ep_size=4 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# Exp 09: Max CP=8 — DP=1, TP=1, PP=1, CP=8, EP=8, ETP=1
run_benchmark "exp09-dp1-tp1-cp8-ep8" \
    --megatron_worker.dp_size=1 \
    --megatron_worker.tp_size=1 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=8 \
    --megatron_worker.ep_size=8 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# =====================================================================
# Group 3: Varying PP (pipeline parallel)
# =====================================================================

# Exp 10: PP=2, high DP — DP=4, TP=1, PP=2, CP=1, EP=4, ETP=1
run_benchmark "exp10-dp4-tp1-pp2-cp1-ep4" \
    --megatron_worker.dp_size=4 \
    --megatron_worker.tp_size=1 \
    --megatron_worker.pp_size=2 \
    --megatron_worker.cp_size=1 \
    --megatron_worker.ep_size=4 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# Exp 11: PP=2 + CP=2 — DP=2, TP=1, PP=2, CP=2, EP=2, ETP=1
run_benchmark "exp11-dp2-tp1-pp2-cp2-ep2" \
    --megatron_worker.dp_size=2 \
    --megatron_worker.tp_size=1 \
    --megatron_worker.pp_size=2 \
    --megatron_worker.cp_size=2 \
    --megatron_worker.ep_size=2 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# Exp 12: PP=2 + TP=2 — DP=2, TP=2, PP=2, CP=1, EP=2, ETP=1
run_benchmark "exp12-dp2-tp2-pp2-cp1-ep2" \
    --megatron_worker.dp_size=2 \
    --megatron_worker.tp_size=2 \
    --megatron_worker.pp_size=2 \
    --megatron_worker.cp_size=1 \
    --megatron_worker.ep_size=2 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# Exp 13: PP=4 — DP=2, TP=1, PP=4, CP=1, EP=2, ETP=1
run_benchmark "exp13-dp2-tp1-pp4-cp1-ep2" \
    --megatron_worker.dp_size=2 \
    --megatron_worker.tp_size=1 \
    --megatron_worker.pp_size=4 \
    --megatron_worker.cp_size=1 \
    --megatron_worker.ep_size=2 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# Exp 14: PP=4 + CP=2 — DP=1, TP=1, PP=4, CP=2, EP=1, ETP=1
run_benchmark "exp14-dp1-tp1-pp4-cp2-ep1" \
    --megatron_worker.dp_size=1 \
    --megatron_worker.tp_size=1 \
    --megatron_worker.pp_size=4 \
    --megatron_worker.cp_size=2 \
    --megatron_worker.ep_size=1 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# =====================================================================
# Group 4: PP + VPP (virtual pipeline parallel)
# Known limitation: VPP requires TP=1 (VPP+TP>1 fails due to Megatron
# core bug with interleaved schedule + sequence_parallel). VPP>=4 also
# fails. Only VPP=2 and VPP=3 with TP=1 are valid.
# 48 / (PP * VPP) must be integer.
# =====================================================================

# Exp 15: VPP=2, PP=2 — DP=4, TP=1, PP=2, CP=1, EP=4, ETP=1, VPP=2
# 4 virtual stages, 12 layers each
run_benchmark "exp15-dp4-tp1-pp2-cp1-ep4-vpp2" \
    --megatron_worker.dp_size=4 \
    --megatron_worker.tp_size=1 \
    --megatron_worker.pp_size=2 \
    --megatron_worker.cp_size=1 \
    --megatron_worker.ep_size=4 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=2

# Exp 16: VPP=3, PP=4 — DP=2, TP=1, PP=4, CP=1, EP=2, ETP=1, VPP=3
# 12 virtual stages, 4 layers each
run_benchmark "exp16-dp2-tp1-pp4-cp1-ep2-vpp3" \
    --megatron_worker.dp_size=2 \
    --megatron_worker.tp_size=1 \
    --megatron_worker.pp_size=4 \
    --megatron_worker.cp_size=1 \
    --megatron_worker.ep_size=2 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=3

# Exp 17: VPP=2, PP=4 — DP=2, TP=1, PP=4, CP=1, EP=2, ETP=1, VPP=2
# 8 virtual stages, 6 layers each
run_benchmark "exp17-dp2-tp1-pp4-cp1-ep2-vpp2" \
    --megatron_worker.dp_size=2 \
    --megatron_worker.tp_size=1 \
    --megatron_worker.pp_size=4 \
    --megatron_worker.cp_size=1 \
    --megatron_worker.ep_size=2 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=2

# =====================================================================
# Group 5: Varying EP (expert parallel)
# =====================================================================

# Exp 18: EP=1 (all 128 experts per rank — may OOM)
run_benchmark "exp18-dp2-tp1-cp4-ep1" \
    --megatron_worker.dp_size=2 \
    --megatron_worker.tp_size=1 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=4 \
    --megatron_worker.ep_size=1 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# Exp 19: EP=2 (64 experts per rank)
run_benchmark "exp19-dp2-tp1-cp4-ep2" \
    --megatron_worker.dp_size=2 \
    --megatron_worker.tp_size=1 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=4 \
    --megatron_worker.ep_size=2 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# Exp 20: EP=8 (16 experts per rank)
run_benchmark "exp20-dp2-tp1-cp4-ep8" \
    --megatron_worker.dp_size=2 \
    --megatron_worker.tp_size=1 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=4 \
    --megatron_worker.ep_size=8 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# =====================================================================
# Group 6: ETP (expert tensor parallel)
# =====================================================================

# Exp 21: ETP=2, TP=2 — DP=2, TP=2, PP=1, CP=2, EP=2, ETP=2
run_benchmark "exp21-dp2-tp2-cp2-ep2-etp2" \
    --megatron_worker.dp_size=2 \
    --megatron_worker.tp_size=2 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=2 \
    --megatron_worker.ep_size=2 \
    --megatron_worker.etp_size=2 \
    --megatron_worker.vpp_size=null

# Exp 22: TP=2, ETP=1 (decoupled) — DP=4, TP=2, PP=1, CP=1, EP=2, ETP=1
run_benchmark "exp22-dp4-tp2-cp1-ep2-etp1" \
    --megatron_worker.dp_size=4 \
    --megatron_worker.tp_size=2 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=1 \
    --megatron_worker.ep_size=2 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

# =====================================================================
# Group 7: Combined optimized
# =====================================================================

# Exp 23: TP=2 + CP=4 — DP=1, TP=2, PP=1, CP=4, EP=4, ETP=1
run_benchmark "exp23-dp1-tp2-cp4-ep4" \
    --megatron_worker.dp_size=1 \
    --megatron_worker.tp_size=2 \
    --megatron_worker.pp_size=1 \
    --megatron_worker.cp_size=4 \
    --megatron_worker.ep_size=4 \
    --megatron_worker.etp_size=1 \
    --megatron_worker.vpp_size=null

echo "=== All Parallelism Benchmarks Complete at $(date) ===" | tee -a "${SUMMARY_FILE}"
