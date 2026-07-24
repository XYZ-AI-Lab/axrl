"""End-to-end R3 benchmark regression test."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pytest
import torch

from axrl.configs import ModelConfig
from axrl.pipeline import PipelineController
from tests.moe.pipeline_moe_helpers import MoeMathRecipe, make_moe_pipeline_config

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from axrl.data.rollout_result import RolloutResult
    from axrl.data.sample import SampleTensorDict
    from axrl.pipeline import PipelineExperimentConfig
    from axrl.utils.tensor_store import TensorHandle

TOPOLOGY = "dp2-tp2-cp2-pp1-ep8"
NUM_GPUS_REQUIRED = 8
PROJECT_NAME = "2026-05-09-MoE-R3-Benchmark"
MAX_SEQ_LENGTH = 16_384
MAX_NEW_TOKENS_WITH_BOUNDARY = 8192
GLOBAL_BATCH_SIZE = 512
PIPELINE_MAX_RUNNING_ROLLOUTS = 512
R3_BENCHMARK_MODEL_NAME = os.environ.get("AXRL_R3_BENCHMARK_MODEL")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkCase:
    forward_path: Literal["magi-merged", "gptmodel-forward"]
    variant: Literal["baseline", "r3", "r3-loss-tokens"]
    use_magi_merged_forward: bool
    deterministic_mode: bool = True

    @property
    def run_name(self) -> str:
        suffix = "" if self.deterministic_mode else "-nondeterministic"
        return f"{TOPOLOGY}-{self.variant}{suffix}"

    @property
    def enable_r3(self) -> bool:
        return self.variant != "baseline"

    @property
    def replay_routing_for_loss_tokens_only(self) -> bool:
        return self.variant == "r3-loss-tokens"

    def result_path(self, root: Path) -> Path:
        return root / self.forward_path / "results" / "log" / f"{self.run_name}.zst"

    def baseline_ref(self) -> str | None:
        if not self.enable_r3:
            return None
        suffix = "" if self.deterministic_mode else "-nondeterministic"
        return f"{TOPOLOGY}-baseline{suffix}"


@dataclass(frozen=True)
class BenchmarkMetrics:
    mean_kl: float
    std_kl: float
    max_kl: float
    rollout_throughput: float
    mcore_runtime_seconds: float
    score_all: float
    token_count_mean: float
    token_count_min: float
    token_count_max: float
    length_finish_ratio: float


def _format_score(value: float) -> str:
    return f"{value:.6f}"


CASES = (
    BenchmarkCase("magi-merged", "baseline", use_magi_merged_forward=True),
    BenchmarkCase("magi-merged", "r3", use_magi_merged_forward=True),
    BenchmarkCase("magi-merged", "r3-loss-tokens", use_magi_merged_forward=True),
    BenchmarkCase("magi-merged", "baseline", use_magi_merged_forward=True, deterministic_mode=False),
    BenchmarkCase("magi-merged", "r3", use_magi_merged_forward=True, deterministic_mode=False),
    BenchmarkCase("gptmodel-forward", "baseline", use_magi_merged_forward=False),
    BenchmarkCase("gptmodel-forward", "r3", use_magi_merged_forward=False),
)


def _metric(metrics: dict[str, float] | None, key: str) -> float:
    assert metrics is not None and key in metrics, f"Missing metric {key}"
    value = float(metrics[key])
    assert math.isfinite(value), f"Metric {key} is not finite: {value}"
    return value


def _extract_metrics(run_name: str, result: object) -> BenchmarkMetrics:
    from axrl.metrics.report_mismatch import MismatchRunResult

    assert isinstance(result, MismatchRunResult), f"Unexpected benchmark payload: {type(result).__name__}"
    assert result.success, f"{run_name} was marked unsuccessful"
    assert result.rollout_throughput is not None, f"{run_name} missing rollout throughput"
    assert result.mcore is not None and result.mcore.end_to_end_time_sec is not None, f"{run_name} missing MCore runtime"

    return BenchmarkMetrics(
        mean_kl=_metric(result.mismatch_metrics, f"KL-Mismatch/mean_KL1_{run_name}"),
        std_kl=_metric(result.mismatch_metrics, f"KL-Mismatch/std_KL1_{run_name}"),
        max_kl=_metric(result.mismatch_metrics, f"KL-Mismatch/max_KL1_{run_name}"),
        rollout_throughput=float(result.rollout_throughput),
        mcore_runtime_seconds=float(result.mcore.end_to_end_time_sec),
        score_all=_metric(result.response_metrics, "Mismatch-Rollout/score__all"),
        token_count_mean=_metric(result.response_metrics, "Mismatch-Rollout/token_count__all"),
        token_count_min=_metric(result.response_metrics, "Mismatch-Rollout/token_count_min__all"),
        token_count_max=_metric(result.response_metrics, "Mismatch-Rollout/token_count_max__all"),
        length_finish_ratio=_metric(result.response_metrics, "Mismatch-Rollout/rollout_finish_reason_length__all"),
    )


def _case_output_dir(root: Path, case: BenchmarkCase) -> Path:
    return root / case.forward_path


def _case_key(case: BenchmarkCase) -> tuple[str, bool, str]:
    return (case.forward_path, case.deterministic_mode, case.variant)


def _r3_benchmark_model_path() -> Path:
    assert R3_BENCHMARK_MODEL_NAME is not None
    return ModelConfig(name=R3_BENCHMARK_MODEL_NAME, seq_length=MAX_SEQ_LENGTH).get_full_path()


def _make_config(
    case: BenchmarkCase,
    output_dir: Path,
    *,
    override_rollouts_if_exists: bool = True,
    rollout_enable_r3: bool | None = None,
) -> PipelineExperimentConfig:
    assert R3_BENCHMARK_MODEL_NAME is not None
    config = make_moe_pipeline_config(
        model_name=R3_BENCHMARK_MODEL_NAME,
        max_length=MAX_SEQ_LENGTH,
        project_name=PROJECT_NAME,
        run_name=case.run_name,
        output_dir=str(output_dir),
        baseline_name=case.baseline_ref(),
    )
    config.mismatch_test.override_rollouts_if_exists = override_rollouts_if_exists

    config.rollout_worker.engine_type = "sglang"
    config.rollout_worker.tp_size = NUM_GPUS_REQUIRED
    config.rollout_worker.pp_size = 1
    config.rollout_worker.ep_size = 8
    config.rollout_worker.enable_routing_replay = case.enable_r3 if rollout_enable_r3 is None else rollout_enable_r3
    config.rollout_worker.num_workers = 1
    config.rollout_worker.max_running_requests = 2048
    config.rollout_worker.gpu_memory_utilization = 0.8
    config.rollout_worker.model.seq_length = MAX_SEQ_LENGTH
    config.rollout_worker.sampling_config.max_new_tokens = MAX_NEW_TOKENS_WITH_BOUNDARY
    config.train_sampling_config.max_new_tokens = MAX_NEW_TOKENS_WITH_BOUNDARY

    config.megatron_worker.model.seq_length = MAX_SEQ_LENGTH
    config.megatron_worker.dp_size = 2
    config.megatron_worker.tp_size = 2
    config.megatron_worker.cp_size = 2
    config.megatron_worker.pp_size = 1
    config.megatron_worker.ep_size = 8
    config.megatron_worker.etp_size = 1
    config.megatron_worker.enable_routing_replay = case.enable_r3
    config.megatron_worker.replay_routing_for_loss_tokens_only = case.replay_routing_for_loss_tokens_only
    config.megatron_worker.use_magi_merged_forward = case.use_magi_merged_forward
    config.megatron_worker.use_language_model_only = True
    config.megatron_worker.deterministic_mode = case.deterministic_mode
    config.megatron_worker.eval_micro_batch_size = 1
    config.megatron_worker.global_batch_size = GLOBAL_BATCH_SIZE

    config.online_rl_train.filter_zero_std = False
    config.online_rl_train.model_sync_every_n_global_updates = 4
    config.online_rl_train.batch_rollout_for_n_global_updates = 4
    config.controller.max_running_requests = PIPELINE_MAX_RUNNING_ROLLOUTS
    config.controller.allow_prefix_merging = True

    config.logger.project_name = PROJECT_NAME
    config.logger.group_name = case.run_name
    return config


async def _run_case(
    case: BenchmarkCase,
    root: Path,
    *,
    shared_rollout_path: Path | None = None,
    routing_payload_path: Path | None = None,
) -> tuple[Path, Path]:
    from axrl.utils.megatron.spike_snapshot_routing import (
        collect_unique_routing_handles_from_batch,
        restore_spike_snapshot_routing,
        save_spike_snapshot_routing,
    )

    output_dir = _case_output_dir(root, case)
    (output_dir / "results" / "log").mkdir(parents=True, exist_ok=True)
    (output_dir / "results" / "figs").mkdir(parents=True, exist_ok=True)
    config = _make_config(
        case,
        output_dir,
        override_rollouts_if_exists=shared_rollout_path is None,
        rollout_enable_r3=True,
    )
    valid_rollout_path = config.mismatch_test.get_valid_rollouts_path(case.run_name)
    if shared_rollout_path is not None:
        valid_rollout_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shared_rollout_path, valid_rollout_path)

    controller = PipelineController(config, MoeMathRecipe(config))
    original_prepare_packed_sample_tensor_dict = controller.prepare_packed_sample_tensor_dict
    original_delete_r3_handles_and_caches = controller.delete_r3_handles_and_caches
    restored_handles: list[TensorHandle] = []
    payload_path = routing_payload_path or (root / "shared-routing-payload.pt")

    async def _prepare_packed_sample_tensor_dict_with_shared_routing(
        group_results: list[list[RolloutResult]],
    ) -> SampleTensorDict:
        batch = await original_prepare_packed_sample_tensor_dict(group_results)
        if routing_payload_path is None:
            assert save_spike_snapshot_routing(batch, payload_path) > 0
        else:
            assert restore_spike_snapshot_routing(batch, payload_path) > 0
            restored_handles.extend(collect_unique_routing_handles_from_batch(batch))
        return batch

    async def _delete_restored_routing_handles(rollout_results: Sequence[RolloutResult], *, clear_trainer_caches: bool = True) -> None:
        if routing_payload_path is None:
            await original_delete_r3_handles_and_caches(rollout_results, clear_trainer_caches=clear_trainer_caches)
        elif clear_trainer_caches:
            assert controller.megatron_worker is not None, "Trainer cache cleanup requires a Megatron worker."
            controller.megatron_worker.clear_r3_caches()
        if restored_handles:
            from axrl.utils import tensor_store as store

            store.delete_batch(restored_handles)
            restored_handles.clear()

    controller.prepare_packed_sample_tensor_dict = _prepare_packed_sample_tensor_dict_with_shared_routing  # type: ignore[method-assign]
    controller.delete_r3_handles_and_caches = _delete_restored_routing_handles  # type: ignore[method-assign]
    await controller.start()
    return valid_rollout_path, payload_path


async def _run_benchmark(output_dir: Path) -> Path:
    from axrl.ray import ray_utils

    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir / "r3-benchmark"
    ray_utils.restart()
    try:
        shared_rollout_path: Path | None = None
        routing_payload_path: Path | None = None
        for case in CASES:
            shared_rollout_path, routing_payload_path = await _run_case(
                case,
                root,
                shared_rollout_path=shared_rollout_path,
                routing_payload_path=routing_payload_path,
            )
    finally:
        ray_utils.stop()
    return root


def _assert_rollouts_include_appended_assistant_boundary(case: BenchmarkCase, root: Path) -> None:
    from axrl.data.rollout_trace import RolloutTrace
    from axrl.utils import zst_utils

    output_dir = _case_output_dir(root, case)
    rollout_path = _make_config(case, output_dir).mismatch_test.get_valid_rollouts_path(case.run_name)
    groups = zst_utils.load_zst(rollout_path)
    traces = [result.trace for group in groups for result in group if result.trace is not None]
    assert traces, f"{case.run_name} did not save rollout traces"
    assert all(isinstance(trace, RolloutTrace) for trace in traces)
    assert any(
        trace.token_trace is not None and any(info.token_type == "assistant_boundary" for info in trace.token_trace.token_infos) for trace in traces
    ), f"{case.run_name} did not append any assistant_boundary token"


def _format_benchmark_score_snapshot(metrics_by_case: dict[tuple[str, bool, str], BenchmarkMetrics]) -> str:
    rows = [
        "R3 benchmark score snapshot:",
        "| forward | deterministic | R3 | mean KL1 | std KL1 | max KL1 | rollout tok/s | MCore sec | score "
        "| token mean | token min | token max | len cap |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in CASES:
        metrics = metrics_by_case[_case_key(case)]
        rows.append(
            f"| {case.forward_path} | {'on' if case.deterministic_mode else 'off'} "
            f"| {'on' if case.enable_r3 else 'off'} "
            f"| {_format_score(metrics.mean_kl)} "
            f"| {_format_score(metrics.std_kl)} "
            f"| {_format_score(metrics.max_kl)} "
            f"| {_format_score(metrics.rollout_throughput)} "
            f"| {_format_score(metrics.mcore_runtime_seconds)} "
            f"| {_format_score(metrics.score_all)} "
            f"| {_format_score(metrics.token_count_mean)} "
            f"| {_format_score(metrics.token_count_min)} "
            f"| {_format_score(metrics.token_count_max)} "
            f"| {_format_score(metrics.length_finish_ratio)} |"
        )
    return "\n".join(rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_r3_benchmark_end_to_end_matches_expected_kl_ranges(tmp_path: Path) -> None:
    """Run the boundary-aware R3 benchmark and check KL ranges.

    2026-06-17 snapshot from a completed 8xH200 pipeline-controller run with
    ``MAX_NEW_TOKENS_WITH_BOUNDARY=8192`` and ``GLOBAL_BATCH_SIZE=512``. These
    values may include rollout and runtime randomness, so the assertions below
    use ranges instead of exact equality.
    Rows were refreshed from the completed 2026-06-30 Qwen3.6 native-FFA run.

    | fwd | det | R3 | mean | std | max | tok/s | mcore | score | tok avg | tok min | tok max | len cap |
    | magi | on | off | 0.012387 | 0.037179 | 5.443064 | 16163.103428 | 378.931849 | 0.122559 | 7562.855957 | 1616 | 8192 | 0.793945 |
    | magi | on | on | 0.006024 | 0.016301 | 1.544254 | 2668736.655991 | 401.009267 | 0.122559 | 7562.855957 | 1616 | 8192 | 0.793945 |
    | magi | on | loss-only | 0.006272 | 0.017315 | 1.487763 | 2605531.278295 | 403.498962 | 0.122559 | 7562.855957 | 1616 | 8192 | 0.793945 |
    | magi | off | off | 0.012387 | 0.037179 | 5.443064 | 2216989.484874 | 510.899662 | 0.122559 | 7562.855957 | 1616 | 8192 | 0.793945 |
    | magi | off | on | 0.006024 | 0.016301 | 1.544254 | 2716162.381064 | 387.458556 | 0.122559 | 7562.855957 | 1616 | 8192 | 0.793945 |
    | gpt | on | off | 0.012400 | 0.037186 | 5.692363 | 2601367.556601 | 361.764927 | 0.122559 | 7562.855957 | 1616 | 8192 | 0.793945 |
    | gpt | on | on | 0.006044 | 0.016359 | 1.316179 | 2489485.947279 | 388.323470 | 0.122559 | 7562.855957 | 1616 | 8192 | 0.793945 |
    """
    if torch.cuda.device_count() != NUM_GPUS_REQUIRED:
        pytest.skip(f"Need exactly {NUM_GPUS_REQUIRED} visible GPUs")
    if R3_BENCHMARK_MODEL_NAME is None:
        pytest.skip("Set AXRL_R3_BENCHMARK_MODEL to run this benchmark.")
    model_path = _r3_benchmark_model_path()
    if not model_path.is_dir():
        pytest.skip(f"R3 benchmark model is not available at {model_path}. Set AXRL_R3_BENCHMARK_MODEL to run this benchmark.")

    root = asyncio.run(_run_benchmark(tmp_path / "axrl-output"))
    missing = [case.result_path(root) for case in CASES if not case.result_path(root).is_file()]
    assert not missing, "Missing R3 benchmark artifacts:\n" + "\n".join(str(path) for path in missing)

    from axrl.utils import zst_utils

    metrics_by_case: dict[tuple[str, bool, str], BenchmarkMetrics] = {}
    for case in CASES:
        _assert_rollouts_include_appended_assistant_boundary(case, root)
        result = zst_utils.load_zst(case.result_path(root))
        metrics_by_case[_case_key(case)] = _extract_metrics(case.run_name, result)

    score_snapshot = _format_benchmark_score_snapshot(metrics_by_case)
    logger.info("\n%s", score_snapshot)
    print(score_snapshot, flush=True)

    case_by_params = {_case_key(case): case for case in CASES}
    for baseline_case in (case for case in CASES if not case.enable_r3):
        r3_case = case_by_params.get((baseline_case.forward_path, baseline_case.deterministic_mode, "r3"))
        if r3_case is None:
            continue

        baseline = metrics_by_case[_case_key(baseline_case)]
        r3 = metrics_by_case[_case_key(r3_case)]

        assert 0.008 < baseline.mean_kl < 0.020, score_snapshot
        assert 0.004 < r3.mean_kl < 0.012, score_snapshot
        assert 0.030 < baseline.std_kl < 0.060, score_snapshot
        assert 0.015 < r3.std_kl < 0.060, score_snapshot

        assert r3.mean_kl < baseline.mean_kl, score_snapshot
        assert r3.mean_kl / baseline.mean_kl < 0.75, score_snapshot

    magi_baseline = metrics_by_case[("magi-merged", True, "baseline")]
    magi_r3 = metrics_by_case[("magi-merged", True, "r3")]
    magi_loss_token_r3 = metrics_by_case[("magi-merged", True, "r3-loss-tokens")]
    assert magi_loss_token_r3.mean_kl < magi_baseline.mean_kl, score_snapshot
    assert abs(magi_loss_token_r3.mean_kl - magi_r3.mean_kl) < 0.005, score_snapshot
