"""End-to-end FP8 mismatch regression test."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import pytest
import torch

from axrl.pipeline import PipelineController
from tests.moe.pipeline_moe_helpers import MoeMathRecipe, make_moe_pipeline_config

if TYPE_CHECKING:
    from pathlib import Path

    from axrl.pipeline import PipelineExperimentConfig

BF16_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
FP8_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
NUM_GPUS_REQUIRED = 8
PROJECT_NAME = "2026-04-13-MoE-FP8-Mismatch"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fp8MismatchCase:
    precision: Literal["bf16", "fp8"]
    enable_r3: bool

    @property
    def run_name(self) -> str:
        suffix = "r3" if self.enable_r3 else "baseline"
        return f"{self.precision}-{suffix}"

    @property
    def rollout_model_name(self) -> str:
        return FP8_MODEL_NAME if self.precision == "fp8" else BF16_MODEL_NAME

    @property
    def megatron_fp8(self) -> Literal["e4m3"] | None:
        return "e4m3" if self.precision == "fp8" else None

    @property
    def baseline_ref(self) -> str | None:
        return None if self.run_name == "bf16-baseline" else "bf16-baseline"

    def result_path(self, root: Path) -> Path:
        return root / "results" / "log" / f"{self.run_name}.zst"


@dataclass(frozen=True)
class Fp8MismatchMetrics:
    mean_kl: float
    std_kl: float
    max_kl: float
    rollout_throughput: float
    mcore_runtime_seconds: float
    score_all: float


CASES = (
    Fp8MismatchCase("bf16", enable_r3=False),
    Fp8MismatchCase("fp8", enable_r3=False),
    Fp8MismatchCase("bf16", enable_r3=True),
    Fp8MismatchCase("fp8", enable_r3=True),
)


def _format_score(value: float) -> str:
    return f"{value:.6f}"


def _metric(metrics: dict[str, float] | None, key: str) -> float:
    assert metrics is not None and key in metrics, f"Missing metric {key}"
    value = float(metrics[key])
    assert math.isfinite(value), f"Metric {key} is not finite: {value}"
    return value


def _extract_metrics(run_name: str, result: object) -> Fp8MismatchMetrics:
    from axrl.metrics.report_mismatch import MismatchRunResult

    assert isinstance(result, MismatchRunResult), f"Unexpected benchmark payload: {type(result).__name__}"
    assert result.success, f"{run_name} was marked unsuccessful"
    assert result.rollout_throughput is not None, f"{run_name} missing rollout throughput"
    assert result.mcore is not None and result.mcore.end_to_end_time_sec is not None, f"{run_name} missing MCore runtime"

    return Fp8MismatchMetrics(
        mean_kl=_metric(result.mismatch_metrics, f"KL-Mismatch/mean_KL1_{run_name}"),
        std_kl=_metric(result.mismatch_metrics, f"KL-Mismatch/std_KL1_{run_name}"),
        max_kl=_metric(result.mismatch_metrics, f"KL-Mismatch/max_KL1_{run_name}"),
        rollout_throughput=float(result.rollout_throughput),
        mcore_runtime_seconds=float(result.mcore.end_to_end_time_sec),
        score_all=_metric(result.response_metrics, "Mismatch-Rollout/score__all"),
    )


def _make_config(case: Fp8MismatchCase, output_dir: Path) -> PipelineExperimentConfig:
    config = make_moe_pipeline_config(
        model_name=BF16_MODEL_NAME,
        max_length=16_384,
        project_name=PROJECT_NAME,
        run_name=case.run_name,
        output_dir=str(output_dir),
        baseline_name=case.baseline_ref,
    )

    config.megatron_worker.model.name = BF16_MODEL_NAME
    config.megatron_worker.dp_size = 1
    config.megatron_worker.tp_size = 2
    config.megatron_worker.cp_size = 4
    config.megatron_worker.pp_size = 1
    config.megatron_worker.ep_size = 8
    config.megatron_worker.etp_size = 1
    config.megatron_worker.enable_routing_replay = case.enable_r3
    config.megatron_worker.eval_micro_batch_size = 1
    config.megatron_worker.enable_fp32_lm_head = False
    config.megatron_worker.global_batch_size = 32
    config.megatron_worker.fp8 = case.megatron_fp8

    config.rollout_worker.model.name = case.rollout_model_name
    config.rollout_worker.engine_type = "sglang"
    config.rollout_worker.tp_size = NUM_GPUS_REQUIRED
    config.rollout_worker.pp_size = 1
    config.rollout_worker.ep_size = 8
    config.rollout_worker.enable_routing_replay = case.enable_r3
    config.rollout_worker.num_workers = 1
    config.rollout_worker.max_running_requests = 2048
    config.rollout_worker.gpu_memory_utilization = 0.7

    config.online_rl_train.filter_zero_std = False
    config.online_rl_train.model_sync_every_n_global_updates = 4
    config.online_rl_train.batch_rollout_for_n_global_updates = 4
    config.controller.max_running_requests = config.rollout_worker.max_running_requests

    config.logger.project_name = PROJECT_NAME
    config.logger.group_name = case.run_name
    return config


async def _run_case(case: Fp8MismatchCase, root: Path) -> None:
    from axrl.ray import ray_utils

    (root / "results" / "log").mkdir(parents=True, exist_ok=True)
    (root / "results" / "figs").mkdir(parents=True, exist_ok=True)

    ray_utils.restart()
    try:
        config = _make_config(case, root)
        controller = PipelineController(config, MoeMathRecipe(config))
        await controller.start()
    finally:
        ray_utils.stop()


async def _run_fp8_mismatch(output_dir: Path) -> Path:
    root = output_dir / "mismatch_test-fp8"
    root.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        await _run_case(case, root)
    return root


def _format_fp8_score_snapshot(metrics_by_case: dict[str, Fp8MismatchMetrics]) -> str:
    rows = [
        "FP8 mismatch score snapshot:",
        "| precision | R3 | mean KL1 | std KL1 | max KL1 | score |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in CASES:
        metrics = metrics_by_case[case.run_name]
        rows.append(
            f"| {case.precision} | {'on' if case.enable_r3 else 'off'} "
            f"| {_format_score(metrics.mean_kl)} "
            f"| {_format_score(metrics.std_kl)} "
            f"| {_format_score(metrics.max_kl)} "
            f"| {_format_score(metrics.score_all)} |"
        )
    return "\n".join(rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp8_mismatch_end_to_end_matches_expected_kl_ranges(tmp_path: Path) -> None:
    """Run the FP8 mismatch matrix directly through ``PipelineController``.

    The test compares BF16 and FP8 rollout/MCore execution, with R3 on and
    off, without invoking any shell script. ``regression-test-report.md`` and
    its Git history currently have ``N/A`` in the FP8 mismatch section, so the
    default table below is a 2026-05-28 snapshot from a passing full 8xH200
    run. These values may contain rollout and runtime randomness, so the
    assertions below use ranges instead of exact equality.

    | precision | R3 | mean KL1 | std KL1 | max KL1 | score |
    | bf16 | off | 0.012810 | 0.041754 | 3.675469 | 0.000000 |
    | fp8  | off | 0.024618 | 0.070255 | 2.722107 | 0.015625 |
    | bf16 | on  | 0.007303 | 0.021468 | 0.572372 | 0.015625 |
    | fp8  | on  | 0.020452 | 0.056842 | 3.035720 | 0.031250 |
    """
    if torch.cuda.device_count() != NUM_GPUS_REQUIRED:
        pytest.skip(f"Need exactly {NUM_GPUS_REQUIRED} visible GPUs")

    root = asyncio.run(_run_fp8_mismatch(tmp_path / "axrl-output"))
    missing = [case.result_path(root) for case in CASES if not case.result_path(root).is_file()]
    assert not missing, "Missing FP8 mismatch artifacts:\n" + "\n".join(str(path) for path in missing)

    from axrl.utils import zst_utils

    metrics_by_case = {case.run_name: _extract_metrics(case.run_name, zst_utils.load_zst(case.result_path(root))) for case in CASES}
    score_snapshot = _format_fp8_score_snapshot(metrics_by_case)
    logger.info("\n%s", score_snapshot)
    print(score_snapshot, flush=True)

    bf16_baseline = metrics_by_case["bf16-baseline"]
    fp8_baseline = metrics_by_case["fp8-baseline"]
    bf16_r3 = metrics_by_case["bf16-r3"]
    fp8_r3 = metrics_by_case["fp8-r3"]

    assert 0.010 < bf16_baseline.mean_kl < 0.020, score_snapshot
    assert 0.020 < fp8_baseline.mean_kl < 0.035, score_snapshot
    assert 0.004 < bf16_r3.mean_kl < 0.012, score_snapshot
    assert 0.018 < fp8_r3.mean_kl < 0.035, score_snapshot

    assert 0.030 < bf16_baseline.std_kl < 0.060, score_snapshot
    assert 0.050 < fp8_baseline.std_kl < 0.100, score_snapshot
    assert 0.015 < bf16_r3.std_kl < 0.035, score_snapshot
    assert 0.045 < fp8_r3.std_kl < 0.090, score_snapshot

    assert bf16_r3.mean_kl < bf16_baseline.mean_kl, score_snapshot
    assert fp8_baseline.mean_kl > bf16_baseline.mean_kl, score_snapshot
    assert fp8_r3.mean_kl > bf16_r3.mean_kl, score_snapshot
