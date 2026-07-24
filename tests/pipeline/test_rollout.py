from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from axrl.configs import AXRL_DIR, DatasetConfig, MegatronWorkerConfig, ModelConfig, RolloutWorkerConfig, SamplingConfig
from axrl.pipeline.config import ControllerConfig, PipelineExperimentConfig
from axrl.pipeline.controller import PipelineController
from axrl.ray import ray_utils
from axrl.utils import setup_logger
from tests.pipeline.math_controller import MathRecipe

if TYPE_CHECKING:
    from collections.abc import Iterator


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MIN_GSM8K_ACCURACY = 0.07
pytestmark = pytest.mark.usefixtures("ray_runtime")


@pytest.fixture
def ray_runtime() -> Iterator[None]:
    setup_logger("info")
    ray_utils.restart()
    try:
        yield
    finally:
        ray_utils.stop()


def _assert_gsm8k_eval_assets_exist(model: ModelConfig) -> None:
    assert torch.cuda.is_available(), "Need CUDA for GSM8K eval."
    assert torch.cuda.device_count() >= 1, f"Need >= 1 GPU, found {torch.cuda.device_count()}."
    assert model.get_full_path().exists(), f"Model path does not exist: {model.get_full_path()}."
    gsm8k_test_path = Path(AXRL_DIR.data) / "openai/gsm8k/main/test-00000-of-00001.parquet"
    assert gsm8k_test_path.exists(), f"GSM8K test parquet does not exist: {gsm8k_test_path}."


def _rollout_eval_config() -> PipelineExperimentConfig:
    model = ModelConfig(name=MODEL_NAME, seq_length=2048, trust_remote_code=True)
    sampling_config = SamplingConfig(temperature=0.0, top_p=1.0, max_total_tokens=2048)
    return PipelineExperimentConfig(
        controller=ControllerConfig(
            run_mode="eval_only",
            colocated=True,
            num_rollout_actors=8,
            num_cpus_per_actor=2,
            max_running_requests=256,
            save_eval_rollouts=True,
        ),
        test_datasets=[DatasetConfig(name="openai/gsm8k/test", eval_num_rollouts_per_prompt=1)],
        eval_sampling_config=sampling_config,
        megatron_worker=MegatronWorkerConfig(model=model),
        rollout_worker=RolloutWorkerConfig(
            model=model,
            sampling_config=sampling_config,
            tp_size=1,
            pp_size=1,
            num_workers=1,
            max_running_requests=256,
            max_running_requests_eval=256,
            load_dummy_weights=False,
            gpu_memory_utilization=0.7,
            enable_metrics=False,
        ),
    )


def test_pipeline_eval_only_gsm8k_accuracy() -> None:
    config = _rollout_eval_config()
    _assert_gsm8k_eval_assets_exist(config.rollout_worker.model)
    controller = PipelineController(config, MathRecipe(config))

    async def run() -> None:
        try:
            await controller.initialize()
            expected_results = sum(
                len(dataset) * (1 if dataset_config.eval_num_rollouts_per_prompt is None else dataset_config.eval_num_rollouts_per_prompt)
                for dataset, dataset_config in zip(controller.test_datasets, controller.eval_dataset_configs, strict=True)
            )
            results = await controller.run_eval_rollouts()
        finally:
            controller.shutdown()

        assert len(results) == expected_results
        for result in results:
            assert result.conversation.gen_state.session_id is not None
            assert result.metric.score in {0, 1}
            assert result.metric.token_count > 0
            assert result.trace is not None
        accuracy = sum(float(result.metric.score or 0.0) for result in results) / len(results)
        assert accuracy > MIN_GSM8K_ACCURACY

    asyncio.run(run())
