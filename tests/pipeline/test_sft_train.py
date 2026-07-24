from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import pytest
import ray
import torch

from axrl.configs import DatasetConfig, MegatronWorkerConfig, MetricLoggerConfig, ModelConfig
from axrl.data import Conversation, GenerationState
from axrl.datasets import register_dataset
from axrl.datasets.base_dataset import BaseDataset
from axrl.example.config_examples import CONV_EXAMPLE_PATH, get_megatron_trainer_config
from axrl.pipeline import ControllerConfig, PipelineController, PipelineExperimentConfig
from axrl.ray import ray_utils
from axrl.recipe.base_recipe import BaseRecipe
from axrl.utils import setup_logger, zst_utils

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


MODEL_NAME = "Qwen/Qwen3-0.6B"
TRAIN_DATASET_NAME = "pipeline_conv_example_sft_train"
EVAL_DATASET_NAME = "pipeline_conv_example_sft_eval"

pytestmark = pytest.mark.usefixtures("ray_runtime")


class _ConvExampleSftDataset(BaseDataset):
    start: int
    stop: int
    source: str

    def initialize(self) -> None:
        assert CONV_EXAMPLE_PATH.exists(), f"Missing example dataset at {CONV_EXAMPLE_PATH}"
        conversations: list[Conversation] = zst_utils.load_zst(CONV_EXAMPLE_PATH)
        selected = [conv.deep_copy() for conv in conversations[self.start : self.stop]]
        assert selected, f"No conversations selected from {CONV_EXAMPLE_PATH} for {self.source}."
        for index, conv in enumerate(selected):
            if not hasattr(conv, "extra"):
                conv.extra = {}
            if not hasattr(conv, "gen_state"):
                conv.gen_state = GenerationState()
            conv.conversation_id = f"{self.source}:{index}"
            conv.source = self.source
        self._conversations = selected
        self._label = [str(conv.messages[-1].content) for conv in selected]
        self._score_history = [[] for _ in selected]
        self._length_history = [[] for _ in selected]
        self._conversation_id_to_index = {conv.conversation_id: index for index, conv in enumerate(selected)}
        self._check_initialized()


class _ConvExampleSftTrainDataset(_ConvExampleSftDataset):
    start = 0
    stop = 128
    source = TRAIN_DATASET_NAME


class _ConvExampleSftEvalDataset(_ConvExampleSftDataset):
    start = -128
    stop = 16710
    source = EVAL_DATASET_NAME


def _register_dataset_once(name: str, dataset_cls: type[BaseDataset]) -> None:
    with contextlib.suppress(ValueError):
        register_dataset(name, dataset_cls)


_register_dataset_once(TRAIN_DATASET_NAME, _ConvExampleSftTrainDataset)
_register_dataset_once(EVAL_DATASET_NAME, _ConvExampleSftEvalDataset)


@pytest.fixture
def ray_runtime() -> Iterator[None]:
    setup_logger("info")
    ray_utils.restart()
    try:
        yield
    finally:
        ray_utils.stop()


def _skip_if_not_enough_gpus(required_gpus: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < required_gpus:
        pytest.skip(f"Need >= {required_gpus} GPUs, found {torch.cuda.device_count()}.")


def _sft_curve_config(tmp_path: Path) -> PipelineExperimentConfig:
    model_config = ModelConfig(name=MODEL_NAME, seq_length=2048)
    megatron_config: MegatronWorkerConfig = get_megatron_trainer_config(
        tp_size=2,
        dp_size=1,
        pp_size=2,
        cp_size=2,
        model_config=model_config,
    )
    megatron_config.global_batch_size = 32
    megatron_config.train_micro_batch_size = 1
    megatron_config.eval_micro_batch_size = 8
    megatron_config.log_every_k_steps = 1
    megatron_config.use_dynamic_batch_size = True
    megatron_config.num_epochs = 2
    megatron_config.checkpoint_dir = str(tmp_path / "checkpoints")
    return PipelineExperimentConfig(
        controller=ControllerConfig(run_mode="sft_train"),
        megatron_worker=megatron_config,
        train_datasets=[DatasetConfig(name=TRAIN_DATASET_NAME)],
        test_datasets=[DatasetConfig(name=EVAL_DATASET_NAME)],
        logger=MetricLoggerConfig(logger_type="console"),
    )


def test_pipeline_sft_train_reduces_conv_example_validation_loss(tmp_path: Path) -> None:
    config = _sft_curve_config(tmp_path)
    _skip_if_not_enough_gpus(config.megatron_worker.world_size())
    assert config.megatron_worker.model.get_full_path().exists(), f"Model path does not exist: {config.megatron_worker.model.get_full_path()}"

    controller = PipelineController(config, BaseRecipe(config))

    async def run() -> tuple[float, list[dict[str, float]]]:
        try:
            await controller.initialize()
            await controller.load_checkpoint_if_existed()
            assert controller.megatron_worker is not None
            eval_dataset = controller.build_sft_eval_samples()
            assert eval_dataset is not None, "SFT smoke test requires eval samples from CONV_EXAMPLE_PATH."
            initial_eval_metrics = controller.megatron_worker.eval(controller.global_step, eval_dataset)
            curve = await controller.run_sft_train()
            return float(initial_eval_metrics["eval/loss"]), curve
        finally:
            controller.shutdown()
            if ray.is_initialized():
                ray_utils.stop()

    initial_eval_loss, curve = asyncio.run(run())
    assert len(curve) == config.megatron_worker.num_epochs
    assert curve[-1]["global_step"] == config.megatron_worker.num_epochs * 4
    assert curve[-1]["train_loss"] < curve[0]["train_loss"]
    assert curve[-1]["eval_loss"] < initial_eval_loss
    assert curve[-1]["eval_loss"] < 1.1
