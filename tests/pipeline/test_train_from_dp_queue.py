from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
import torch

from axrl.configs import IGNORE_INDEX, GrpoTrainerConfig, MegatronWorkerConfig, ModelConfig, SftTrainerConfig
from axrl.data import Sample, SampleTensorDict
from axrl.example.config_examples import get_megatron_trainer_config
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.trainer.grpo_trainer import GrpoTrainer
from axrl.trainer.sft_trainer import SftTrainer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from axrl.trainer.base_trainer import BaseTrainer


MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
SEQ_LENGTH = 128
pytestmark = pytest.mark.usefixtures("ray_runtime")


@pytest.fixture
def ray_runtime() -> Iterator[None]:
    ray_utils.restart()
    try:
        yield
    finally:
        ray_utils.stop()


def _skip_if_not_enough_gpus(required_gpus: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < required_gpus:
        pytest.skip(f"Need >= {required_gpus} GPUs, found {torch.cuda.device_count()}.")


def _config() -> MegatronWorkerConfig:
    config = get_megatron_trainer_config(
        tp_size=2,
        dp_size=2,
        pp_size=1,
        cp_size=2,
        model_config=ModelConfig(name=MODEL_NAME, seq_length=SEQ_LENGTH),
    )
    config.global_batch_size = 2
    config.train_micro_batch_size = 1
    config.eval_micro_batch_size = 1
    config.log_every_k_steps = 1
    config.num_epochs = 1
    config.use_dynamic_batch_size = False
    config.inference_only = False
    config.use_magi_merged_forward = False
    config.use_magi_flat_forward = False
    config.padding_sample_length = 4
    return config


def _sample(*, include_grpo_fields: bool, trajectory_id: int = -1) -> Sample:
    input_ids = np.arange(10, 10 + SEQ_LENGTH, dtype=np.int32)
    labels = np.concatenate([input_ids[1:], np.asarray([IGNORE_INDEX], dtype=np.int32)])
    loss_mask = np.zeros(SEQ_LENGTH, dtype=np.bool_)
    loss_mask[8:16] = True
    advantage = np.zeros(SEQ_LENGTH, dtype=np.float32)
    advantage[8:16] = 1.0
    sample = Sample(
        input_ids=input_ids,
        labels=labels,
        loss_mask=loss_mask,
        attention_mask=np.ones(SEQ_LENGTH, dtype=np.bool_),
        position_ids=np.arange(SEQ_LENGTH, dtype=np.int32),
        reward=1.0,
        reward_baseline=0.0,
        advantage=advantage,
        trajectory_id=trajectory_id,
    )
    if include_grpo_fields:
        sample.rollout_logprobs = np.full(SEQ_LENGTH, -1.0, dtype=np.float32)
        sample.old_logprobs = np.full(SEQ_LENGTH, -1.0, dtype=np.float32)
        sample.ref_logprobs = np.full(SEQ_LENGTH, -1.0, dtype=np.float32)
    return sample


def _train_samples(*, count: int, include_grpo_fields: bool) -> SampleTensorDict:
    samples = [_sample(include_grpo_fields=include_grpo_fields, trajectory_id=trajectory_id) for trajectory_id in range(count)]
    return SampleTensorDict.from_samples(samples, max_length=SEQ_LENGTH)


def _trainer(kind: Literal["sft", "grpo"]) -> BaseTrainer:
    if kind == "sft":
        return SftTrainer(SftTrainerConfig())
    return GrpoTrainer(GrpoTrainerConfig())


def _run_one_train(worker: RayMegatronWorker, *, kind: Literal["sft", "grpo"], num_train_batches: int) -> tuple[int, dict[str, float]]:
    worker.set_trainer(_trainer(kind))
    samples = _train_samples(
        count=worker.config.global_batch_size * num_train_batches,
        include_grpo_fields=(kind == "grpo"),
    )
    return worker.train(
        global_step=0,
        samples=samples,
        compute_logprobs=(kind == "grpo"),
    )


def _run_one_logprob(worker: RayMegatronWorker, *, kind: Literal["sft", "grpo"]) -> None:
    worker.set_trainer(_trainer(kind))
    samples = _train_samples(
        count=worker.config.global_batch_size,
        include_grpo_fields=(kind == "grpo"),
    )
    logprobs, gpu_usage_infos = worker.compute_logprobs(samples)
    assert logprobs.shape == samples["input_ids"].shape
    assert len(gpu_usage_infos) == worker.config.world_size()


def test_ray_megatron_train_and_logprobs_real_megatron_sft_and_grpo_dp2_cp2_tp2() -> None:
    config = _config()
    _skip_if_not_enough_gpus(config.world_size())
    worker: RayMegatronWorker | None = None
    try:
        resource_group = ResourceGroup([Request(cpu=1, gpu=config.world_size())])
        worker = RayMegatronWorker(config=config, resource_group=resource_group)
        worker.initialize()
        worker.copy_weights_to_cpu("init_weights")
        for kind in ("sft", "grpo"):
            global_step, metrics = _run_one_train(worker, kind=kind, num_train_batches=2)
            assert global_step == 2
            assert metrics, f"{kind} training should report train metrics."
            assert any(key.endswith("/loss") for key in metrics), metrics
        for kind in ("sft", "grpo"):
            _run_one_logprob(worker, kind=kind)
    finally:
        with contextlib.suppress(Exception):
            if worker is not None:
                worker.shutdown()
