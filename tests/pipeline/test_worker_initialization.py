from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
import torch

from axrl.configs import MegatronWorkerConfig, ModelConfig, RolloutWorkerConfig
from axrl.pipeline import ControllerConfig, PipelineController, PipelineExperimentConfig, PipelineRunMode
from axrl.ray import ray_utils
from axrl.recipe.base_recipe import BaseRecipe
from axrl.utils import setup_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from axrl.ray.resource_group import ResourceGroup


MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
pytestmark = pytest.mark.usefixtures("ray_runtime")


@dataclass(frozen=True)
class _InitializeWorkersCase:
    run_mode: PipelineRunMode
    colocated: bool
    expected_rollout: bool
    expected_megatron: bool
    required_gpus: int


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


def _config(
    *,
    run_mode: PipelineRunMode = "online_rl_train",
    colocated: bool = True,
    rollout_tp_size: int = 1,
    rollout_num_workers: int = 1,
    megatron_tp_size: int = 1,
) -> PipelineExperimentConfig:
    model = ModelConfig(name=MODEL_NAME, seq_length=64)
    return PipelineExperimentConfig(
        controller=ControllerConfig(
            run_mode=run_mode,
            colocated=colocated,
        ),
        rollout_worker=RolloutWorkerConfig(
            model=model,
            tp_size=rollout_tp_size,
            pp_size=1,
            num_workers=rollout_num_workers,
            load_dummy_weights=False,
            gpu_memory_utilization=0.35,
            enable_metrics=False,
        ),
        megatron_worker=MegatronWorkerConfig(
            model=model,
            tp_size=megatron_tp_size,
            pp_size=1,
            dp_size=1,
            cp_size=1,
            vpp_size=None,
            inference_only=True,
        ),
    )


def _pipeline_controller(config: PipelineExperimentConfig) -> PipelineController:
    return PipelineController(config, BaseRecipe(config))


def _shutdown_resource_groups(*groups: ResourceGroup | None) -> None:
    seen: set[str] = set()
    for group in groups:
        if group is None:
            continue
        group_id = str(group.pg.id)
        if group_id in seen:
            continue
        seen.add(group_id)
        group.shutdown()


def _shutdown_controller(controller: PipelineController) -> None:
    rollout_group = controller.rollout_worker.get_resource_group() if controller.rollout_worker is not None else None
    megatron_group = controller.megatron_worker.resource_group if controller.megatron_worker is not None else None
    with contextlib.suppress(Exception):
        if controller.rollout_worker is not None:
            controller.rollout_worker.shutdown()
    with contextlib.suppress(Exception):
        if controller.megatron_worker is not None:
            controller.megatron_worker.shutdown()
    _shutdown_resource_groups(rollout_group, megatron_group)


def test_worker_placement_colocated_shares_real_resource_group() -> None:
    _skip_if_not_enough_gpus(required_gpus=2)
    controller = _pipeline_controller(_config(colocated=True, rollout_num_workers=2, megatron_tp_size=2))
    placement = controller.get_worker_placement()

    try:
        assert placement.rollout is placement.megatron
        assert placement.rollout is not None
        assert [request.gpu for request in placement.rollout.requests] == [1, 1]
        assert len(placement.rollout.bundle_infos) == 2
    finally:
        _shutdown_resource_groups(placement.rollout, placement.megatron)


def test_worker_placement_disaggregated_uses_real_separate_resource_groups() -> None:
    _skip_if_not_enough_gpus(required_gpus=4)
    controller = _pipeline_controller(_config(colocated=False, rollout_num_workers=2, megatron_tp_size=2))
    placement = controller.get_worker_placement()

    try:
        assert placement.rollout is not None
        assert placement.megatron is not None
        assert placement.rollout is not placement.megatron
        assert placement.rollout.pg.id != placement.megatron.pg.id
        assert [request.gpu for request in placement.rollout.requests] == [1, 1]
        assert [request.gpu for request in placement.megatron.requests] == [1, 1]
    finally:
        _shutdown_resource_groups(placement.rollout, placement.megatron)


def test_worker_placement_eval_only_skips_megatron_group() -> None:
    _skip_if_not_enough_gpus(required_gpus=1)
    controller = _pipeline_controller(_config(run_mode="eval_only"))
    placement = controller.get_worker_placement()

    try:
        assert placement.rollout is not None
        assert placement.megatron is None
        assert [request.gpu for request in placement.rollout.requests] == [1]
    finally:
        _shutdown_resource_groups(placement.rollout)


def test_worker_placement_sft_train_skips_rollout_group() -> None:
    _skip_if_not_enough_gpus(required_gpus=1)
    controller = _pipeline_controller(_config(run_mode="sft_train"))
    placement = controller.get_worker_placement()

    try:
        assert placement.rollout is None
        assert placement.megatron is not None
        assert [request.gpu for request in placement.megatron.requests] == [1]
    finally:
        _shutdown_resource_groups(placement.megatron)


def test_worker_placement_checks_rollout_and_megatron_gpu_totals() -> None:
    _skip_if_not_enough_gpus(required_gpus=1)
    controller = _pipeline_controller(_config(rollout_num_workers=1, megatron_tp_size=2))

    with pytest.raises(AssertionError, match="same number of GPUs"):
        controller.get_worker_placement()


def test_worker_placement_checks_cluster_gpu_capacity() -> None:
    available_gpus = torch.cuda.device_count()
    controller = _pipeline_controller(
        _config(
            colocated=True,
            rollout_num_workers=available_gpus + 1,
            megatron_tp_size=available_gpus + 1,
        )
    )

    with pytest.raises(AssertionError, match="Need at least"):
        controller.get_worker_placement()


@pytest.mark.parametrize(
    "case",
    [
        _InitializeWorkersCase(
            run_mode="online_rl_train",
            colocated=True,
            expected_rollout=True,
            expected_megatron=True,
            required_gpus=1,
        ),
        _InitializeWorkersCase(
            run_mode="eval_only",
            colocated=True,
            expected_rollout=True,
            expected_megatron=False,
            required_gpus=1,
        ),
        _InitializeWorkersCase(
            run_mode="sft_train",
            colocated=True,
            expected_rollout=False,
            expected_megatron=True,
            required_gpus=1,
        ),
    ],
)
def test_initialize_workers_uses_real_workers_needed_by_run_mode(case: _InitializeWorkersCase) -> None:
    _skip_if_not_enough_gpus(required_gpus=case.required_gpus)
    from axrl.ray.ray_megatron_worker import RayMegatronWorker
    from axrl.ray.ray_rollout_worker import RayRolloutWorker

    controller = _pipeline_controller(_config(run_mode=case.run_mode, colocated=case.colocated))

    try:
        asyncio.run(controller.initialize_workers())

        assert isinstance(controller.rollout_worker, RayRolloutWorker) is case.expected_rollout
        assert isinstance(controller.megatron_worker, RayMegatronWorker) is case.expected_megatron
        if controller.rollout_worker is not None:
            assert controller.rollout_worker.get_config().model.name == MODEL_NAME
            rollout_memory_released = asyncio.run(controller.rollout_worker.is_gpu_memory_released())
            assert rollout_memory_released is (case.run_mode != "eval_only")
            if case.run_mode == "eval_only":
                assert not rollout_memory_released, "eval_only should keep the rollout worker on GPU after initialization."
        if controller.megatron_worker is not None:
            assert controller.megatron_worker.config.model.name == MODEL_NAME
    finally:
        _shutdown_controller(controller)
