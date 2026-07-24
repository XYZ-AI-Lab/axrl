from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, cast

import pytest
import ray
import torch

from axrl.configs import IGNORE_INDEX, DatasetConfig, GrpoTrainerConfig, MegatronWorkerConfig, ModelConfig, OPDConfig, RolloutWorkerConfig
from axrl.controller.stage_manager import ColocatedStageManager, DisaggregatedStageManager
from axrl.data import Conversation, Message, RolloutResult, Sample, SampleTensorDict, array_utils
from axrl.data.rollout_trace import RolloutTrace
from axrl.datasets.base_dataset import BaseDataset
from axrl.metrics.response_metric import ResponseMetric
from axrl.pipeline import (
    ControllerConfig,
    EvalOnlyConfig,
    OnlineRLTrainConfig,
    PipelineController,
    PipelineExperimentConfig,
    PipelineRunMode,
    ReplayRLTrainConfig,
    RolloutGroup,
    TrainGroupBatch,
)
from axrl.processor.processor_pool import ProcessorPoolTaskError
from axrl.recipe.base_recipe import BaseRecipe
from axrl.utils import zst_utils
from axrl.utils.ray_task_queue import RayTaskQueue

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from axrl.ray.ray_megatron_worker import RayMegatronWorker
    from axrl.ray.ray_rollout_worker import RayRolloutWorker


@pytest.fixture
def ray_runtime() -> Iterator[None]:
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=2, include_dashboard=False)
    try:
        yield
    finally:
        ray.shutdown()


def _metric(score: float) -> ResponseMetric:
    return ResponseMetric(
        token_count=1,
        token_unique_ratio=1.0,
        word_length_max=1,
        line_length_max=1,
        ngram_repetition=0.0,
        reasoning_behavior_backtracking=0.0,
        reasoning_behavior_verification=0.0,
        reasoning_behavior_causal=0.0,
        rollout_cached_tokens=0,
        rollout_num_retry=0,
        rollout_e2e_elapsed_seconds=0.0,
        rollout_finish_reason_stop=1,
        rollout_finish_reason_length=0,
        rollout_finish_reason_tool_calls=0,
        rollout_finish_reason_function_call=0,
        rollout_finish_reason_content_filter=0,
        score=score,
    )


def _sample_from_ids(input_ids: list[int], *, trainable_positions: set[int] | None = None) -> Sample:
    if trainable_positions is None:
        trainable_positions = set(range(1, len(input_ids) - 1))
    return Sample(
        input_ids=array_utils.as_i32(input_ids),
        labels=array_utils.as_i32([*input_ids[1:], IGNORE_INDEX]),
        loss_mask=array_utils.as_bool([index in trainable_positions for index in range(len(input_ids))]),
        attention_mask=array_utils.as_bool([True] * len(input_ids)),
        position_ids=array_utils.as_i32(list(range(len(input_ids)))),
        reward=0.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([0.0] * len(input_ids)),
    )


def _sample(seq_len: int = 4) -> Sample:
    return _sample_from_ids(list(range(1, seq_len + 1)))


def _result(*, conversation_id: str, group_id: str, score: float, rollout_index: int = 0) -> RolloutResult:
    conv = Conversation(
        conversation_id=conversation_id,
        messages=[Message(role="user", content="question"), Message(role="assistant", content="answer")],
        extra={"group_id": group_id, "rollout_index": rollout_index},
    )
    trace = RolloutTrace(conv.deep_copy(), token_in_token_out=False)
    trace.turn_samples = [_sample()]
    packed_samples = SampleTensorDict.from_samples(trace.turn_samples, max_length=128)
    return RolloutResult(conversation=conv, trace=trace, metric=_metric(score), packed_samples=packed_samples)


def _pipeline_controller(config: PipelineExperimentConfig) -> PipelineController:
    return PipelineController(config, BaseRecipe(config))


def _controller(*, filter_zero_std: bool = True, global_batch_size: int = 2) -> PipelineController:
    return _pipeline_controller(
        PipelineExperimentConfig(
            controller=ControllerConfig(
                allow_prefix_merging=False,
            ),
            online_rl_train=OnlineRLTrainConfig(
                num_rollouts_per_conversation=2,
                model_sync_every_n_global_updates=1,
                filter_zero_std=filter_zero_std,
            ),
            megatron_worker=MegatronWorkerConfig(
                model=ModelConfig(seq_length=128),
                global_batch_size=global_batch_size,
            ),
        )
    )


def test_pipeline_experiment_config_nests_controller_settings() -> None:
    config = PipelineExperimentConfig(
        controller=ControllerConfig(
            run_mode="eval_only",
            max_running_requests=32,
        ),
        replay_rl_train=ReplayRLTrainConfig(
            sample_dict_path="tmp/replay-samples.zst",
        ),
    )

    assert config.controller.run_mode == "eval_only"
    assert config.controller.max_running_requests == 32
    assert config.replay_rl_train.sample_dict_path == "tmp/replay-samples.zst"


def test_eval_only_model_override_updates_rollout_config_for_recipe_hooks() -> None:
    config = PipelineExperimentConfig(
        controller=ControllerConfig(run_mode="eval_only"),
        eval_only=EvalOnlyConfig(model=ModelConfig(name="eval-model", seq_length=256)),
        rollout_worker=RolloutWorkerConfig(model=ModelConfig(name="base-model", seq_length=128)),
        test_datasets=[DatasetConfig(name="unit-test")],
    )
    controller = _pipeline_controller(config)

    controller._apply_eval_only_model_override()

    assert controller.config.rollout_worker.model.name == "eval-model"
    assert controller.config.rollout_worker.model.seq_length == 256


def test_recipe_dataset_configs_preserve_eval_rollouts_per_prompt() -> None:
    class RecipeWithEvalDatasetConfig(BaseRecipe):
        def get_test_dataset_configs(self) -> Sequence[DatasetConfig]:
            return [DatasetConfig(name="recipe-eval", eval_num_rollouts_per_prompt=7)]

    config = PipelineExperimentConfig(
        controller=ControllerConfig(run_mode="eval_only"),
        test_datasets=None,
    )
    controller = PipelineController(config, RecipeWithEvalDatasetConfig(config))

    configs = controller.get_test_dataset_configs()

    assert configs is not None
    assert configs[0].eval_num_rollouts_per_prompt == 7
    controller.check_configs()


def test_start_opd_teacher_services_propagates_resolved_sglang_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    teacher_model = ModelConfig(name="teacher", seq_length=128)
    config = PipelineExperimentConfig(
        grpo=GrpoTrainerConfig(
            opd=OPDConfig(
                enabled=True,
                backend="sglang",
                teacher_model=teacher_model,
                sglang_worker=RolloutWorkerConfig(model=teacher_model),
                sglang_port=31080,
            )
        )
    )
    controller = _pipeline_controller(config)
    captured: dict[str, object] = {}

    class FakeResourceGroup:
        def __init__(self, requests: object) -> None:
            captured["requests"] = requests

        def shutdown(self) -> None:
            captured["shutdown"] = True

    async def fake_start_sglang_router(
        resource_group: object,
        worker_config: RolloutWorkerConfig,
        *,
        router_host: str | None = None,
        router_port: int | None = None,
    ) -> SimpleNamespace:
        captured["resource_group"] = resource_group
        captured["worker_config"] = worker_config
        captured["router_host"] = router_host
        captured["router_port"] = router_port
        return SimpleNamespace(host="10.0.0.9", port=31080, base_url="http://10.0.0.9:31080")

    monkeypatch.setattr("axrl.pipeline.controller.ResourceGroup", FakeResourceGroup)
    monkeypatch.setattr("axrl.pipeline.controller.assert_cluster_has_gpus", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("axrl.utils.sglang_launch_utils.start_sglang_router", fake_start_sglang_router)

    asyncio.run(controller.start_opd_teacher_services())

    assert captured["router_host"] is None
    assert captured["router_port"] == 31080
    assert controller.config.grpo.opd.sglang_host == "10.0.0.9"
    assert controller.config.grpo.opd.sglang_port == 31080
    assert "opd_teacher_sglang" in controller.recipe_services


def test_train_rollout_ids_include_monotonic_schedule_id() -> None:
    controller = _controller(filter_zero_std=False)
    dataset = BaseDataset()
    conv = Conversation(
        conversation_id="conv-1",
        source="unit-test",
        messages=[Message(role="user", content="question")],
    )
    dataset._conversations = [conv]
    dataset._label = ["42"]
    dataset._score_history = [[]]
    dataset._length_history = [[]]
    dataset._conversation_id_to_index = {"conv-1": 0}
    controller.train_dataset = dataset

    first = controller.build_train_rollout_conversations(num_groups=1)
    second = controller.build_train_rollout_conversations(num_groups=1)

    first_group_ids = {conv.extra["group_id"] for conv in first}
    second_group_ids = {conv.extra["group_id"] for conv in second}
    first_session_ids = {conv.gen_state.session_id for conv in first}
    second_session_ids = {conv.gen_state.session_id for conv in second}
    assert first_group_ids.isdisjoint(second_group_ids)
    assert first_session_ids.isdisjoint(second_session_ids)
    assert all(conv.extra["pack_rollout_trace"] is True for conv in first + second)


def test_online_rl_config_helpers_reject_incompatible_batch_shapes() -> None:
    controller = _controller(global_batch_size=3)

    with pytest.raises(AssertionError, match=r"global_batch_size .* divisible by num_rollouts_per_conversation"):
        controller.get_num_groups_per_batch_rollout()

    controller = _pipeline_controller(
        PipelineExperimentConfig(
            online_rl_train=OnlineRLTrainConfig(
                num_rollouts_per_conversation=4,
                model_sync_every_n_global_updates=1,
            ),
            megatron_worker=MegatronWorkerConfig(
                model=ModelConfig(seq_length=128),
                global_batch_size=2,
            ),
        )
    )

    with pytest.raises(AssertionError, match=r"global_batch_size \* model_sync_every_n_global_updates .* divisible"):
        controller.get_num_groups_per_model_sync()


class _RunModeDispatchController(PipelineController):
    def __init__(self, config: PipelineExperimentConfig) -> None:
        super().__init__(config, BaseRecipe(config))
        self.calls: list[str] = []

    async def initialize(self) -> None:
        self.calls.append("initialize")

    def shutdown(self) -> None:
        self.calls.append("shutdown")

    async def run_online_rl_train(self) -> None:
        self.calls.append("online_rl_train")

    async def run_replay_rl_train(self) -> None:
        self.calls.append("replay_rl_train")

    async def run_eval_only(self) -> list[RolloutResult]:
        self.calls.append("eval_only")
        return []

    async def run_sft_train(self) -> list[dict[str, float]]:
        self.calls.append("sft_train")
        return []

    async def run_mismatch_test(self) -> None:
        self.calls.append("mismatch_test")


class _StreamingController(PipelineController):
    def __init__(self, config: PipelineExperimentConfig) -> None:
        super().__init__(config, BaseRecipe(config))
        self.eval_calls = 0
        self.scheduled_groups: list[int] = []
        self.pack_calls = 0

    async def switch_to_rollout(self) -> None:
        return

    def build_train_rollout_conversations(self, num_groups: int) -> list[Conversation]:
        self.scheduled_groups.append(num_groups)
        return []

    async def enqueue_rollout_conversations(self, conversations: Sequence[Conversation]) -> None:
        assert list(conversations) == []

    async def run_evals_if_needed(self) -> None:
        self.eval_calls += 1

    async def pack_rollout_group(self, group: Sequence[RolloutResult]) -> list[RolloutResult]:
        self.pack_calls += 1
        return list(group)

    def _check_rollout_ready(self) -> tuple[RayTaskQueue[Conversation], RayTaskQueue[RolloutResult]]:
        assert self.rollout_queue is not None
        assert self.result_queue is not None
        return self.rollout_queue, self.result_queue


@pytest.mark.parametrize("run_mode", ["online_rl_train", "replay_rl_train", "eval_only", "sft_train", "mismatch_test"])
def test_start_dispatches_all_pipeline_run_modes(run_mode: PipelineRunMode) -> None:
    controller = _RunModeDispatchController(PipelineExperimentConfig(controller=ControllerConfig(run_mode=run_mode)))

    asyncio.run(controller.start())

    assert controller.calls == ["initialize", run_mode, "shutdown"]


@pytest.mark.parametrize(
    ("placement_mode", "expected_stage_manager_cls"),
    [
        ("colocated", ColocatedStageManager),
        ("disaggregated", DisaggregatedStageManager),
    ],
)
def test_initialize_stage_manager_selects_placement_manager(
    placement_mode: Literal["colocated", "disaggregated"],
    expected_stage_manager_cls: type[ColocatedStageManager | DisaggregatedStageManager],
) -> None:
    controller = _pipeline_controller(PipelineExperimentConfig(controller=ControllerConfig(colocated=placement_mode == "colocated")))
    controller.rollout_worker = cast("RayRolloutWorker", object())
    controller.megatron_worker = cast("RayMegatronWorker", object())

    controller.initialize_stage_manager()

    assert isinstance(controller.stage_manager, expected_stage_manager_cls)


def test_group_helpers_normalize_filter_pack_and_collect() -> None:
    controller = _controller(filter_zero_std=True)
    group = [
        _result(conversation_id="conv-1", group_id="group-1", score=0.0, rollout_index=0),
        _result(conversation_id="conv-1", group_id="group-1", score=1.0, rollout_index=1),
    ]

    controller.normalize_group_rewards(group)
    assert group[0].metric.score_mean == pytest.approx(0.5)
    assert group[1].metric.score_std == pytest.approx(0.70710678)
    assert controller.get_group_filter_type(group) == "pass"

    packed_group = asyncio.run(controller.pack_rollout_group(group))
    rollout_group = RolloutGroup(results=packed_group, filter_type="pass")
    samples = controller.collect_packed_samples(TrainGroupBatch(valid_groups=[rollout_group]))

    assert rollout_group.is_valid
    assert all(result.packed_samples is not None for result in group)
    assert len(samples) == 2
    assert torch.equal(samples["index"], torch.arange(len(samples), dtype=samples["index"].dtype))
    assert int(samples["loss_mask"].sum().item()) == 4


def test_pack_rollout_group_without_magi_accepts_flat_linear_trace() -> None:
    controller = _pipeline_controller(
        PipelineExperimentConfig(
            controller=ControllerConfig(allow_prefix_merging=True),
            megatron_worker=MegatronWorkerConfig(
                model=ModelConfig(seq_length=128),
                use_magi_merged_forward=False,
            ),
        )
    )
    group = [_result(conversation_id="conv-1", group_id="group-1", score=1.0)]
    assert group[0].trace is not None
    group[0].trace.turn_samples = [
        _sample_from_ids([1, 2, 3], trainable_positions={1}),
        _sample_from_ids([1, 2, 3, 4], trainable_positions={2}),
    ]

    packed_group = asyncio.run(controller.pack_rollout_group(group))

    assert packed_group[0].packed_samples is not None
    assert "merge_info" not in packed_group[0].packed_samples.keys()  # noqa: SIM118 - tensordict semantics


def test_pack_rollout_group_without_magi_rejects_tree_structured_trace() -> None:
    controller = _pipeline_controller(
        PipelineExperimentConfig(
            controller=ControllerConfig(allow_prefix_merging=True),
            megatron_worker=MegatronWorkerConfig(
                model=ModelConfig(seq_length=128),
                use_magi_merged_forward=False,
            ),
        )
    )
    group = [_result(conversation_id="conv-1", group_id="group-1", score=1.0)]
    assert group[0].trace is not None
    group[0].trace.turn_samples = [
        _sample_from_ids([1, 2, 3], trainable_positions={1}),
        _sample_from_ids([1, 2, 4], trainable_positions={1}),
    ]

    with pytest.raises(ProcessorPoolTaskError, match="linear prefix chain"):
        asyncio.run(controller.pack_rollout_group(group))


def test_pack_rollout_group_reuses_actor_packed_ref_without_trace(ray_runtime: None) -> None:
    del ray_runtime

    controller = _controller(filter_zero_std=False, global_batch_size=1)
    result = _result(conversation_id="conv-1", group_id="group-1", score=1.0)
    assert result.packed_samples is not None
    packed_samples = result.packed_samples
    result.packed_samples = None
    result.packed_samples_ref = ray.put(packed_samples)
    result.trainable_token_count = int(packed_samples["loss_mask"].sum().item())
    result.trace = None

    packed_group = asyncio.run(controller.pack_rollout_group([result]))

    assert packed_group == [result]
    assert result.packed_samples is None
    assert result.packed_samples_ref is not None

    samples = controller.collect_packed_samples(TrainGroupBatch(valid_groups=[RolloutGroup(results=packed_group, filter_type="pass")]))
    assert int(samples["loss_mask"].sum().item()) == result.trainable_token_count
    assert result.packed_samples_ref is None


def test_group_filter_type_detects_zero_std_failures() -> None:
    controller = _controller(filter_zero_std=True)
    group = [
        _result(conversation_id="conv-1", group_id="group-1", score=0.0, rollout_index=0),
        _result(conversation_id="conv-1", group_id="group-1", score=0.0, rollout_index=1),
    ]

    controller.normalize_group_rewards(group)

    assert controller.get_group_filter_type(group) == "zero_std_all_fail"


def test_collect_scheduled_rollout_groups_collects_valid_and_skipped_groups(ray_runtime: None) -> None:
    del ray_runtime
    controller = _controller(filter_zero_std=True)

    async def run() -> None:
        queue = RayTaskQueue[RolloutResult](RayTaskQueue.initialize_remote_actor(max_running_tasks=8))
        await queue.put_many(
            [
                _result(conversation_id="skip", group_id="skip", score=0.0, rollout_index=0),
                _result(conversation_id="skip", group_id="skip", score=0.0, rollout_index=1),
                _result(conversation_id="valid", group_id="valid", score=0.0, rollout_index=0),
                _result(conversation_id="valid", group_id="valid", score=1.0, rollout_index=1),
            ]
        )

        batch = await controller.collect_scheduled_rollout_groups(queue, scheduled_groups=2, max_valid_groups=1)

        assert len(batch.valid_groups) == 1
        assert len(batch.skipped_groups) == 1
        assert batch.filter_type_counts["pass"] == 1
        assert batch.filter_type_counts["zero_std_all_fail"] == 1
        samples = controller.collect_packed_samples(batch)
        assert len(samples) == 2

    asyncio.run(run())


def test_collect_scheduled_rollout_groups_returns_when_no_valid_groups(ray_runtime: None) -> None:
    del ray_runtime
    controller = _controller(filter_zero_std=True)

    async def run() -> None:
        queue = RayTaskQueue[RolloutResult](RayTaskQueue.initialize_remote_actor(max_running_tasks=4))
        await queue.put_many(
            [
                _result(conversation_id="skip", group_id="skip", score=0.0, rollout_index=0),
                _result(conversation_id="skip", group_id="skip", score=0.0, rollout_index=1),
            ]
        )

        batch = await controller.collect_scheduled_rollout_groups(queue, scheduled_groups=1, max_valid_groups=1)

        assert len(batch.valid_groups) == 0
        assert len(batch.skipped_groups) == 1
        assert batch.filter_type_counts["zero_std_all_fail"] == 1

    asyncio.run(run())


def test_rollout_queue_config_requires_enough_running_slots_for_actors() -> None:
    controller = _pipeline_controller(
        PipelineExperimentConfig(
            controller=ControllerConfig(
                run_mode="online_rl_train",
                num_rollout_actors=4,
                max_running_requests=3,
            )
        )
    )

    with pytest.raises(AssertionError, match=r"max_running_requests .* greater than or equal to num_rollout_actors"):
        controller.check_configs()


def test_rollout_wait_log_includes_queue_stats(ray_runtime: None, caplog: pytest.LogCaptureFixture) -> None:
    del ray_runtime
    controller = _controller(filter_zero_std=False)

    async def run() -> None:
        rollout_queue = RayTaskQueue[Conversation](RayTaskQueue.initialize_remote_actor(max_running_tasks=2))
        result_queue = RayTaskQueue[RolloutResult](RayTaskQueue.initialize_remote_actor(max_running_tasks=2))
        controller.rollout_queue = rollout_queue
        await rollout_queue.put_many(
            [
                Conversation(conversation_id="conv-1"),
                Conversation(conversation_id="conv-2"),
            ]
        )
        assignment = await rollout_queue.get("worker")
        assert assignment is not None

        with caplog.at_level(logging.WARNING, logger="axrl.pipeline.controller"):
            await controller._log_rollout_wait(
                result_queue,
                context="eval",
                completed_count=0,
                expected_count=2,
            )

    asyncio.run(run())

    assert "Still waiting for rollout results: context=eval progress=0/2" in caplog.text
    assert "rollout_queue=pending:1 running:1 completed:0" in caplog.text
    assert "result_queue=pending:0 running:0 completed:0" in caplog.text


def test_reset_init_weights_cadence_must_align_with_model_sync() -> None:
    valid_controller = _pipeline_controller(
        PipelineExperimentConfig(
            controller=ControllerConfig(run_mode="online_rl_train"),
            online_rl_train=OnlineRLTrainConfig(
                model_sync_every_n_global_updates=4,
            ),
            megatron_worker=MegatronWorkerConfig(
                model=ModelConfig(seq_length=128),
                reset_init_weights_every_k_steps=8,
            ),
        )
    )
    controller = _pipeline_controller(
        PipelineExperimentConfig(
            controller=ControllerConfig(run_mode="online_rl_train"),
            online_rl_train=OnlineRLTrainConfig(
                model_sync_every_n_global_updates=4,
            ),
            megatron_worker=MegatronWorkerConfig(
                model=ModelConfig(seq_length=128),
                reset_init_weights_every_k_steps=6,
            ),
        )
    )

    valid_controller.check_configs()
    with pytest.raises(AssertionError, match=r"reset_init_weights_every_k_steps .* must be divisible"):
        controller.check_configs()


def test_select_replay_train_rollouts_uses_leading_valid_groups() -> None:
    controller = _controller(global_batch_size=4)
    groups = [
        [
            _result(
                conversation_id=f"group-{group_index}",
                group_id=f"group-{group_index}",
                score=1.0,
                rollout_index=rollout_index,
            )
            for rollout_index in range(2)
        ]
        for group_index in range(3)
    ]

    selected_groups = controller.select_replay_train_rollouts(groups)

    assert selected_groups == groups[:2]


def test_online_train_stream_runs_eval_after_scheduled_groups_are_drained(ray_runtime: None) -> None:
    del ray_runtime
    controller = _StreamingController(
        PipelineExperimentConfig(
            online_rl_train=OnlineRLTrainConfig(
                num_rollouts_per_conversation=2,
                model_sync_every_n_global_updates=1,
                batch_rollout_for_n_global_updates=4,
                max_global_updates=4,
            ),
            megatron_worker=MegatronWorkerConfig(
                model=ModelConfig(seq_length=128),
                global_batch_size=2,
            ),
        )
    )

    async def run() -> None:
        rollout_queue = RayTaskQueue[Conversation](RayTaskQueue.initialize_remote_actor(max_running_tasks=8))
        result_queue = RayTaskQueue[RolloutResult](RayTaskQueue.initialize_remote_actor(max_running_tasks=8))
        controller.rollout_worker = cast("RayRolloutWorker", object())
        controller.rollout_queue = rollout_queue
        controller.result_queue = result_queue
        await result_queue.put_many(
            [
                _result(
                    conversation_id=f"group-{group_index}",
                    group_id=f"group-{group_index}",
                    score=float(rollout_index),
                    rollout_index=rollout_index,
                )
                for group_index in range(4)
                for rollout_index in range(2)
            ]
        )

        stream = controller.stream_online_train_group_batches()
        for _ in range(4):
            batch = await anext(stream)
            assert len(batch.valid_groups) == 1
            assert controller.eval_calls == 0

        controller.global_step = controller.config.online_rl_train.max_global_updates
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert controller.eval_calls == 1
        assert controller.scheduled_groups == [4]
        assert controller.pack_calls == 0

    asyncio.run(run())


def test_online_train_stream_shuffles_valid_groups_before_yield(ray_runtime: None, monkeypatch: pytest.MonkeyPatch) -> None:
    del ray_runtime
    shuffle_calls = 0

    def reverse_shuffle(items: list[RolloutGroup]) -> None:
        nonlocal shuffle_calls
        shuffle_calls += 1
        items.reverse()

    monkeypatch.setattr("axrl.pipeline.controller.random.shuffle", reverse_shuffle)
    controller = _StreamingController(
        PipelineExperimentConfig(
            online_rl_train=OnlineRLTrainConfig(
                num_rollouts_per_conversation=2,
                model_sync_every_n_global_updates=1,
                batch_rollout_for_n_global_updates=1,
                max_global_updates=1,
            ),
            megatron_worker=MegatronWorkerConfig(
                model=ModelConfig(seq_length=128),
                global_batch_size=4,
            ),
        )
    )

    async def run() -> None:
        rollout_queue = RayTaskQueue[Conversation](RayTaskQueue.initialize_remote_actor(max_running_tasks=8))
        result_queue = RayTaskQueue[RolloutResult](RayTaskQueue.initialize_remote_actor(max_running_tasks=8))
        controller.rollout_worker = cast("RayRolloutWorker", object())
        controller.rollout_queue = rollout_queue
        controller.result_queue = result_queue
        await result_queue.put_many(
            [
                _result(
                    conversation_id=f"group-{group_index}",
                    group_id=f"group-{group_index}",
                    score=float(rollout_index),
                    rollout_index=rollout_index,
                )
                for group_index in range(2)
                for rollout_index in range(2)
            ]
        )

        batch = await anext(controller.stream_online_train_group_batches())

        group_ids = [PipelineController.get_rollout_group_id(group.results[0]) for group in batch.valid_groups]
        assert group_ids == ["group-1", "group-0"]
        assert shuffle_calls == 1
        assert controller.pack_calls == 0

    asyncio.run(run())


def test_save_rollouts_snapshot_saves_all_rollouts_to_single_path_with_packed_samples(tmp_path: Path) -> None:
    controller = _pipeline_controller(
        PipelineExperimentConfig(
            online_rl_train=OnlineRLTrainConfig(
                rollout_save_filename="valid-rollouts",
                rollout_save_every_n_global_updates=2,
                save_all_rollouts=True,
            )
        )
    )
    controller.output_dir = tmp_path
    controller.global_step = 2
    try:
        valid_result = _result(conversation_id="valid", group_id="valid", score=1.0)
        skipped_result = _result(conversation_id="skipped", group_id="skipped", score=0.0)
        packed_samples = asyncio.run(controller.pack_rollout_group([valid_result]))[0].packed_samples
        assert packed_samples is not None

        controller.save_rollouts_snapshot([[valid_result]], [[skipped_result]])

        rollout_path = tmp_path / "valid-rollouts-step2.zst"
        saved = zst_utils.load_zst(rollout_path, verbose=False)
        assert len(saved) == 2
        assert saved[0][0].packed_samples is None
        assert not (tmp_path / "valid-rollouts-all-step2.zst").exists()
        assert valid_result.packed_samples is packed_samples
    finally:
        controller.shutdown()
