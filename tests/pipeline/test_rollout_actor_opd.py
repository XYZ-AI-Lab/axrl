from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

import pytest

from axrl.configs import (
    IGNORE_INDEX,
    GrpoTrainerConfig,
    MegatronWorkerConfig,
    ModelConfig,
    OAIClientConfig,
    OPDConfig,
    RolloutWorkerConfig,
    SamplingConfig,
)
from axrl.data import Conversation, GenerationInput, GenerationOutput, Message, RolloutResult, Sample, SampleTensorDict, array_utils
from axrl.data.rollout_trace import RolloutTrace
from axrl.metrics.response_metric import ResponseMetric
from axrl.pipeline import ControllerConfig, PipelineExperimentConfig
from axrl.pipeline.rollout_actor import RemoteRolloutActor
from axrl.pipeline.rollout_data import RolloutRuntime
from axrl.recipe.base_recipe import BaseRecipe
from axrl.utils.ray_task_queue import TaskAssignment
from axrl.worker.oai_client import OAICompatibleGenerationClient

if TYPE_CHECKING:
    from axrl.data.rollout_trace_packing import RolloutTracePackRequest
    from axrl.ray.ray_rollout_worker import RayRolloutWorker
    from axrl.utils.ray_task_queue import RayTaskQueue


def _actor_class() -> type[Any]:
    return cast("type[Any]", cast("Any", RemoteRolloutActor).__ray_metadata__.modified_class)


def _new_actor() -> Any:
    return object.__new__(_actor_class())


def _metric() -> ResponseMetric:
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
        score=1.0,
    )


def _sample() -> Sample:
    return Sample(
        input_ids=array_utils.as_i32([1, 2, 3]),
        labels=array_utils.as_i32([2, 3, IGNORE_INDEX]),
        loss_mask=array_utils.as_bool([False, True, False]),
        attention_mask=array_utils.as_bool([True, True, True]),
        position_ids=array_utils.as_i32([0, 1, 2]),
        reward=0.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([0.0, 1.0, 0.0]),
    )


def _result(conversation: Conversation | None = None) -> RolloutResult:
    if conversation is None:
        conversation = Conversation(
            conversation_id="conv",
            messages=[Message(role="user", content="q")],
            extra={"group_id": "g"},
        )
    trace = RolloutTrace(conversation.deep_copy(), token_in_token_out=False)
    trace.turn_samples = [_sample()]
    return RolloutResult(conversation=conversation, trace=trace, metric=_metric())


class _ResultQueue:
    def __init__(self) -> None:
        self.items: list[RolloutResult] = []

    async def put(self, task: RolloutResult, *, key: str = "default") -> None:
        del key
        self.items.append(task)


class _RolloutQueue:
    def __init__(self) -> None:
        self.completed: list[str] = []

    async def complete(self, assignment_id: str) -> None:
        self.completed.append(assignment_id)


class _RecordingRecipe(BaseRecipe):
    def __init__(self, config: PipelineExperimentConfig, calls: list[str], result: RolloutResult) -> None:
        super().__init__(config)
        self.calls = calls
        self.result = result

    async def run_rollout(self, conversation: Conversation, runtime: RolloutRuntime) -> RolloutResult:
        del conversation, runtime
        self.calls.append("run_rollout")
        return self.result


class _RecordingTeacherClient(OAICompatibleGenerationClient):
    def __init__(self, calls: list[str]) -> None:
        super().__init__(OAIClientConfig(base_url="http://127.0.0.1:30000", sampling_config=SamplingConfig(max_total_tokens=8)))
        self.calls = calls

    async def generate(self, req: GenerationInput) -> GenerationOutput:
        self.calls.append("teacher_oai_client.generate")
        token_ids = array_utils.to_int_list(req.input_ids[req.input_logprob_start_index :])
        return GenerationOutput(
            session_id=req.session_id,
            output_ids=array_utils.as_i32([]),
            output_logprobs=array_utils.as_f32([]),
            output_text="",
            output_text_with_special_tokens="",
            cached_tokens=0,
            finish_reason="stop",
            e2e_elapsed_seconds=0.0,
            stop_reason=None,
            retry=0,
            input_logprobs=array_utils.as_f32([-0.5] * len(token_ids)),
            input_logprob_token_ids=array_utils.as_i32(token_ids),
            input_logprob_start_index=req.input_logprob_start_index,
        )


def test_rollout_actor_annotates_teacher_logprobs_before_packing() -> None:
    calls: list[str] = []
    conversation = Conversation(
        conversation_id="conv",
        messages=[Message(role="user", content="q")],
        extra={"group_id": "g", "pack_rollout_trace": True},
    )
    result = _result(conversation)
    result_queue = _ResultQueue()
    rollout_queue = _RolloutQueue()
    teacher_model = ModelConfig(seq_length=8)
    runtime = RolloutRuntime(
        rollout_worker=cast("RayRolloutWorker", object()),
        rollout_queue=cast("RayTaskQueue[Conversation]", rollout_queue),
        result_queue=cast("RayTaskQueue[RolloutResult]", result_queue),
    )
    recipe = _RecordingRecipe(
        PipelineExperimentConfig(
            grpo=GrpoTrainerConfig(
                opd=OPDConfig(
                    enabled=True,
                    backend="sglang",
                    teacher_model=teacher_model,
                    sglang_worker=RolloutWorkerConfig(model=teacher_model),
                    sglang_port=30000,
                )
            )
        ),
        calls,
        result,
    )
    actor = _new_actor()
    actor.recipe = recipe
    actor.runtime = runtime
    actor._teacher_oai_client = _RecordingTeacherClient(calls)

    async def pack_rollout_result(rollout_result: RolloutResult) -> RolloutResult:
        calls.append("pack_rollout_result")
        rollout_result.packed_samples = SampleTensorDict.from_samples(cast("RolloutTrace", rollout_result.trace).turn_samples, max_length=8)
        return rollout_result

    actor.pack_rollout_result = pack_rollout_result
    assignment = TaskAssignment(
        assignment_id="assignment-1",
        task=conversation,
        worker_id="worker",
        queued_at=time.monotonic(),
        assigned_at=time.monotonic(),
    )

    asyncio.run(actor._run_rollout_assignment(assignment, conversation))

    assert calls == ["run_rollout", "teacher_oai_client.generate", "pack_rollout_result"]
    assert result_queue.items == [result]
    assert rollout_queue.completed == ["assignment-1"]
    assert "teacher_metrics" in result.conversation.extra
    teacher_logprobs = cast("RolloutTrace", result.trace).turn_samples[0].teacher_logprobs
    assert teacher_logprobs is not None
    assert teacher_logprobs.tolist() == [0.0, -0.5, 0.0]
    assert result.packed_samples is not None


def test_rollout_actor_pack_rollout_result_clones_packing_pool_output() -> None:
    config = PipelineExperimentConfig(
        controller=ControllerConfig(allow_prefix_merging=False),
        megatron_worker=MegatronWorkerConfig(model=ModelConfig(seq_length=8)),
    )
    actor = _new_actor()
    actor.recipe = BaseRecipe(config)
    result = _result()
    returned = SampleTensorDict.from_samples(cast("RolloutTrace", result.trace).turn_samples, max_length=8)

    class FakePackingPool:
        def __init__(self) -> None:
            self.requests: list[RolloutTracePackRequest] = []

        async def generate(self, request: RolloutTracePackRequest) -> SampleTensorDict:
            self.requests.append(request)
            return returned

    fake_pool = FakePackingPool()
    actor._packing_pool = fake_pool

    asyncio.run(actor.pack_rollout_result(result))

    assert len(fake_pool.requests) == 1
    assert fake_pool.requests[0].max_pack_length == 8
    assert not fake_pool.requests[0].allow_prefix_sharing
    assert result.packed_samples is not None
    assert result.packed_samples is not returned
    assert result.packed_samples["input_ids"].data_ptr() != returned["input_ids"].data_ptr()
    assert result.packed_samples["input_ids"].tolist() == returned["input_ids"].tolist()


def test_rollout_actor_run_assignment_skips_packing_without_flag() -> None:
    config = PipelineExperimentConfig(
        controller=ControllerConfig(allow_prefix_merging=False),
        megatron_worker=MegatronWorkerConfig(model=ModelConfig(seq_length=8)),
    )
    actor = _new_actor()
    result = _result(
        Conversation(
            conversation_id="conv",
            messages=[Message(role="user", content="q")],
        )
    )
    calls: list[str] = []
    result_queue = _ResultQueue()
    rollout_queue = _RolloutQueue()
    runtime = RolloutRuntime(
        rollout_worker=cast("RayRolloutWorker", object()),
        rollout_queue=cast("RayTaskQueue[Conversation]", rollout_queue),
        result_queue=cast("RayTaskQueue[RolloutResult]", result_queue),
    )
    actor.recipe = _RecordingRecipe(config, calls, result)
    actor.runtime = runtime
    actor._teacher_oai_client = None

    class FakePackingPool:
        def __init__(self) -> None:
            self.requests: list[RolloutTracePackRequest] = []

        async def generate(self, request: RolloutTracePackRequest) -> SampleTensorDict:
            self.requests.append(request)
            raise AssertionError("Eval-style rollout without pack_rollout_trace should not call packing.")

    fake_pool = FakePackingPool()
    actor._packing_pool = fake_pool
    assignment = TaskAssignment(
        assignment_id="assignment-1",
        task=result.conversation,
        worker_id="worker",
        queued_at=time.monotonic(),
        assigned_at=time.monotonic(),
    )

    asyncio.run(actor._run_rollout_assignment(assignment, result.conversation))

    assert calls == ["run_rollout"]
    assert fake_pool.requests == []
    assert result_queue.items == [result]
    assert rollout_queue.completed == ["assignment-1"]
    assert result.packed_samples is None


def test_rollout_actor_packing_pool_does_not_scale_with_request_concurrency(monkeypatch: Any) -> None:
    created_num_processors: list[int] = []

    class FakeProcessorPool:
        def __init__(self, processor_cls: Any, config: Any, num_processors: int, timeout_seconds: float) -> None:
            del processor_cls, config, timeout_seconds
            created_num_processors.append(num_processors)

    monkeypatch.setattr("axrl.processor.processor_pool.ProcessorPool", FakeProcessorPool)
    actor = _new_actor()
    actor.max_running_tasks = 128

    actor.initialize_packing_pool()

    assert created_num_processors == [1]


def test_rollout_actor_health_check_reraises_background_task_error() -> None:
    actor = _new_actor()

    async def run() -> None:
        raise RuntimeError("rollout worker failed")

    async def check() -> None:
        actor._run_task = asyncio.create_task(run())
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="rollout worker failed"):
            await actor.check_health()

    asyncio.run(check())
