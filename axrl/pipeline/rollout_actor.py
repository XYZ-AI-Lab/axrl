from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any, cast

import ray

from axrl.pipeline.rollout_data import RolloutRuntime
from axrl.pipeline.utils import shutdown_local_workers

if TYPE_CHECKING:
    from ray.actor import ActorHandle

    from axrl.data import Conversation, RolloutResult, SampleTensorDict
    from axrl.processor.processor_pool import ProcessorPool
    from axrl.recipe.base_recipe import BaseRecipe
    from axrl.utils.ray_task_queue import TaskAssignment
    from axrl.worker.oai_client import OAICompatibleGenerationClient


logger = logging.getLogger(__name__)
_ROLLOUT_ACTOR_SHUTDOWN_TIMEOUT_SECONDS = 30


@ray.remote
class RemoteRolloutActor:
    def __init__(
        self,
        worker_id: str,
        max_running_tasks: int,
        runtime: RolloutRuntime,
        recipe: BaseRecipe,
    ) -> None:
        assert max_running_tasks > 0, "max_running_tasks must be greater than zero."
        self.worker_id = worker_id
        self.max_running_tasks = max_running_tasks
        self.runtime = RolloutRuntime(
            rollout_worker=runtime.rollout_worker,
            rollout_queue=runtime.rollout_queue,
            result_queue=runtime.result_queue,
            local_workers=dict(recipe.initialize_local_processors(worker_id)),
            shared_workers=runtime.shared_workers,
        )
        self.recipe = recipe
        self._run_task: asyncio.Task[None] | None = None
        self._teacher_oai_client: OAICompatibleGenerationClient | None = self.initialize_teacher_oai_client()
        self._packing_pool: ProcessorPool[Any, SampleTensorDict] | None = self.initialize_packing_pool()

    def initialize_teacher_oai_client(self) -> OAICompatibleGenerationClient | None:
        from axrl.opd.teacher_logprobs import initialize_local_teacher_oai_client

        return initialize_local_teacher_oai_client(self.recipe.config)

    def initialize_packing_pool(self) -> ProcessorPool[Any, SampleTensorDict]:
        from axrl.data.rollout_trace_packing import RolloutTracePackingProcessor
        from axrl.processor.processor_pool import ProcessorPool

        return ProcessorPool(
            RolloutTracePackingProcessor,
            config=None,
            num_processors=1,
            timeout_seconds=600,
        )

    async def initialize(self) -> None:
        assert self._run_task is None, "RemoteRolloutActor.initialize() must only be called once."
        self._run_task = asyncio.create_task(self.run())

    async def set_max_running_tasks(self, max_running_tasks: int) -> None:
        assert max_running_tasks > 0, "max_running_tasks must be greater than zero."
        self.max_running_tasks = max_running_tasks

    async def check_health(self) -> None:
        if self._run_task is not None and self._run_task.done():
            # Task.result() re-raises the original background rollout exception.
            self._run_task.result()

    async def shutdown(self) -> None:
        if self._run_task is not None:
            self._run_task.cancel()
            await asyncio.gather(self._run_task, return_exceptions=True)
            self._run_task = None
        if self._packing_pool is not None:
            self._packing_pool.shutdown()
            self._packing_pool = None
        if self._teacher_oai_client is not None:
            self._teacher_oai_client.shutdown()
            self._teacher_oai_client = None
        shutdown_local_workers(self.runtime.local_workers)

    async def run(self) -> None:
        # Each local worker keeps one rollout in flight: finish one task, then
        # pull the next one.
        parallel_rollout_workers = []
        for local_index in range(self.max_running_tasks):
            parallel_rollout_workers.append(self._run_parallel_rollout_worker(local_index))
        await asyncio.gather(*parallel_rollout_workers)

    async def _run_parallel_rollout_worker(self, local_index: int) -> None:
        worker_id = f"{self.worker_id}:parallel-rollout-{local_index}"
        while True:
            assignment = await self.runtime.rollout_queue.get(worker_id)
            assert assignment is not None, "Rollout queue returned None for a blocking get without timeout."
            conversation = assignment.task
            await self._run_rollout_assignment(assignment, conversation)

    async def _run_rollout_assignment(self, assignment: TaskAssignment[Conversation], conversation: Conversation) -> None:
        try:
            rollout_result = await self.recipe.run_rollout(conversation, self.runtime)
            rollout_result = await self.annotate_teacher_logprobs(rollout_result)
            if rollout_result.conversation.extra.get("pack_rollout_trace", False):
                rollout_result = await self.pack_rollout_result(rollout_result)
            await self.runtime.result_queue.put(rollout_result)
            await self.runtime.rollout_queue.complete(assignment.assignment_id)
        except Exception:
            logger.exception(
                "Rollout assignment failed: worker_id=%s assignment_id=%s conversation_id=%s session_id=%s.",
                self.worker_id,
                assignment.assignment_id,
                conversation.conversation_id,
                conversation.gen_state.session_id,
            )
            raise

    async def annotate_teacher_logprobs(self, result: RolloutResult) -> RolloutResult:
        from axrl.opd.teacher_logprobs import annotate_sglang_teacher_logprobs

        opd = self.recipe.config.grpo.opd
        if not opd.enabled or opd.backend != "sglang":
            return result

        assert self._teacher_oai_client is not None, "SGLang OPD requires actor-local teacher_oai_client."
        result.conversation.extra["teacher_metrics"] = await annotate_sglang_teacher_logprobs(result, self._teacher_oai_client)
        return result

    async def pack_rollout_result(self, result: RolloutResult) -> RolloutResult:
        from axrl.data.rollout_trace_packing import RolloutTracePackRequest
        from axrl.data.sample import collect_unique_handles_from_sample_tensor_dict

        assert result.trace is not None, "RolloutResult.trace is required before actor-side packing."
        allow_prefix_sharing = self.recipe.config.controller.allow_prefix_merging and self.recipe.config.megatron_worker.use_magi_merged_forward
        assert self._packing_pool is not None, "RemoteRolloutActor packing pool is not initialized."
        packed_samples = await self._packing_pool.generate(
            RolloutTracePackRequest(
                trajectory_id=0,
                turn_samples=result.trace.turn_samples,
                max_pack_length=self.recipe.config.megatron_worker.model.seq_length,
                allow_prefix_sharing=allow_prefix_sharing,
            )
        )
        local_packed_samples = packed_samples.clone()
        del packed_samples
        result.trainable_token_count = int(local_packed_samples["loss_mask"].sum().item())
        result.routing_handles = collect_unique_handles_from_sample_tensor_dict(local_packed_samples)
        if ray.is_initialized():
            result.packed_samples_ref = ray.put(local_packed_samples)
            result.packed_samples = None
            self._strip_heavy_rollout_payload(result)
        else:
            result.packed_samples = local_packed_samples
        return result

    @staticmethod
    def _strip_heavy_rollout_payload(result: RolloutResult) -> None:
        """Keep result-queue payloads small after actor-side packing."""
        if result.trace is not None:
            result.trace = None
        result.conversation.messages.clear()
        result.conversation.gen_state.input_ids = None
        result.conversation.gen_state.sampling_config = None
        result.conversation.gen_state.tools = None
        result.conversation.gen_state.tool_choice = None
        result.conversation.gen_state.tool_call_parser = None
        result.conversation.gen_state.capture_routing = False
        result.conversation.gen_state.captured_routing_rows = 0
        keep_extra_keys = {"answer", "group_id", "rollout_index", "pack_rollout_trace"}
        result.conversation.extra = {key: value for key, value in result.conversation.extra.items() if key in keep_extra_keys}


class RolloutActor:
    def __init__(self, actor: ActorHandle, max_running_tasks: int) -> None:
        self._actor = actor
        self.max_running_tasks = max_running_tasks

    async def set_max_running_tasks(self, max_running_tasks: int) -> None:
        assert max_running_tasks > 0, "max_running_tasks must be greater than zero."
        self.max_running_tasks = max_running_tasks
        await self._actor.set_max_running_tasks.remote(max_running_tasks)

    async def check_health(self) -> None:
        await self._actor.check_health.remote()

    def shutdown(self) -> None:
        with contextlib.suppress(Exception):
            ray.get(self._actor.shutdown.remote(), timeout=_ROLLOUT_ACTOR_SHUTDOWN_TIMEOUT_SECONDS)
        ray.kill(self._actor, no_restart=True)

    @staticmethod
    async def initialize_remote_actor(
        worker_id: str,
        num_cpus_per_actor: int,
        max_running_tasks: int,
        runtime: RolloutRuntime,
        recipe: BaseRecipe,
        scheduling_strategy: Any | None = None,
    ) -> ActorHandle:
        options: dict[str, Any] = {"num_cpus": num_cpus_per_actor}
        if scheduling_strategy is not None:
            options["scheduling_strategy"] = scheduling_strategy
        actor = cast(
            "ActorHandle",
            RemoteRolloutActor.options(**options).remote(  # type: ignore[attr-defined]
                worker_id,
                max_running_tasks,
                runtime,
                recipe,
            ),
        )
        await actor.initialize.remote()
        return actor
