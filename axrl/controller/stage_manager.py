from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axrl.ray.ray_megatron_worker import RayMegatronWorker
    from axrl.ray.ray_rollout_worker import RayRolloutWorker


class StageManager:
    async def initialize_placement(self) -> None:
        pass

    async def switch_to_weight_sync(self) -> None:
        pass

    async def switch_to_rollout(self) -> None:
        pass

    async def switch_to_train(self) -> None:
        pass

    async def switch_to_value_train(self) -> None:
        pass


class ColocatedStageManager(StageManager):
    """Colocated placement manager."""

    def __init__(
        self,
        rollout_worker: RayRolloutWorker,
        megatron_worker: RayMegatronWorker,
        value_worker: RayMegatronWorker | None = None,
    ) -> None:
        super().__init__()
        self.rollout_worker = rollout_worker
        self.megatron_worker = megatron_worker
        self.value_worker = value_worker

    async def switch_to_weight_sync(self) -> None:
        await self.rollout_worker.pause_generation()
        await self.rollout_worker.release_gpu_memory(backup_weights_on_cpu=False)
        await self.rollout_worker.resume_gpu_memory(["weights"])
        if self.value_worker is not None:
            self.value_worker.to_cpu()
        self.megatron_worker.to_gpu()

    async def switch_to_rollout(self) -> None:
        self.megatron_worker.to_cpu()
        if self.value_worker is not None:
            self.value_worker.to_cpu()
        await self.rollout_worker.resume_gpu_memory(["weights"])
        await self.rollout_worker.resume_gpu_memory(["kv_cache"])
        await self.rollout_worker.resume_generation()

    async def switch_to_train(self) -> None:
        await self.rollout_worker.pause_generation()
        await self.rollout_worker.release_gpu_memory(backup_weights_on_cpu=False)
        if self.value_worker is not None:
            self.value_worker.to_cpu()
        self.megatron_worker.to_gpu()

    async def switch_to_value_train(self) -> None:
        assert self.value_worker is not None, "switch_to_value_train requires a value worker."
        await self.rollout_worker.pause_generation()
        await self.rollout_worker.release_gpu_memory(backup_weights_on_cpu=False)
        self.megatron_worker.to_cpu()
        self.value_worker.to_gpu()


class DisaggregatedStageManager(StageManager):
    """Disaggregated placement manager."""

    def __init__(
        self,
        rollout_worker: RayRolloutWorker,
        megatron_worker: RayMegatronWorker,
        value_worker: RayMegatronWorker | None = None,
    ) -> None:
        super().__init__()
        self.rollout_worker = rollout_worker
        self.megatron_worker = megatron_worker
        self.value_worker = value_worker

    async def initialize_placement(self) -> None:
        await self.rollout_worker.resume_gpu_memory()
        if self.value_worker is not None:
            self.value_worker.to_cpu()
        self.megatron_worker.to_gpu()

    async def switch_to_weight_sync(self) -> None:
        if self.value_worker is not None:
            self.value_worker.to_cpu()
        self.megatron_worker.to_gpu()

    async def switch_to_rollout(self) -> None:
        await self.rollout_worker.resume_gpu_memory(["weights"])
        await self.rollout_worker.resume_gpu_memory(["kv_cache"])
        await self.rollout_worker.resume_generation()

    async def switch_to_train(self) -> None:
        if self.value_worker is not None:
            self.value_worker.to_cpu()
        self.megatron_worker.to_gpu()

    async def switch_to_value_train(self) -> None:
        assert self.value_worker is not None, "switch_to_value_train requires a value worker."
        self.megatron_worker.to_cpu()
        self.value_worker.to_gpu()
