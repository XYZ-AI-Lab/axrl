from abc import ABC, abstractmethod
from typing import override

from axrl.configs import RolloutWorkerConfig
from axrl.data import GenerationInput, GenerationOutput
from axrl.worker.infer_worker import InferWorker


class RolloutWorker(InferWorker[GenerationInput, GenerationOutput], ABC):
    """Abstract interface for rollout workers.

    A rollout worker is an inference worker that can:
    - generate rollouts (required)
    - be paused/resumed (optional for some backends, but required by this interface)
    - receive weight updates (required)
    - participate in GPU memory management hooks (required)
    """

    def __init__(self, config: RolloutWorkerConfig) -> None:
        super().__init__()
        self.config = config

    @staticmethod
    def get_worker(config: RolloutWorkerConfig) -> "RolloutWorker":
        from axrl.worker.sglang_worker import SGLangWorker

        if config.engine_type == "sglang":
            return SGLangWorker(config)
        raise ValueError(f"Unsupported engine type: {config.engine_type}")

    @override
    @abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @override
    @abstractmethod
    async def generate(self, req: GenerationInput) -> GenerationOutput:
        raise NotImplementedError

    @abstractmethod
    async def pause_generation(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def resume_generation(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[bytes],
        load_format: str | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def update_weights_from_distributed(
        self,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
    ) -> None:
        raise NotImplementedError

    @override
    @abstractmethod
    def shutdown(self) -> None:
        raise NotImplementedError

    @override
    @abstractmethod
    async def flush_cache(self) -> None:
        raise NotImplementedError

    @override
    @abstractmethod
    async def release_gpu_memory(self, *, backup_weights_on_cpu: bool = True) -> None:
        raise NotImplementedError

    @override
    @abstractmethod
    async def resume_gpu_memory(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    @override
    @abstractmethod
    async def is_gpu_memory_released(self) -> bool:
        raise NotImplementedError
