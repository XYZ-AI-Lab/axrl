from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from axrl.configs import DatasetConfig
    from axrl.data import Conversation, RolloutResult
    from axrl.datasets.base_dataset import BaseDataset
    from axrl.pipeline.config import PipelineExperimentConfig
    from axrl.pipeline.rollout_data import RolloutRuntime
    from axrl.ray.ray_infer_worker import RayInferWorker
    from axrl.ray.ray_rollout_worker import RayRolloutWorker
    from axrl.worker.infer_worker import InferWorker


class BaseRecipe:
    """User-facing base class for defining recipe-specific components.

    Recipe objects are serialized to rollout actors. Most recipes should keep
    only the experiment config as instance state; avoid storing driver-owned
    services, tokenizers, process handles, or other non-portable objects on the
    recipe itself. Pass runtime handles through controller/shared workers
    instead.
    """

    def __init__(self, config: PipelineExperimentConfig) -> None:
        self.config = config

    async def register_datasets(self) -> None:
        """Register recipe dataset classes before the controller loads datasets."""

    def prepare_dataset(self, dataset: BaseDataset, config: DatasetConfig) -> None:
        """Inject recipe-specific dataset state before ``dataset.initialize()``.

        Search-agent uses this hook to attach MCP/Hermes tool schemas so prompt
        and ``gen_state.tools`` are built correctly by the dataset itself.
        """
        _ = dataset, config

    def validate_dataset(self, dataset: BaseDataset, config: DatasetConfig) -> None:
        """Validate recipe-specific dataset invariants after initialization."""
        _ = dataset, config

    async def shutdown(self) -> None:
        """Clean up driver-side resources started by ``initialize``."""

    async def start_services(self) -> dict[str, Any]:
        """Start driver-only services and return their lifecycle handles.

        Services are not sent to rollout actors. Use them for objects such as
        HTTP servers, registries, or other driver-owned resources that need an
        explicit shutdown path.
        """
        return {}

    async def stop_services(self, services: Mapping[str, Any]) -> None:
        """Stop services returned by ``start_services``."""
        _ = services

    def get_train_dataset_configs(self) -> Sequence[DatasetConfig] | None:
        """Return train dataset configs, or ``None`` to use config.train_datasets."""
        return None

    def get_test_dataset_configs(self) -> Sequence[DatasetConfig] | None:
        """Return test dataset configs, or ``None`` to use config.test_datasets."""
        return None

    async def run_rollout(
        self,
        conversation: Conversation,
        runtime: RolloutRuntime,
    ) -> RolloutResult:
        """Run one rollout using the runtime handles prepared by the pipeline.

        ``runtime`` contains the rollout worker, actor-local workers initialized
        by ``initialize_local_processors``, and shared workers initialized by
        ``initialize_shared_workers``.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.run_rollout() must be implemented.")

    def initialize_local_processors(self, worker_id: str) -> Mapping[str, InferWorker[Any, Any]]:
        """Create processors owned by one rollout actor/process.

        This hook is called once when each rollout actor is initialized. Objects
        returned here live for that actor's lifetime and are reused across
        rollouts handled by the same actor.

        Use this for cheap or frequently called processors where avoiding a Ray
        hop matters, such as tokenizers, adapters, and lightweight verifiers.
        The ``worker_id`` is provided so recipes can create per-actor resources
        or namespaced logs.
        """
        _ = worker_id
        return {}

    def initialize_shared_workers(self, services: Mapping[str, Any]) -> dict[str, RayRolloutWorker | RayInferWorker[Any, Any]]:
        """Create actor-portable shared runtime handles.

        This hook is called once per experiment/controller lifetime. Workers
        returned here are shared by rollout actors and are available to
        ``run_rollout`` through ``runtime.shared_workers``.

        Unlike services, shared workers are serialized to rollout actors. Return
        only portable Ray workers or lightweight handles here; if a handle is
        backed by a driver service, keep the service itself in ``services``.
        """
        _ = services
        return {}
