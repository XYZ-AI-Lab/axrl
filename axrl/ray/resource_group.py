import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import ray
from ray.util.placement_group import PlacementGroup, placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tqdm import tqdm

from axrl.configs import PlacementGroupStrategy
from axrl.utils import network_utils
from axrl.worker.worker import Worker

logger = logging.getLogger(__name__)


@dataclass
class Request:
    cpu: int = 0
    gpu: int = 0


@dataclass
class BundleInfo:
    index: int
    specs: Request
    ip: str
    cuda_visible_devices: str
    node_id: str | None = None


def _get_ray_node_id() -> str:
    return str(ray.get_runtime_context().get_node_id())


class ResourceGroup:
    """Manages Ray placement groups for resource allocation."""

    def __init__(
        self,
        requests: list[Request],
        strategy: PlacementGroupStrategy = "PACK",
    ) -> None:
        self.requests = requests
        self.strategy = strategy
        self._closed = False
        self.pg: PlacementGroup = self._allocate_resource()
        self.bundle_infos = self._get_bundle_infos()
        logger.info(f"Created placement group with ID: {self.pg.id}")
        logger.info(f"Bundle infos: {self.bundle_infos}")

    def _allocate_resource(self) -> PlacementGroup:
        bundle_specs = [{"CPU": float(x.cpu), "GPU": float(x.gpu)} for x in self.requests]
        pg = placement_group(bundles=bundle_specs, strategy=self.strategy)
        pg.wait()
        return pg

    def _get_bundle_infos(self) -> list[BundleInfo]:
        infos: list[BundleInfo] = []
        for i, request in tqdm(enumerate(self.requests), desc=f"Gathering bundle info: {self.requests}", total=len(self.requests)):
            info = BundleInfo(
                index=i,
                specs=request,
                ip=self.run_task(func=network_utils.get_ip, bundle_index=i),
                cuda_visible_devices=self.run_task(func=Worker.get_cuda_visible_devices, bundle_index=i),
                node_id=self.run_task(func=_get_ray_node_id, bundle_index=i),
            )
            infos.append(info)
        return infos

    def get_scheduling_strategy(self, bundle_index: int = -1) -> PlacementGroupSchedulingStrategy:
        return PlacementGroupSchedulingStrategy(
            placement_group=self.pg,
            placement_group_bundle_index=bundle_index,
        )

    def run_task(
        self,
        func: Callable[..., Any],
        bundle_index: int,
        num_gpus: float | None = None,
        num_cpus: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute ``func`` inside the specified placement group bundle."""
        num_gpus = num_gpus if num_gpus is not None else self.pg.bundle_specs[bundle_index].get("GPU", 0)
        num_cpus = num_cpus if num_cpus is not None else self.pg.bundle_specs[bundle_index].get("CPU", 0)

        @ray.remote
        def _remote_func(*_a: Any, **_k: Any) -> Any:  # type: ignore[explicit-any]
            return func(*_a, **_k)

        future = _remote_func.options(
            num_gpus=num_gpus,
            num_cpus=num_cpus,
            scheduling_strategy=self.get_scheduling_strategy(bundle_index),
        ).remote(*args, **kwargs)
        result = ray.get(future)
        return result

    def shutdown(self) -> None:
        """Release the Ray placement group backing this resource group."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            remove_placement_group(self.pg)
        logger.info("Removed placement group with ID: %s", self.pg.id)


if __name__ == "__main__":
    from axrl.utils import setup_logger

    setup_logger(level="info")
    if not ray.is_initialized():
        ray.init(logging_level=logging.INFO)
    resource_group = ResourceGroup(requests=[Request(cpu=5, gpu=2), Request(cpu=10, gpu=1)])
    assert len(resource_group.bundle_infos) == 2
