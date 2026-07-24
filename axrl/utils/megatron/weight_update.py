from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable

    from megatron.bridge.models.conversion.model_bridge import HFWeightTuple

    from axrl.ray.ray_rollout_worker import RayRolloutWorker
    from axrl.worker.megatron_worker import MegatronWorker

logger = logging.getLogger(__name__)


class WeightUpdater:
    """Synchronizes model weights from Megatron to a rollout worker."""

    def __init__(
        self,
        cur_megatron_worker: MegatronWorker,
        ray_rollout_worker: RayRolloutWorker,
        bucket_size_gb: float = 1.0,
    ) -> None:
        self.cur_megatron_worker = cur_megatron_worker
        self.ray_rollout_worker = ray_rollout_worker
        self.bridge = self.cur_megatron_worker.bridge
        self.bucket_size_gb = bucket_size_gb

    @staticmethod
    def create_weight_updater(
        cur_megatron_worker: MegatronWorker,
        ray_rollout_worker: RayRolloutWorker,
        bucket_size_gb: float = 1.0,
        *,
        colocated: bool,
    ) -> WeightUpdater:
        from axrl.utils.megatron.sglang_weight_updater import SGLangDistributedWeightUpdater, SGLangTensorWeightUpdater

        # key: (engine_type, colocated)
        mapping: dict[tuple[str, bool], type[WeightUpdater]] = {
            ("sglang", True): SGLangTensorWeightUpdater,
            ("sglang", False): SGLangDistributedWeightUpdater,
        }

        logger.info(
            "Resolving weight updater class: dist=%s, colocated=%s, bucket_size_gb=%s.",
            cur_megatron_worker.mcore_dist_info,
            colocated,
            bucket_size_gb,
        )
        engine_type = ray_rollout_worker.get_config().engine_type
        assert (engine_type, colocated) in mapping, f"Unsupported combination of engine_type={engine_type} and colocated={colocated}"
        updater_class = mapping[(engine_type, colocated)]
        logger.info(
            "Selected weight updater class: %s for engine_type=%s, colocated=%s.",
            updater_class.__name__,
            engine_type,
            colocated,
        )
        return updater_class(
            cur_megatron_worker=cur_megatron_worker,
            ray_rollout_worker=ray_rollout_worker,
            bucket_size_gb=bucket_size_gb,
        )

    def connect(self) -> None:
        raise NotImplementedError

    def update_weights(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def get_named_tensor_buckets(
        self,
        iterable: Iterable[HFWeightTuple],
        bucket_size_gb: float,
    ) -> Iterable[list[tuple[str, torch.Tensor]]]:
        """Groups named tensors into buckets of `bucket_size_gb` GiB."""
        assert bucket_size_gb > 0, "bucket_size_gb must be greater than 0"
        bucket_bytes = bucket_size_gb * 1024**3
        current_bucket: list[tuple[str, torch.Tensor]] = []
        current_size = 0
        for name, _tensor in iterable:
            tensor = _tensor
            tensor_size = tensor.element_size() * tensor.numel()
            if current_size + tensor_size > bucket_bytes:
                if current_bucket:
                    yield current_bucket
                current_bucket = [(name, tensor)]
                current_size = tensor_size
            else:
                current_bucket.append((name, tensor))
                current_size += tensor_size

        if current_bucket:
            yield current_bucket
