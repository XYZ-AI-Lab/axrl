import logging

from torch import Tensor

from axrl.data import SampleTensorDict
from axrl.utils.gpu_utils import GpuUsageInfo
from axrl.worker.worker import Worker

logger = logging.getLogger(__name__)


class TrainerWorker(Worker):
    """Worker for model training tasks.

    Policy model, value model should be subclasses of TrainingWorker.
    """

    def __init__(self) -> None:
        super().__init__()

    def train_step(self, global_step: int, batch: SampleTensorDict) -> dict[str, float] | None:
        raise NotImplementedError

    def eval_step(self, global_step: int, batch: SampleTensorDict) -> dict[str, float] | None:
        raise NotImplementedError

    def train(self, global_step: int, samples: SampleTensorDict, data_shuffle_seed: int = 0) -> tuple[int, dict[str, float]]:
        raise NotImplementedError

    def eval(self, global_step: int, samples: SampleTensorDict) -> dict[str, float]:
        raise NotImplementedError

    def compute_logprobs(self, samples: SampleTensorDict, batch_size: int) -> tuple[Tensor, list[GpuUsageInfo]]:
        raise NotImplementedError

    def compute_values(self, samples: SampleTensorDict, batch_size: int) -> tuple[Tensor, list[GpuUsageInfo]]:
        raise NotImplementedError
