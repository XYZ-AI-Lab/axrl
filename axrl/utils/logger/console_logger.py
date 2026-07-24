import logging
from typing import override

from matplotlib.figure import Figure
from pydantic import BaseModel

from axrl.utils.logger.metric_logger import MetricLogger

logger = logging.getLogger(__name__)


class ConsoleLogger(MetricLogger):
    """Wandb implementation of MetricLogger for logging metrics to Weights & Biases."""

    def __init__(
        self,
        project_name: str = "axrl",
        group_name: str = "default_group",
        name: str = "default_experiment",
        log_dir: str = "/data/axrl/metrics",
        run_id: str | None = None,
    ) -> None:
        super().__init__(
            project_name=project_name,
            group_name=group_name,
            name=name,
            log_dir=log_dir,
            run_id=run_id,
        )
        self._init()

    @override
    def _init(self) -> None:
        pass

    @override
    def log_scalar(self, name: str, value: float, step: int) -> None:
        logger.info(f"Step {step}: {name} = {value:.4f}")

    @override
    def log_scalars(self, name_values: dict[str, float], step: int) -> None:
        logger.info(f"Step {step}: Logging multiple scalars")
        for name, value in name_values.items():
            logger.info(f"  {name} = {value:.7f}")

    @override
    def close(self) -> None:
        pass

    @override
    def log_config(self, config: BaseModel) -> None:
        flatted_config = self.flatten_config(config)
        logger.info("Logging configuration:")
        for key, value in flatted_config.items():
            logger.info(f"  {key}: {value}")

    @override
    def log_image(self, name: str, figure: Figure, step: int) -> None:
        logger.warning("log_image is not implemented for ConsoleLogger. Skipping.")
