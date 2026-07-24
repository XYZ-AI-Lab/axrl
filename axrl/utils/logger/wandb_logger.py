from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, override

import wandb

from axrl.utils.logger.metric_logger import MetricLogger

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from pydantic import BaseModel

logger = logging.getLogger(__name__)


class WandbLogger(MetricLogger):
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
        """Initializes wandb logging and creates necessary directories."""
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set wandb directory to our log directory
        os.environ["WANDB_DIR"] = str(self.log_dir)
        # Initialize wandb run
        logger.info(f"WandbLogger name: {self.name}")
        self.run = wandb.init(
            project=self.project_name,
            name=self.name,
            group=self.group_name,
            dir=self.log_dir,
            id=self.run_id,
        )

    @override
    def log_config(self, config: BaseModel) -> None:
        flatted_config = self.flatten_config(config)
        assert self.run is not None
        self.run.config.update(flatted_config)

    @override
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Logs a single numerical value (scalar) to wandb.

        Args:
            name: Name of the metric
            value: Numerical value to log
            step: Step/iteration number
        """
        self.log_scalars({name: value}, step)

    @override
    def log_scalars(self, name_values: dict[str, float], step: int) -> None:
        """Logs multiple numerical values (scalars) at once."""
        self.run.log(name_values, step=step)
        self._log_scalars_to_console(name_values, step)

    @override
    def log_image(self, name: str, figure: Figure, step: int) -> None:
        """Logs a matplotlib figure as an image to Weights & Biases.

        Args:
            name: Name/tag for the image
            figure: A matplotlib Figure object
            step: Step/iteration number
        """
        self.run.log({name: wandb.Image(figure)}, step=step)

    def _log_scalars_to_console(self, name_values: dict[str, float], step: int) -> None:
        logger.info(f"[Wandb] Step {step}:")
        for name, value in name_values.items():
            print(f"{name:>60}: {value:.6f}")

    @override
    def close(self) -> None:
        """Performs cleanup and finishes the wandb run."""
        self.run.finish()
