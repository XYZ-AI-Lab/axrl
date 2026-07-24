from __future__ import annotations

import logging
from typing import TYPE_CHECKING, override

from torch.utils.tensorboard import SummaryWriter

from axrl.utils.logger.metric_logger import MetricLogger

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from pydantic import BaseModel


logger = logging.getLogger(__name__)


class TBLogger(MetricLogger):
    """TensorBoard implementation of MetricLogger for logging metrics to TensorBoard."""

    def __init__(
        self,
        project_name: str = "axrl",
        group_name: str = "default_group",
        name: str = "default_experiment",
        log_dir: str = "/data/log",
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
        """Initializes TensorBoard logging and creates necessary directories."""
        full_log_path = self.log_dir / self.project_name / self.group_name
        full_log_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized TensorBoard logger at {full_log_path}")

        # Initialize TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=str(full_log_path), flush_secs=30)

    @override
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Logs a single numerical value (scalar) to TensorBoard.

        Args:
            name: Name of the metric
            value: Numerical value to log
            step: Step/iteration number
        """
        self.writer.add_scalar(tag=name, scalar_value=value, global_step=step)

    @override
    def log_scalars(self, name_values: dict[str, float], step: int) -> None:
        """Logs multiple numerical values (scalars) to TensorBoard.

        Args:
            name_values: Dictionary of metric names and their corresponding values
            step: Step/iteration number
        """
        logger.info(f"[Tensorboard] Step {step}:")
        for key, value in name_values.items():
            self.writer.add_scalar(tag=f"{key}", scalar_value=value, global_step=step)
            print(f"{key:>60}: {value:.6f}")

    @override
    def log_image(self, name: str, figure: Figure, step: int) -> None:
        """Logs a matplotlib figure as an image to TensorBoard.

        Args:
            name: Name/tag for the image
            figure: A matplotlib Figure object
            step: Step/iteration number
        """
        self.writer.add_figure(tag=name, figure=figure, global_step=step)

    @override
    def log_config(self, config: BaseModel) -> None:
        """Logs the configuration dictionary to TensorBoard as hyperparameters."""
        hparams = self.flatten_config(config)
        self.writer.add_hparams(
            run_name=".",
            hparam_dict=hparams,
            metric_dict={},
        )

    @override
    def close(self) -> None:
        """Performs cleanup and closes the TensorBoard writer."""
        self.writer.flush()
        self.writer.close()
