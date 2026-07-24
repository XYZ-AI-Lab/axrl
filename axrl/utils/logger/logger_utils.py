from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axrl.configs import LogLevel, MetricLoggerConfig
    from axrl.utils.logger.metric_logger import MetricLogger

logger = logging.getLogger(__name__)


def setup_logger(
    level: LogLevel = "info",
    fmt: str = "%(asctime)s - %(pathname)s:%(lineno)d - %(levelname)s - %(message)s",
) -> None:
    """Set up the global logger configuration."""
    logging.basicConfig(
        level=level.upper(),
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def get_metric_logger(metric_recorder_config: MetricLoggerConfig) -> MetricLogger:
    """Factory function to create a MetricLogger instance based on the configuration."""
    from axrl.utils.logger.console_logger import ConsoleLogger
    from axrl.utils.logger.tb_logger import TBLogger
    from axrl.utils.logger.wandb_logger import WandbLogger

    logger.info(f"Creating MetricLogger: {metric_recorder_config}")
    log_dir = metric_recorder_config.get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    if metric_recorder_config.logger_type == "tensorboard":
        return TBLogger(
            project_name=metric_recorder_config.project_name,
            group_name=metric_recorder_config.group_name,
            name=metric_recorder_config.name,
            log_dir=str(log_dir),
            run_id=metric_recorder_config.run_id,
        )
    if metric_recorder_config.logger_type == "wandb":
        return WandbLogger(
            project_name=metric_recorder_config.project_name,
            group_name=metric_recorder_config.group_name,
            name=metric_recorder_config.name,
            log_dir=str(log_dir),
            run_id=metric_recorder_config.run_id,
        )
    if metric_recorder_config.logger_type == "console":
        return ConsoleLogger(
            project_name=metric_recorder_config.project_name,
            group_name=metric_recorder_config.group_name,
            name=metric_recorder_config.name,
            log_dir=str(log_dir),
            run_id=metric_recorder_config.run_id,
        )

    raise ValueError(f"Unsupported metric recorder type: {metric_recorder_config.logger_type}")
