from typing import TYPE_CHECKING, Any

from axrl.utils.logger.logger_utils import get_metric_logger, setup_logger

if TYPE_CHECKING:
    from axrl.utils.logger.logger_buffer import LoggerBuffer
    from axrl.utils.logger.metric_logger import MetricLogger
    from axrl.utils.logger.tb_logger import TBLogger
    from axrl.utils.logger.wandb_logger import WandbLogger

__all__ = [
    "LoggerBuffer",
    "MetricLogger",
    "TBLogger",
    "WandbLogger",
    "get_metric_logger",
    "setup_logger",
]


def __getattr__(name: str) -> Any:
    if name == "LoggerBuffer":
        from axrl.utils.logger.logger_buffer import LoggerBuffer

        return LoggerBuffer
    if name == "MetricLogger":
        from axrl.utils.logger.metric_logger import MetricLogger

        return MetricLogger
    if name == "TBLogger":
        from axrl.utils.logger.tb_logger import TBLogger

        return TBLogger
    if name == "WandbLogger":
        from axrl.utils.logger.wandb_logger import WandbLogger

        return WandbLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
