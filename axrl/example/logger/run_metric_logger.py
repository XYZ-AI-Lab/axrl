import logging

from axrl.configs import MetricLoggerConfig
from axrl.utils.logger import get_metric_logger

logger = logging.getLogger(__name__)


def test_metric_logger() -> None:
    """Test function for the TensorBoard and Wandbb metric recorder."""
    # test both TensorBoard and Wandb metric recorders
    configs = [
        MetricLoggerConfig(
            logger_type=logger_type,  # type: ignore
        )
        for logger_type in ["tensorboard", "wandb", "console"]
    ]
    for config in configs:
        recorder = get_metric_logger(config)
        for step in range(50):
            recorder.log_scalar("train/scalar_1", step * 0.1, step=step)
            recorder.log_scalars({"val/scalar_1": step * 0.1, "val/scalar_2": step * 0.2}, step=step)
        recorder.log_config(config)
        recorder.close()
        logger.info(f"{config.logger_type} test completed successfully, output dir: {recorder.log_dir}")  # type: ignore


if __name__ == "__main__":
    from axrl.utils import setup_logger

    setup_logger("info")
    test_metric_logger()
