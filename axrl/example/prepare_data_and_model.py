import logging

from axrl.datasets import get_dataset
from axrl.trainer.grpo_exp_config import GrpoExperimentConfig
from axrl.utils import setup_logger
from axrl.utils.config_utils import load_and_validate_config
from axrl.utils.hf.download_model_from_hf import download_model

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logger("info")
    config = load_and_validate_config(
        GrpoExperimentConfig,
        config_path="axis_recipe/grpo_gsm8k/grpo_config.yaml",
        print_configs=True,
    )
    assert config.train_datasets is not None
    assert config.test_datasets is not None

    for item in config.train_datasets:
        dataset = get_dataset(item)
        dataset.initialize()

    for eval_config in config.test_datasets:
        dataset = get_dataset(eval_config)
        dataset.initialize()

    model_config = config.rollout_worker.model
    model_dir = model_config.get_full_path()
    if model_dir.exists():
        logger.info(f"Model already exists at {model_dir}.")
        return
    download_model(config=model_config)
    logger.info(f"Model downloaded to {model_dir}.")
    logger.info("Finished preparing GSM8K GRPO artifacts.")


if __name__ == "__main__":
    main()
