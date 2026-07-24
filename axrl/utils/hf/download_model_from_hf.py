import logging

from huggingface_hub import snapshot_download

from axrl.configs import ModelConfig
from axrl.utils.config_utils import load_and_validate_config

logger = logging.getLogger(__name__)


def download_model(config: ModelConfig) -> None:
    """Download a model from Hugging Face."""
    model_name = config.name
    model_dir = config.get_full_path()
    action = "Resuming/verifying" if model_dir.exists() else "Downloading"
    print(f"{action} model {model_name} to {model_dir}")
    downloaded_path = snapshot_download(
        repo_id=model_name,
        local_dir=model_dir,
    )
    logger.info(f"Downloadeded model from {model_name} to {downloaded_path}")


if __name__ == "__main__":
    config = load_and_validate_config(config_class=ModelConfig)
    download_model(config)
