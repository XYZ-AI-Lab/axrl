import logging

from huggingface_hub import hf_hub_download

from axrl.configs import AXRL_DIR, HfDataConfig
from axrl.utils.config_utils import load_and_validate_config

logger = logging.getLogger(__name__)


def download_data_file(config: HfDataConfig) -> None:
    """Download data from Hugging face hub."""
    target_file = config.get_full_path()
    if target_file.exists():
        logger.info(f"Data file already exists at {target_file}, skipping download.")
        return
    local_dir = AXRL_DIR.data
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {config.repo_id}/{config.filename} to {local_dir}")
    local_dir = local_dir / config.repo_id
    downloaded_filepath = hf_hub_download(
        repo_id=config.repo_id,
        filename=config.filename,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    logger.info(f"Downloadeded file to {downloaded_filepath}")


if __name__ == "__main__":
    config = load_and_validate_config(config_class=HfDataConfig)
    download_data_file(config)
