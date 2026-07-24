import logging
from pathlib import Path

import pandas as pd

from axrl.configs import HfDataConfig
from axrl.data.conversation import Conversation, Message
from axrl.example.config_examples import CONV_EXAMPLE_PATH
from axrl.utils import zst_utils
from axrl.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def prepare_data(input_path: Path, output_path: Path) -> None:
    """Convert raw data to Conversation objects and save to parquet format."""
    logger.info(f"Preparing data from {input_path}")
    data = pd.read_parquet(path=input_path)
    logger.info(data.head())

    processed_data: list[Conversation] = []

    for row in data.itertuples():
        system_prompt: str = row.system  # type: ignore
        conversations_data: list[dict] = row.conversations  # type: ignore
        assert system_prompt and len(conversations_data) == 2
        messages = []
        messages.append(Message(role="system", content=system_prompt))

        for conv_msg in conversations_data:
            messages.append(Message(role=conv_msg["from"], content=conv_msg["value"]))

        processed_data.append(Conversation(messages=messages))

    logger.info(f"Processed {len(processed_data)} conversations.")
    zst_utils.save_zst(processed_data, output_path, verbose=True)
    assert processed_data


if __name__ == "__main__":
    setup_logger(level="info")
    output_path = CONV_EXAMPLE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        data_config = HfDataConfig(
            repo_id="bespokelabs/Bespoke-Stratos-17k",
            filename="data/train-00000-of-00001.parquet",
        )
        input_path = data_config.get_full_path()
        prepare_data(input_path, output_path)
