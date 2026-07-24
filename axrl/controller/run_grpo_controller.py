from __future__ import annotations

import asyncio
import logging

from axrl.controller.grpo_controller import GrpoController
from axrl.trainer.grpo_exp_config import GrpoExperimentConfig
from axrl.utils import setup_logger
from axrl.utils.config_utils import load_and_validate_config

logger = logging.getLogger(__name__)


async def main() -> None:
    config = load_and_validate_config(
        GrpoExperimentConfig,
        config_path="axis_recipe/grpo/grpo_config.yaml",  # can be overridden by CLI arg --config_path
        print_configs=True,
    )
    controller = GrpoController(config)
    await controller.start()


if __name__ == "__main__":
    setup_logger(level="info")
    asyncio.run(main())
