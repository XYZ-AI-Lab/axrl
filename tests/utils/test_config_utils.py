from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import Field

from axrl.configs import StrictBaseModel
from axrl.utils.config_utils import load_and_validate_config

FILE_CONFIG_YAML = """
a:
    b: 1
flag: false
name: from-file
items: [1]
"""


class TestNestedConfig(StrictBaseModel):
    b: int = 0


class TestConfig(StrictBaseModel):
    a: TestNestedConfig = TestNestedConfig()
    flag: bool = False
    name: str = "default"
    items: list[int] = Field(default_factory=list)


def _config_env(env: dict[str, str] | None = None) -> dict[str, str]:
    clean_env = {key: value for key, value in os.environ.items() if not key.startswith("AXRL__")}
    if env is not None:
        clean_env.update(env)
    return clean_env


class ConfigUtilsTest(unittest.TestCase):
    def test_load_and_validate_config_env_overrides_cli_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(FILE_CONFIG_YAML)

            with (
                patch.object(
                    sys,
                    "argv",
                    [
                        "prog",
                        "--a.b=2",
                        "--flag=true",
                        "--name=from-cli",
                    ],
                ),
                patch.dict(
                    os.environ,
                    _config_env(
                        {
                            "AXRL__a__b": "3",
                            "AXRL__name": "from-env",
                            "AXRL__items": "[4,5]",
                        }
                    ),
                    clear=True,
                ),
            ):
                config = load_and_validate_config(TestConfig, str(config_path))

            assert config.a.b == 3
            assert config.flag is True
            assert config.name == "from-env"
            assert config.items == [4, 5]

    def test_load_and_validate_config_env_config_path_has_highest_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file_path = tmp_path / "file.yaml"
            cli_path = tmp_path / "cli.yaml"
            env_path = tmp_path / "env.yaml"
            file_path.write_text("name: from-file\n")
            cli_path.write_text("name: from-cli-file\n")
            env_path.write_text("name: from-env-file\n")

            with (
                patch.object(sys, "argv", ["prog", f"--config_path={cli_path}"]),
                patch.dict(os.environ, _config_env({"AXRL__config_path": str(env_path)}), clear=True),
            ):
                config = load_and_validate_config(TestConfig, str(file_path))

            assert config.name == "from-env-file"


if __name__ == "__main__":
    unittest.main()
