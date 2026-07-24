import os
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError
from rich.pretty import pprint

from axrl.configs import StrictBaseModel

ENV_CONFIG_PREFIX = "AXRL__"


def _model_dump(config: StrictBaseModel) -> dict:
    """Override model_dump to handle torch dtypes that can't be serialized to YAML."""
    data = config.model_dump()

    def convert_torch_dtypes(obj: Any) -> Any:
        """Recursively convert torch dtypes to strings."""
        obj_type = type(obj)
        if obj_type.__module__ == "torch" and obj_type.__qualname__ == "dtype":
            return str(obj)
        if isinstance(obj, dict):
            return {k: convert_torch_dtypes(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_torch_dtypes(item) for item in obj]
        return obj

    converted_data = convert_torch_dtypes(data)
    return converted_data if isinstance(converted_data, dict) else data


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary using dot notation for keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _format_value(value: Any) -> str:
    """Format a value for command-line argument style."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ",".join(_format_value(v) for v in value) + "]"
    return str(value)


def config_to_args(config: StrictBaseModel) -> dict[str, str]:
    """Convert a config object to a flattened dict with CLI-style keys and string values.

    Args:
        config: The config object to convert (must be a StrictBaseModel/Pydantic model)

    Returns:
        A flattened dict with "--" prefixed keys and string values, sorted alphabetically

    Example:
        >>> from axrl.trainer.grpo_exp_config import GrpoExperimentConfig
        >>> config = GrpoExperimentConfig()
        >>> args_dict = config_to_args(config)
        >>> args_dict["--megatron_worker.optimizer.bf16"]
        'true'
    """
    config_dict = _model_dump(config)
    flat_dict = _flatten_dict(config_dict)
    return {f"--{k}": _format_value(v) for k, v in sorted(flat_dict.items())}


def save_to_yaml(config: StrictBaseModel, path: Path) -> None:
    config_dict = _model_dump(config)
    with path.open("w") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False, indent=4)


def _convert_torch_dtype_strings(obj: Any) -> Any:
    """Recursively convert torch dtype strings back to torch dtype objects."""
    if isinstance(obj, str) and obj.startswith("torch."):
        # Try to convert torch dtype strings back to actual dtypes
        import torch

        dtype_map = {
            "torch.float32": torch.float32,
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.int32": torch.int32,
            "torch.int64": torch.int64,
            "torch.uint8": torch.uint8,
            "torch.bool": torch.bool,
        }
        return dtype_map.get(obj, obj)
    if isinstance(obj, dict):
        return {k: _convert_torch_dtype_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_torch_dtype_strings(item) for item in obj]
    return obj


def _load_env_config(prefix: str = ENV_CONFIG_PREFIX) -> DictConfig:
    """Load config overrides from environment variables.

    Environment variables follow the pattern `AXRL__a__b=2`, which maps to
    the same nested key as the CLI flag `--a.b=2`.
    """
    dotlist = []
    for key, value in sorted(os.environ.items()):
        if not key.startswith(prefix):
            continue
        config_key = key.removeprefix(prefix)
        if not config_key:
            continue
        dotlist.append(f"{config_key.replace('__', '.')}={value}")
    return OmegaConf.from_dotlist(dotlist) if dotlist else OmegaConf.create()


def load_and_validate_config[ConfigType: StrictBaseModel](
    config_class: type[ConfigType],
    config_path: str | None = None,
    *,
    load_env_config: bool = True,
    print_configs: bool = False,
) -> ConfigType:
    """Load and validate configurations from a file and command line arguments.

    Args:
        config_class: The config class to validate
        config_path: Optional path to a YAML configuration file. Can also be specified
            via CLI using --config_path=<path>
        load_env_config: If True, loads config overrides from environment variables
            with the `AXRL__` prefix
        print_configs: If True, prints the loaded configuration

    Returns:
        A validated instance of the specified config_class

    Notes:
        - If config_path is None, creates a default configuration
        - Command line arguments override values from the config file
        - Environment variables with the `AXRL__` prefix override CLI values
        - The final configuration is validated against the provided config_class
    """
    cli_cfg: DictConfig = OmegaConf.from_cli()
    cli_cfg = OmegaConf.create({str(k).lstrip("-"): v for k, v in cli_cfg.items()})  # remove leading '-'
    env_cfg = _load_env_config() if load_env_config else OmegaConf.create()

    cli_config_path = cli_cfg.pop("config_path", None)
    if cli_config_path is not None:
        if config_path is not None:
            print(f"Overriding config_path from CLI: {config_path}")
        config_path = cli_config_path

    env_config_path = env_cfg.pop("config_path", None)
    if env_config_path is not None:
        if config_path is not None:
            print(f"Overriding config_path from env: {config_path}")
        config_path = env_config_path

    if config_path is not None:
        print(f"Loading configuration from: {config_path}")

    omegaconf_cfg = OmegaConf.load(config_path) if config_path else OmegaConf.create()
    merged_cfg = OmegaConf.merge(omegaconf_cfg, cli_cfg, env_cfg)
    python_dict_cfg = OmegaConf.to_container(merged_cfg, resolve=True)

    # Convert torch dtype strings back to actual torch dtype objects

    python_dict_cfg = _convert_torch_dtype_strings(python_dict_cfg)

    try:
        validated_config = config_class.model_validate(python_dict_cfg, strict=True)
    except ValidationError as e:
        print("Validation failed:")
        for err in e.errors():
            # print(f"Field: {err['loc']}, Error: {err['msg']}")  # should with color orange
            pprint(err)
        # raise from None to suppress trackback as we already printed it
        raise ValueError("Configuration validation failed") from None
    if print_configs:
        print("Loaded config:")
        pprint(validated_config)
    return validated_config
