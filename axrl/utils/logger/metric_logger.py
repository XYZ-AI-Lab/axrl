from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from matplotlib.figure import Figure


REDACTED_CONFIG_VALUE = "[REDACTED]"
SENSITIVE_CONFIG_KEY_MARKERS = (
    "api_key",
    "apikey",
    "api_token",
    "access_token",
    "refresh_token",
    "auth_token",
    "bearer_token",
    "credential",
    "credentials",
    "password",
    "passwd",
    "secret",
)


def _is_sensitive_config_key(key: str) -> bool:
    normalized_key = key.lower().replace("-", "_")
    return any(marker in normalized_key for marker in SENSITIVE_CONFIG_KEY_MARKERS)


class MetricLogger(ABC):
    """Abstract base class for logging metrics."""

    def __init__(
        self,
        project_name: str,
        group_name: str,
        name: str,
        log_dir: str,
        run_id: str | None = None,
    ) -> None:
        """Initializes the MetricLogger."""
        self.name = name
        self.group_name = group_name
        self.project_name = project_name
        self.log_dir = Path(log_dir).absolute()
        self.run_id = run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _init(self) -> None:
        """Initializes the logger, setting up necessary directories and configurations."""
        pass

    @abstractmethod
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Logs a single numerical value (scalar) over time."""
        pass

    @abstractmethod
    def log_scalars(self, name_values: dict[str, float], step: int) -> None:
        """Logs multiple numerical values (scalars) at once."""
        pass

    @abstractmethod
    def log_image(self, name: str, figure: Figure, step: int) -> None:
        """Logs a matplotlib figure as an image."""
        pass

    @abstractmethod
    def log_config(self, config: BaseModel) -> None:
        """Logs configuration parameters."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Performs any necessary cleanup or finalization for the logger."""
        pass

    @staticmethod
    def flatten_config(config: BaseModel) -> dict[str, str | int | float | bool | None]:
        """Convert a nested Pydantic model into a flat dictionary with dot-separated keys.

        This utility function transforms hierarchical configuration objects into a format
        suitable for logging systems like WandB or TensorBoard that prefer flat key-value pairs.
        """
        flat_config: dict[str, str | int | float | bool | None] = {}

        def _flatten(prefix: str, obj: BaseModel | dict) -> None:
            data = obj.model_dump() if isinstance(obj, BaseModel) else obj

            for field_name, field_value in data.items():
                key = f"{prefix}.{field_name}" if prefix else field_name

                # Redact credentials before console, TensorBoard, or WandB can record them.
                if _is_sensitive_config_key(key):
                    flat_config[key] = REDACTED_CONFIG_VALUE
                elif field_value is None:
                    flat_config[key] = None
                elif isinstance(field_value, dict):
                    _flatten(key, field_value)
                elif isinstance(field_value, list):
                    flat_config[key] = ", ".join(str(item) for item in field_value) if field_value else ""
                else:
                    flat_config[key] = field_value

        _flatten("", config)
        return flat_config
