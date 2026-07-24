from __future__ import annotations

from typing import Any


class BaseProcessor[InT, OutT]:
    """Base class for processors that handle various data processing tasks."""

    def __init__(self, config: Any | None = None) -> None:
        self.config = config

    def process(self, item: InT) -> OutT:
        raise NotImplementedError
