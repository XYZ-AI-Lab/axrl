from typing import TYPE_CHECKING

from axrl.processor.base_processor import BaseProcessor

if TYPE_CHECKING:
    from axrl.processor.processor_pool import ProcessorPool

__all__ = ["BaseProcessor", "ProcessorPool"]


def __getattr__(name: str) -> object:
    if name == "ProcessorPool":
        from axrl.processor.processor_pool import ProcessorPool

        return ProcessorPool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
