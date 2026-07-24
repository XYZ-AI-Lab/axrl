from dataclasses import dataclass, field
from typing import Any, override

from axrl.processor.base_processor import BaseProcessor


@dataclass
class VerifierInput:
    label: str | list[str]
    output_text: str
    verbose: bool = False
    question: str = ""
    metadata: dict[str, Any] | None = None


@dataclass
class VerifierOutput:
    score: float
    infos: dict[str, Any] = field(default_factory=dict)


class BaseVerifier(BaseProcessor[VerifierInput, VerifierOutput]):
    def __init__(self, config: Any | None = None) -> None:
        super().__init__(config=config)

    def verify(self, label: str | list[str], output_text: str, *, verbose: bool = False) -> float:
        raise NotImplementedError

    @override
    def process(self, item: VerifierInput) -> VerifierOutput:
        assert item.label is not None
        assert item.output_text is not None
        score = self.verify(label=item.label, output_text=item.output_text, verbose=item.verbose)
        return VerifierOutput(score=score)
