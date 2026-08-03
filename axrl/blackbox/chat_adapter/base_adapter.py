from axrl.data.generation import GenerationInput, GenerationOutput

from typing import Any

class BaseAdapter:

    def to_generation_input(self, request: Any) -> GenerationInput:
        raise NotImplementedError

    def to_response(self, output: GenerationOutput) -> Any:
        raise NotImplementedError
