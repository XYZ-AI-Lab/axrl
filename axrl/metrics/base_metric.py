from typing import Any


class BaseMetric:
    def compute(self, input_data: Any) -> dict[str, float]:
        """Compute metrics based on the input data."""
        raise NotImplementedError
