import logging
from typing import override

from axrl.verifier.base_verifier import BaseVerifier
from axrl.verifier.dapo_utils import compute_score

logger = logging.getLogger(__name__)


class DapoVerifier(BaseVerifier):
    @override
    def verify(self, label: str | list[str], output_text: str, *, verbose: bool = False) -> float:
        assert isinstance(label, str)
        score = compute_score(solution_str=output_text, ground_truth=label, strict_box_verify=True)
        if verbose:
            logger.info(f"DapoVerifier: {score=}, {output_text=}, {label=}")
        return score
