import logging
from typing import override

from axis_recipe.search_r1.qa_em_format import compute_score_em
from axis_recipe.search_r1.search_r1_config import SearchR1VerifierConfig
from axrl.verifier.base_verifier import BaseVerifier

logger = logging.getLogger(__name__)


class SearchR1Verifier(BaseVerifier):
    def __init__(self, config: SearchR1VerifierConfig) -> None:
        super().__init__(config)
        self.config: SearchR1VerifierConfig = config

    @override
    def verify(self, label: str | list[str], output_text: str, *, verbose: bool = False) -> float:
        assert isinstance(label, list)
        return compute_score_em(
            solution_str=output_text,
            ground_truth={"target": label},
            structure_format_score=self.config.structure_format_score,
            retrieval_score=self.config.retrieval_score,
        )
