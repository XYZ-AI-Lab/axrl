# Copied and adapted from https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py


import logging
import re
from typing import override

from axrl.verifier.base_verifier import BaseVerifier

logger = logging.getLogger(__name__)


class GSM8KVerifier(BaseVerifier):
    @override
    def verify(self, label: str | list[str], output_text: str, *, verbose: bool = False) -> float:
        assert isinstance(label, str)
        score = compute_score(solution_str=output_text, ground_truth=label)
        if verbose:
            logger.info(f"GSM8KVerifier: {score=}, {output_text=}, {label=}")
        return score


_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str: str, method: str = "strict") -> str | None:
    assert method in ["strict", "flexible"]

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    else:
        raise ValueError(f"Unknown method: {method}")
    return final_answer


def compute_score(solution_str: str, ground_truth: str, method: str = "strict", format_score: float = 0.0, score: float = 1.0) -> float:
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0

    if answer == ground_truth:
        return score

    return format_score
