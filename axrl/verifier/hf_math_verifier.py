import gc
import logging
from typing import override

from math_verify import parse, verify
from math_verify.errors import TimeoutException
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from axrl.verifier.base_verifier import BaseVerifier

logger = logging.getLogger(__name__)


class HfMathVerifier(BaseVerifier):
    @override
    def verify(self, label: str | list[str], output_text: str, *, verbose: bool = False) -> float:
        assert isinstance(label, str)
        timeout: int = 30
        score = 0.0
        ground_truth_boxed = "\\boxed{" + label + "}"
        max_len = max(4096, len(ground_truth_boxed) * 5)
        output_text = output_text[-max_len:]

        try:
            gc.disable()
            if verbose:
                logger.info(f"HfMathVerifier: verifying ground_truth_boxed={ground_truth_boxed!r}, output: {output_text!r}")
            parsed_label = parse(pred=ground_truth_boxed, extraction_config=(LatexExtractionConfig(),), parsing_timeout=timeout)
            parsed_output = parse(pred=output_text, extraction_config=(ExprExtractionConfig(), LatexExtractionConfig()), parsing_timeout=timeout)
            correct = verify(parsed_label, parsed_output, timeout_seconds=timeout)
            score = 1.0 if correct else 0.0
            if verbose:
                logger.info(f"HfMathVerifier: score={score}")
        except TimeoutException:
            logger.warning(
                "HfMathVerifier timed out for label %s and output %r; returning score 0.",
                label,
                output_text,
            )
        except Exception as exc:
            logger.debug("HfMathVerifier failed with %s for label %s; returning score 0.", exc, label)
        finally:
            gc.enable()

        return score
