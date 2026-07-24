import logging
import random
from typing import override

from rich.pretty import pprint

from axrl.data import Conversation
from axrl.data.generation import GenerationInput, GenerationOutput
from axrl.data.rollout_trace import RolloutTrace
from axrl.envs.base_env import BaseEnv
from axrl.metrics.response_metric import ResponseMetric
from axrl.verifier.base_verifier import VerifierInput, VerifierOutput
from axrl.worker.infer_worker import InferWorker

logger = logging.getLogger(__name__)


class MathEnv(BaseEnv):
    def __init__(
        self,
        score_provider: InferWorker[VerifierInput, VerifierOutput],
        metric_calculator: InferWorker[GenerationOutput, ResponseMetric],
        conv_tokenizer: InferWorker[Conversation, GenerationInput],
        conv: Conversation,
        label: str | list[str],
        *,
        max_length: int,
        return_sample: bool,
    ) -> None:
        super().__init__(conv)
        self.score_provider = score_provider
        self.metric_calculator = metric_calculator
        self.conv_tokenizer = conv_tokenizer
        self.label = label
        self.max_length = max_length
        self.return_sample = return_sample

    @override
    async def step(self, action: GenerationOutput) -> tuple[Conversation, float, bool, RolloutTrace | None, ResponseMetric]:
        """Returns: observation, reward, done, info."""
        done = True  # Single step environment, return done=True after one step
        verifier_output = await self.score_provider.generate(VerifierInput(label=self.label, output_text=action.output_text))
        score = verifier_output.score
        response_metric = await self.metric_calculator.generate(action)
        response_metric.score = score
        trace = self._build_rollout_trace(self.conv, action)
        self.conv = trace.conversation
        if random.random() < 0.0001:
            logger.info("[MathEnv] Debug Info:")
            pprint(response_metric)
            pprint(self.conv.messages)

        return self.conv, score, done, trace if self.return_sample else None, response_metric

    def _build_rollout_trace(self, conv: Conversation, output: GenerationOutput) -> RolloutTrace:
        """Single-call rollout as a ``RolloutTrace`` with one turn sample."""
        assert conv.gen_state.input_ids is not None
        assert output.output_logprobs is not None
        assert len(output.output_logprobs) == len(output.output_ids)
        trace = RolloutTrace(conv, token_in_token_out=True, max_length=self.max_length)
        trace.append_assistant_message(output)
        return trace
