import asyncio
from typing import override

from axrl.data import array_utils
from axrl.data.conversation import Conversation, GenerationState, Message
from axrl.data.generation import GenerationInput, GenerationOutput
from axrl.envs.math_env import MathEnv
from axrl.metrics.response_metric import ResponseMetric
from axrl.processor.base_processor import BaseProcessor
from axrl.processor.processor_pool import ProcessorPool
from axrl.verifier.base_verifier import VerifierInput, VerifierOutput


class _ScoreProvider(BaseProcessor[VerifierInput, VerifierOutput]):
    @override
    def process(self, item: VerifierInput) -> VerifierOutput:
        return VerifierOutput(score=1.0)


class _MetricCalculator(BaseProcessor[GenerationOutput, ResponseMetric]):
    @override
    def process(self, item: GenerationOutput) -> ResponseMetric:
        return ResponseMetric(
            token_count=len(item.output_ids),
            token_unique_ratio=1.0,
            word_length_max=1,
            line_length_max=1,
            ngram_repetition=0.0,
            reasoning_behavior_backtracking=0.0,
            reasoning_behavior_verification=0.0,
            reasoning_behavior_causal=0.0,
            rollout_cached_tokens=0,
            rollout_num_retry=0,
            rollout_e2e_elapsed_seconds=0.0,
            rollout_finish_reason_stop=1,
            rollout_finish_reason_length=0,
            rollout_finish_reason_tool_calls=0,
            rollout_finish_reason_function_call=0,
            rollout_finish_reason_content_filter=0,
        )


class _ConvTokenizer(BaseProcessor[Conversation, GenerationInput]):
    @override
    def process(self, item: Conversation) -> GenerationInput:
        assert item.gen_state.input_ids is not None
        return GenerationInput(session_id="test", input_ids=array_utils.as_i32(item.gen_state.input_ids))


def test_math_env_return_sample_false_advances_conversation_without_trace() -> None:
    conv = Conversation(
        messages=[Message(role="user", content="prompt")],
        gen_state=GenerationState(input_ids=array_utils.as_i32([1, 2])),
    )
    score_provider = ProcessorPool[VerifierInput, VerifierOutput](_ScoreProvider, config=None, num_processors=1)
    metric_calculator = ProcessorPool[GenerationOutput, ResponseMetric](_MetricCalculator, config=None, num_processors=1)
    conv_tokenizer = ProcessorPool[Conversation, GenerationInput](_ConvTokenizer, config=None, num_processors=1)
    try:
        env = MathEnv(
            score_provider=score_provider,
            metric_calculator=metric_calculator,
            conv_tokenizer=conv_tokenizer,
            conv=conv,
            label="answer",
            max_length=16,
            return_sample=False,
        )
        output = GenerationOutput(
            session_id="s",
            output_ids=array_utils.as_i32([3, 4]),
            output_logprobs=array_utils.as_f32([0.0, 0.0]),
            output_text="ok",
            output_text_with_special_tokens="ok",
            cached_tokens=0,
            finish_reason="stop",
            e2e_elapsed_seconds=0.0,
            stop_reason=None,
            retry=0,
            assistant_boundary_token_id=99,
        )

        observation, score, done, trace, metric = asyncio.run(env.step(output))
    finally:
        score_provider.shutdown()
        metric_calculator.shutdown()
        conv_tokenizer.shutdown()

    assert done
    assert score == 1.0
    assert metric.score == 1.0
    assert trace is None
    assert observation.messages[-1] == Message(role="assistant", content="ok")
    assert observation.gen_state.input_ids is not None
    assert observation.gen_state.input_ids.tolist() == [1, 2, 3, 4, 99]
