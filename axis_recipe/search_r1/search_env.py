from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import numpy as np

from axrl.data import Conversation, GenerationOutput, Message, array_utils
from axrl.data.rollout_trace import RolloutTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from axis_recipe.search_r1.search_client import RetrievedDocument, SearchRequestResult
    from axrl.data.generation import ToolCall
    from axrl.worker.infer_worker import InferWorker
from axis_recipe.search_r1.search_r1_verifier import SearchR1Verifier, SearchR1VerifierConfig
from axrl.envs.base_env import BaseEnv
from axrl.metrics.response_metric import ResponseMetric
from axrl.verifier.base_verifier import VerifierInput, VerifierOutput

logger = logging.getLogger(__name__)


@dataclass
class SearchResponseMetric(ResponseMetric):
    num_searches: int = 0
    num_queries: int = 0
    num_thinks: int = 0
    num_answers: int = 0
    num_turns: int = 0
    em_score: float = 0.0
    tool_call_valid_rate: float = 0.0
    search_latency_seconds_mean: float = 0.0
    search_transport_failure_rate: float = 0.0


class SearchEnv(BaseEnv):
    def __init__(
        self,
        conv: Conversation,
        label: str | list[str],
        score_provider: InferWorker[VerifierInput, VerifierOutput],
        metric_calculator: InferWorker[GenerationOutput, ResponseMetric],
        appended_message_tokenizer: InferWorker[Message, NDArray[np.int32]],
        search_client: InferWorker[str, SearchRequestResult],
        pad_token_id: int,
        max_length: int,
        search_limit: int = 5,
    ) -> None:
        super().__init__(conv)
        self.score_provider = score_provider
        self.metric_calculator = metric_calculator
        self.appended_message_tokenizer = appended_message_tokenizer
        self.label = label
        self.search_client = search_client
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.search_limit = search_limit
        self.answer_token_reserve = 128
        assert conv.gen_state.input_ids is not None and len(conv.gen_state.input_ids) > 0
        self.trace = RolloutTrace(conv, token_in_token_out=True, max_length=max_length)
        self.conv = self.trace.conversation
        self.search_queries: list[str | None] = []
        self.search_r1_verifier_input = ""
        self.outputs: list[GenerationOutput] = []
        self.num_valid_searches = 0
        self.search_latency_ms_total = 0.0
        self.search_transport_failure_count = 0
        self.num_tool_calls = 0
        self.num_tool_calls_valid = 0
        self.em_score_provider: SearchR1Verifier = SearchR1Verifier(
            SearchR1VerifierConfig(
                structure_format_score=0.0,
                retrieval_score=0.0,
            )
        )

    @override
    async def step(self, action: GenerationOutput) -> tuple[Conversation, float, bool, RolloutTrace | None, SearchResponseMetric | None]:
        """Apply an assistant turn, optionally fetch search results, and return the (conversation, reward, done, trace, metric) tuple."""
        assert self.conv.gen_state.input_ids is not None
        output = action
        self.outputs.append(output)
        self.update_with_output(output)
        tool_calls = output.tool_calls
        num_searches = len(self.search_queries)

        has_answer_tag = "<answer>" in output.output_text and "</answer>" in output.output_text
        reached_answer_budget_limit = self._reached_answer_budget_limit(len(self.trace.token_ids))
        if reached_answer_budget_limit:
            logger.info(f"Reached answer budget limit: {len(self.trace.token_ids)} >= {self.max_length - self.answer_token_reserve}")
        if not tool_calls or num_searches > self.search_limit or reached_answer_budget_limit:
            return await self._finish()

        search_query = self._get_search_query(tool_calls[0])

        if has_answer_tag:
            search_result = self.get_search_answer_coexistence_message()
        elif len(tool_calls) > 1:
            search_result = self.get_multiple_search_message()
        elif search_query is None:
            self.num_tool_calls += 1
            search_result = self.get_invalid_search_message()
        elif num_searches < self.search_limit:
            self.num_tool_calls += 1
            self.num_tool_calls_valid += 1
            search_result = await self.search(search_query)
        else:
            assert num_searches == self.search_limit
            search_result = self.get_search_limit_reached_message()

        self.search_queries.append(search_query)
        should_skip_search_result = await self.update_with_search_result(search_result, tool_calls[0])
        if should_skip_search_result:
            logger.info(f"Skipping search result append to reserve {self.answer_token_reserve} answer tokens within max length {self.max_length}")
            return await self._finish()
        return self.conv, 0.0, False, None, None

    async def _finish(self) -> tuple[Conversation, float, bool, RolloutTrace, SearchResponseMetric]:
        """Compute final metric/trace and return a done=True result."""
        search_response_metric = await self.calculate_merged_metric()
        score = search_response_metric.score
        assert score is not None
        return self.conv, score, True, self.trace, search_response_metric

    async def calculate_merged_metric(self) -> SearchResponseMetric:
        assert self.conv.gen_state.session_id
        merged_output = GenerationOutput(
            session_id=self.conv.gen_state.session_id,
            output_ids=array_utils.as_i32(np.concatenate([output.output_ids for output in self.outputs])),
            output_text="\n".join([output.output_text for output in self.outputs]),
            output_text_with_special_tokens="\n".join([output.output_text_with_special_tokens for output in self.outputs]),
            output_logprobs=array_utils.as_f32(np.concatenate([output.output_logprobs for output in self.outputs])),
            cached_tokens=sum(output.cached_tokens for output in self.outputs),
            retry=sum(output.retry for output in self.outputs),
            e2e_elapsed_seconds=sum(output.e2e_elapsed_seconds for output in self.outputs),
            finish_reason=self.outputs[-1].finish_reason,
            stop_reason=self.outputs[-1].stop_reason,
            event_timing=max(
                self.outputs,
                key=lambda output: output.event_timing.driver_worker_overhead_seconds or 0.0,
            ).event_timing,
        )

        response_metric = await self.metric_calculator.generate(req=merged_output)
        num_searches = self.num_valid_searches

        assert isinstance(self.label, list)
        verifier_output = await self.score_provider.generate(VerifierInput(label=self.label, output_text=self.search_r1_verifier_input))
        score = verifier_output.score
        em_verifier_output = self.em_score_provider.process(VerifierInput(label=self.label, output_text=self.search_r1_verifier_input))
        em_score = em_verifier_output.score
        total_search_attempts = self.num_valid_searches
        search_response_metric = SearchResponseMetric(
            num_searches=num_searches,
            num_thinks=merged_output.output_text.count("<think>"),
            num_answers=merged_output.output_text.count("<answer>"),
            num_queries=sum(len(o.tool_calls) for o in self.outputs if o.tool_calls),
            num_turns=len(self.outputs),
            em_score=em_score,
            search_latency_seconds_mean=(self.search_latency_ms_total / 1000.0) / total_search_attempts if total_search_attempts > 0 else 0.0,
            tool_call_valid_rate=self.num_tool_calls_valid / self.num_tool_calls if self.num_tool_calls > 0 else 0.0,
            search_transport_failure_rate=self.search_transport_failure_count / total_search_attempts if total_search_attempts > 0 else 0.0,
            **response_metric.__dict__,
        )
        search_response_metric.score = score
        if random.random() < 0.001:
            logger.info(f"Final conversation:\n{self.conv}\nSearchResponseMetric: {search_response_metric}")
        return search_response_metric

    def update_with_output(self, output: GenerationOutput) -> None:
        self.conv = self.trace.append_assistant_message(output)
        self.search_r1_verifier_input += f"\n{self._get_search_r1_compatible_output(output.output_text, output.tool_calls)}\n"

    def _get_search_r1_compatible_output(
        self,
        output_text: str,
        tool_calls: list[ToolCall] | None = None,
    ) -> str:
        """Convert tool-call output into the legacy Search-R1 verifier format."""
        if not tool_calls:
            return output_text

        search_tags: list[str] = []
        for tool_call in tool_calls:
            search_query = self._get_search_query(tool_call) or ""
            search_tags.append(f"<search>{search_query}</search>")
        return output_text + "".join(search_tags)

    def _get_search_query(self, tool_call: ToolCall) -> str | None:
        """Extract the search query from a tool call, or None if arguments are invalid."""
        stripped = tool_call.arguments.strip()
        if not stripped:
            return None
        try:
            arguments = json.loads(stripped)
        except Exception:
            return None
        if not isinstance(arguments, dict):
            return None
        query = arguments.get("query")
        return str(query) if query is not None else None

    def _reached_answer_budget_limit(self, token_count: int) -> bool:
        return token_count >= self.max_length - self.answer_token_reserve

    async def update_with_search_result(self, search_result: str, tool_call: ToolCall) -> bool:
        tool_msg = Message(role="tool", content=search_result, tool_call_id=tool_call.id)
        new_tokens = await self.appended_message_tokenizer.generate(tool_msg)
        if self._reached_answer_budget_limit(len(self.trace.token_ids) + len(new_tokens)):
            return True
        assert new_tokens[0] != self.outputs[-1].output_ids[-1], (
            "New tokens from search result should not start with the same token as the last generated token."
        )
        self.conv = self.trace.append_user_or_tool_message(content=search_result, tokens=new_tokens, tool_call_id=tool_call.id)
        self.search_r1_verifier_input += f"\n{search_result}\n"
        return False

    async def search(self, search_query: str) -> str:
        self.num_valid_searches += 1
        request_result = await self.search_client.generate(search_query)
        self.search_latency_ms_total += request_result.latency_ms
        if request_result.error_type is not None:
            self.search_transport_failure_count += 1
            return (
                "<information>Search request failed due to retrieval service error. "
                "Please try another query or continue with the available evidence.</information>\n\n"
            )

        results: list[RetrievedDocument] = request_result.documents
        text = "\n".join([f"Doc {idx + 1} (Title: {doc.title}) {doc.text}" for idx, doc in enumerate(results)])
        text = f"<information>{text}</information>\n\n"
        return text

    def get_search_limit_reached_message(self) -> str:
        message = (
            "<information>Search limit reached. No more searches available. "
            "Please provide your final answer using the information gathered.</information>\n\n"
        )
        return message

    def get_search_answer_coexistence_message(self) -> str:
        message = (
            "<information>Warning: Both a search tool call and an answer were detected. "
            "Please either use the search tool or provide an <answer>, not both.</information>\n\n"
        )
        return message

    def get_invalid_search_message(self) -> str:
        message = (
            "<information>Warning: The search tool call arguments were invalid. "
            "Please issue one search tool call with JSON arguments containing a non-empty query.</information>\n\n"
        )
        return message

    def get_multiple_search_message(self) -> str:
        message = (
            "<information>Warning: Multiple search tool calls detected in your response. "
            "Please issue only one search tool call per response and wait for the "
            "search result before searching again.</information>\n\n"
        )
        return message
