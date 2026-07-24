from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import orjson

from axis_recipe.blackbox_rl.data import (
    BlackBoxEnvStatus,
    BlackBoxEnvTiming,
    BlackBoxInvalidToolCall,
    BlackBoxModelRequest,
    BlackBoxResponseMetric,
)
from axrl.data import Conversation, GenerationInput, GenerationOutput, Message, array_utils
from axrl.data.event_timing import EventTiming
from axrl.data.rollout_trace import GenerationInputPreparation, RolloutTrace
from axrl.openai_proxy import (
    OpenAIChatBuildResponseRequest,
    OpenAIChatConvertedRequest,
    OpenAIChatConvertRequest,
    OpenAIChatResponseResult,
    OpenAIPendingRequest,
    OpenAIProxySessionRegistry,
)
from axrl.utils.timer import SessionTimer
from axrl.verifier.base_verifier import VerifierInput

if TYPE_CHECKING:
    from collections.abc import Mapping

    from axrl.metrics.response_metric import ResponseMetric
    from axrl.openai_proxy.chat_adapter import OpenAIChatAdapterInput, OpenAIChatAdapterOutput
    from axrl.processor.processor_pool import ProcessorPool
    from axrl.verifier.base_verifier import VerifierOutput
    from axrl.worker.infer_worker import InferWorker

logger = logging.getLogger(__name__)

type Payload = dict[str, object]
type ModelIOEvent = dict[str, object]


class BlackBoxEnv(ABC):
    """Generic black-box RL env for runtimes that call an OpenAI-compatible model API.

    The base class owns the reusable protocol loop:

    1. create an OpenAI proxy session;
    2. launch an external runtime in a subclass;
    3. wait for the runtime's next model request through the OpenAI proxy;
    4. convert the OpenAI request to ``GenerationInput`` on CPU;
    5. let the rollout worker produce ``GenerationOutput``;
    6. pack the output back into an OpenAI-compatible response and unblock the runtime;
    7. stop the runtime and collect verifier text when the trajectory ends.

    Subclasses provide only runtime-specific behavior: how to launch/stop the
    runtime, how to identify normal terminal actions, and how to calculate a
    trajectory score when the default verifier path is not enough. For example,
    ``OpenHandsEnv`` supplies OpenHands process handling and solution-file
    collection, while request capture, adapter conversion, basic scoring, and
    model-boundary logic stay here.
    """

    def __init__(
        self,
        *,
        conv: Conversation,
        label: str | list[str],
        registry: OpenAIProxySessionRegistry,
        adapter: ProcessorPool[OpenAIChatAdapterInput, OpenAIChatAdapterOutput],
        score_provider: InferWorker[VerifierInput, VerifierOutput],
        metric_calculator: InferWorker[GenerationOutput, ResponseMetric],
        initial_request_timeout_seconds: float,
        request_timeout_seconds: float,
        max_model_calls: int,
        max_length: int,
        pad_token_id: int = 0,
        runtime_name: str = "blackbox",
    ) -> None:
        self.original_conv = conv.deep_copy()
        self.conv = conv.deep_copy()
        self.label = label
        self.registry = registry
        self.adapter = adapter
        self.score_provider = score_provider
        self.metric_calculator = metric_calculator
        self.initial_request_timeout_seconds = initial_request_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.max_model_calls = max_model_calls
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.runtime_name = runtime_name
        self.session_id = conv.gen_state.session_id or conv.conversation_id
        assert self.session_id, "BlackBoxEnv requires a rollout session id."
        self.trace: RolloutTrace | None = None
        self.outputs: list[GenerationOutput] = []
        self.model_io_events: list[ModelIOEvent] = []
        self.current_model_request: BlackBoxModelRequest | None = None
        self.done = False
        self._status = BlackBoxEnvStatus()
        self._timing = BlackBoxEnvTiming.start(session_id=self.session_id, runtime_name=self.runtime_name)

    async def start(self) -> Conversation:
        attempt = 0
        while True:
            attempt += 1
            try:
                await self.registry.create_session(self.session_id)
                with SessionTimer(self.session_id, "async", f"{self.runtime_name}: launch runtime"):
                    await self.launch_runtime()
                self.current_model_request = await self._wait_initial_request_or_runtime_done()
                assert self.current_model_request is not None, "BlackBoxEnv start requires an initial model request from the runtime."
                observation, trace, is_prompt_too_long = self._build_observation(self.current_model_request)
                assert not is_prompt_too_long, "Initial black-box model request should fit within the model context length."
                self.trace = trace
                return observation
            except TimeoutError:
                reason = self._get_model_request_timeout_reason(timeout_seconds=self.initial_request_timeout_seconds)
                logger.warning(reason)
                await self.registry.close_session(self.session_id)
                self.current_model_request = None
                await self.terminate_runtime()
            except AssertionError:
                raise
            except Exception as exc:
                reason = self._launch_failure_message(attempt=attempt, exc=exc)
                logger.warning(reason, exc_info=True)
                await self.registry.close_session(self.session_id)
                self.current_model_request = None
                await self.terminate_runtime()

    async def step(self, action: GenerationOutput) -> tuple[Conversation, float, bool, RolloutTrace | None, BlackBoxResponseMetric | None]:
        assert self.trace is not None
        assert self.current_model_request is not None
        model_request = self.current_model_request
        call_index = len(self.outputs)
        self.outputs.append(action)
        invalid_tool_call = _validate_generation_tool_calls(action, generation_input=model_request.converted.generation_input)
        if invalid_tool_call is not None:
            return await self._retry_after_invalid_tool_call(
                action=action,
                invalid_tool_call=invalid_tool_call,
                call_index=call_index,
            )

        self.trace.append_assistant_message(action)
        response_result = await self._build_openai_response_from_generation_output(action)
        self._record_model_output(
            call_index=call_index,
            generation_output=action,
            response_json=response_result.response_json,
        )
        await model_request.pending_request.respond(response_result.response_json)
        self.current_model_request = None
        if self._is_normal_finish_action(action):
            await self._terminate_by_finish_action()
            return await self._finish()
        if len(self.outputs) >= self.max_model_calls:
            await self._terminate_by_max_model_calls()
            return await self._finish()
        try:
            next_model_request = await self._wait_request_or_runtime_done()
        except TimeoutError:
            await self._terminate_by_request_timeout(call_index=call_index)
            return await self._finish()
        if next_model_request is None:
            return await self._finish()
        model_request = next_model_request
        self.current_model_request = model_request
        observation, trace, is_prompt_too_long = self._build_observation(model_request)
        self.trace = trace
        if is_prompt_too_long:
            await self._terminate_by_prompt_too_long(
                observation=observation,
                model_request=model_request,
            )
            return await self._finish()
        return observation, 0.0, False, None, None

    async def _retry_after_invalid_tool_call(
        self,
        *,
        action: GenerationOutput,
        invalid_tool_call: BlackBoxInvalidToolCall,
        call_index: int,
    ) -> tuple[Conversation, float, bool, RolloutTrace | None, BlackBoxResponseMetric | None]:
        assert self.trace is not None
        logger.warning(
            "%s received invalid tool call for session %s call %d: %s",
            self.runtime_name,
            self.session_id,
            call_index,
            invalid_tool_call.message,
        )
        self._status.invalid_tool_calls += 1
        trace_action = copy.copy(action)
        trace_action.tool_calls = None
        self.trace.append_assistant_message(trace_action)
        self._record_model_output(
            call_index=call_index,
            generation_output=action,
            invalid_tool_call=invalid_tool_call,
        )
        assert self.current_model_request is not None, "Invalid tool retry requires the current model request context."
        pending = self.current_model_request.pending_request
        if len(self.outputs) >= self.max_model_calls:
            await self._terminate_by_max_model_calls(
                reason=f"{self._get_max_model_calls_reason()} after invalid tool call.",
                pending_request=pending,
                status_code=400,
            )
            return await self._finish()

        self.trace.conversation.add_message(Message(role="user", content=_invalid_tool_call_feedback(invalid_tool_call)))
        converted = await self._convert_retry_request()
        model_request = BlackBoxModelRequest(pending_request=pending, converted=converted)
        self.current_model_request = model_request
        observation, trace, is_prompt_too_long = self._build_observation(model_request, timestamp=_now_iso())
        self.trace = trace
        if is_prompt_too_long:
            await self._terminate_by_prompt_too_long(
                observation=observation,
                model_request=model_request,
            )
            return await self._finish()
        return observation, 0.0, False, None, None

    async def finish(self) -> tuple[Conversation, float, bool, RolloutTrace | None, BlackBoxResponseMetric]:
        return await self._finish()

    def _set_terminal_score(self, score: float, reason: str) -> None:
        self._status.forced_score = score
        self._status.forced_score_reason = reason

    @abstractmethod
    async def launch_runtime(self) -> None:
        """Prepare and launch the external black-box runtime."""

    @abstractmethod
    async def terminate_runtime(self) -> None:
        """Stop the runtime and collect any runtime output needed for scoring."""

    async def calculate_trajectory_score(self) -> float | None:
        """Return a runtime-specific score, or ``None`` to use the default verifier."""
        return None

    def _is_normal_finish_action(self, _action: GenerationOutput) -> bool:
        """Return whether ``action`` should be treated as a normal finish."""
        return False

    async def _wait_request_or_runtime_done(self) -> BlackBoxModelRequest | None:
        """Wait for the next model request, or ``None`` if the runtime finished.

        Most runtimes expose termination through a model-visible finish action,
        so the default behavior is identical to waiting for another request.
        Subclasses whose Python runtime can complete after the latest response
        may override this and return ``None`` to end the trajectory immediately.
        """
        return await self._wait_request()

    async def _wait_initial_request_or_runtime_done(self) -> BlackBoxModelRequest | None:
        """Wait for the first model request, or ``None`` if the runtime finished."""
        return await self._wait_request(timeout_seconds=self.initial_request_timeout_seconds)

    def build_response_metric(self, base_metric: ResponseMetric, score: float) -> BlackBoxResponseMetric:
        total_seconds = self._total_elapsed_seconds()
        llm_turn_latencies = [output.e2e_elapsed_seconds for output in self.outputs]
        output_token_counts = [float(len(output.output_ids)) for output in self.outputs]
        llm_total_seconds = sum(llm_turn_latencies)
        env_overhead_seconds = max(total_seconds - llm_total_seconds, 0.0)
        metric = BlackBoxResponseMetric(
            **base_metric.__dict__,
            num_model_calls=len(self.outputs),
            normal_finish=int(self._status.normal_finish),
            initial_request_timeout=int(self._status.initial_request_timeout),
            request_timeout=int(self._status.request_timeout),
            verifier_timeout=int(self._status.verifier_timeout),
            num_invalid_tool_calls=self._status.invalid_tool_calls,
            blackbox_total_seconds=total_seconds,
            blackbox_llm_total_seconds=llm_total_seconds,
            blackbox_env_overhead_seconds=env_overhead_seconds,
            blackbox_env_overhead_ratio=env_overhead_seconds / total_seconds if total_seconds > 0 else 0.0,
            blackbox_wait_request_total_seconds=sum(self._timing.wait_request_seconds),
            blackbox_wait_request_mean_seconds=_mean_or_zero(self._timing.wait_request_seconds),
            blackbox_adapter_convert_total_seconds=sum(self._timing.adapter_convert_seconds),
            blackbox_adapter_convert_mean_seconds=_mean_or_zero(self._timing.adapter_convert_seconds),
            blackbox_adapter_build_response_total_seconds=sum(self._timing.adapter_build_response_seconds),
            blackbox_adapter_build_response_mean_seconds=_mean_or_zero(self._timing.adapter_build_response_seconds),
            blackbox_prepare_generation_total_seconds=sum(self._timing.prepare_generation_seconds),
            blackbox_prepare_generation_mean_seconds=_mean_or_zero(self._timing.prepare_generation_seconds),
            blackbox_verifier_seconds=self._timing.verifier_seconds,
            blackbox_metric_seconds=self._timing.metric_seconds,
            blackbox_drain_runtime_seconds=self._timing.terminate_runtime_seconds,
            llm_turn_min_latency=_min_or_zero(llm_turn_latencies),
            llm_turn_mean_latency=_mean_or_zero(llm_turn_latencies),
            llm_turn_max_latency=_max_or_zero(llm_turn_latencies),
            llm_turn_min_output_tokens=int(_min_or_zero(output_token_counts)),
            llm_turn_mean_output_tokens=_mean_or_zero(output_token_counts),
            llm_turn_max_output_tokens=int(_max_or_zero(output_token_counts)),
        )
        metric.score = score
        return metric

    def _total_elapsed_seconds(self) -> float:
        return self._timing.total_elapsed_seconds()

    async def _wait_request(self, *, timeout_seconds: float | None = None) -> BlackBoxModelRequest:
        """Wait for one OpenAI proxy request and convert it for model generation."""
        timeout_seconds = timeout_seconds if timeout_seconds is not None else self.request_timeout_seconds
        with SessionTimer(self.session_id, "async", f"{self.runtime_name}: wait model request") as timer:
            pending_request = await self.registry.wait_for_request(self.session_id, timeout_seconds=timeout_seconds)
        self._timing.wait_request_seconds.append(timer.elapsed_seconds)
        converted = await self._build_model_request_from_body(
            pending_request.body,
            timer_label="convert OpenAI request",
        )
        return BlackBoxModelRequest(pending_request=pending_request, converted=converted)

    def _build_observation(
        self,
        model_request: BlackBoxModelRequest,
        *,
        timestamp: str | None = None,
    ) -> tuple[Conversation, RolloutTrace | None, bool]:
        """Return the next observation, updated trace, and whether the prompt is too long."""
        observation = self._build_raw_observation(model_request)
        is_prompt_too_long = self._get_prompt_too_long_reason(model_request.converted.generation_input) is not None
        if is_prompt_too_long:
            return observation, self.trace, True
        prepared_observation, trace, routing_preparation = self._prepare_observation_for_generation(
            observation,
            model_request.converted.generation_input,
            trace=self.trace,
        )
        self.conv = prepared_observation
        self._record_model_input(
            call_index=len(self.outputs),
            pending_request=model_request.pending_request,
            converted=model_request.converted,
            routing_preparation=routing_preparation,
            timestamp=timestamp,
        )
        return prepared_observation, trace, False

    def _build_raw_observation(self, model_request: BlackBoxModelRequest) -> Conversation:
        converted = model_request.converted
        conv = Conversation(
            messages=copy.deepcopy(converted.messages),
            conversation_id=self.original_conv.conversation_id,
            source=self.original_conv.source,
        )
        conv.extra.update(self.original_conv.extra)
        conv.extra["openai_proxy_generation_input"] = converted.generation_input
        conv.extra["openai_proxy_response_context"] = converted.context
        conv.extra["openai_proxy_pending_request"] = model_request.pending_request
        conv.gen_state.session_id = converted.generation_input.session_id
        conv.gen_state.input_ids = converted.generation_input.input_ids
        conv.gen_state.sampling_config = converted.generation_input.sampling_config
        conv.gen_state.tools = converted.generation_input.tools
        conv.gen_state.tool_choice = converted.generation_input.tool_choice
        conv.gen_state.tool_call_parser = converted.generation_input.tool_call_parser
        conv.gen_state.capture_routing = self.original_conv.gen_state.capture_routing
        return conv

    def _get_prompt_too_long_reason(self, generation_input: GenerationInput) -> str | None:
        token_count = len(generation_input.input_ids)
        if token_count <= self.max_length:
            return None
        return (
            f"{self.runtime_name} prompt exceeded max_length for session {self.session_id}: "
            f"generation prompt tokens ({token_count}) exceed max_length ({self.max_length})."
        )

    def _get_model_request_timeout_reason(
        self,
        *,
        timeout_seconds: float | None = None,
        after_model_response: bool = False,
    ) -> str:
        timeout_seconds = timeout_seconds if timeout_seconds is not None else self.request_timeout_seconds
        reason = f"{self.runtime_name} did not send a model request within {timeout_seconds:.1f}s"
        if after_model_response:
            reason = f"{reason} after receiving the previous model response"
        return f"{reason}."

    def _get_max_model_calls_reason(self) -> str:
        return f"{self.runtime_name} reached max_model_calls={self.max_model_calls} for session {self.session_id}"

    def _get_finish_action_reason(self) -> str:
        return f"{self.runtime_name} received finish action for session {self.session_id}."

    def _launch_failure_message(self, *, attempt: int, exc: BaseException) -> str:
        exception_text = _preview_text(str(exc).strip() or type(exc).__name__, limit=500)
        return f"{self.runtime_name} launch attempt {attempt} failed for session {self.session_id}; restarting runtime: {exception_text}"

    async def _terminate_by_finish_action(self) -> None:
        reason = self._get_finish_action_reason()
        self._status.normal_finish = True
        self._status.normal_finish_reason = reason
        await self._close_session_and_terminate(reason=reason, log_warning=False)

    async def _terminate_by_max_model_calls(
        self,
        *,
        reason: str | None = None,
        pending_request: OpenAIPendingRequest | None = None,
        status_code: int = 500,
    ) -> None:
        await self._close_session_and_terminate(
            reason=reason or self._get_max_model_calls_reason(),
            pending_request=pending_request,
            status_code=status_code,
        )

    async def _terminate_by_prompt_too_long(
        self,
        *,
        observation: Conversation,
        model_request: BlackBoxModelRequest,
    ) -> None:
        reason = self._get_prompt_too_long_reason(model_request.converted.generation_input)
        assert reason is not None
        await self._close_session_and_terminate(
            reason=reason,
            score=0.0,
            observation=observation,
            pending_request=model_request.pending_request,
            status_code=413,
        )

    async def _terminate_by_request_timeout(self, *, call_index: int) -> None:
        reason = self._get_model_request_timeout_reason(after_model_response=True)
        self._status.request_timeout = True
        self._attach_request_timeout_to_last_output(call_index=call_index, reason=reason)
        assert self.trace is not None
        self.trace.conversation.add_message(Message(role="user", content=_request_timeout_feedback(reason)))
        await self._close_session_and_terminate(reason=reason, score=0.0, status_code=504)

    async def _close_session_and_terminate(
        self,
        *,
        reason: str,
        score: float | None = None,
        observation: Conversation | None = None,
        pending_request: OpenAIPendingRequest | None = None,
        status_code: int = 500,
        log_warning: bool = True,
    ) -> None:
        """Mark this rollout done, close its proxy session, and stop the runtime.

        ``score`` forces a terminal score for errors such as prompt overflow.
        ``pending_request`` is answered with an error before runtime shutdown so
        an in-flight OpenAI request is not left waiting for proxy timeout.
        """
        if log_warning:
            logger.warning(reason)
        self.done = True
        if observation is not None:
            self.conv = observation
        if score is not None:
            self._set_terminal_score(score, reason)
        if pending_request is not None:
            await pending_request.fail(reason, status_code=status_code)
        await self.registry.close_session(self.session_id)
        self.current_model_request = None
        await self.terminate_runtime()

    def _prepare_observation_for_generation(
        self,
        observation: Conversation,
        generation_input: GenerationInput,
        *,
        trace: RolloutTrace | None,
    ) -> tuple[Conversation, RolloutTrace, GenerationInputPreparation]:
        """Prepare the returned observation so ``RolloutAgent.act`` sends the right prompt.

        The OpenAI adapter renders each black-box model call as a full prompt.
        ``RolloutTrace.prepare_generation_input`` may preserve only the
        token-identical routing prefix and then rewrites
        ``generation_input.input_ids`` plus ``routed_expert_start_index``.
        Keep the observation's ``gen_state`` aligned with that prepared prompt,
        because ``RolloutAgent.act`` builds the actual ``GenerationInput`` from
        the returned conversation.
        """
        trace_observation = _trace_conversation_from_observation(observation)
        with SessionTimer(self.session_id, "sync", f"{self.runtime_name}: prepare generation input") as timer:
            if trace is None:
                trace = RolloutTrace(trace_observation, token_in_token_out=True, max_length=self.max_length)
            else:
                trace.conversation = trace_observation
            routing_preparation = trace.prepare_generation_input(generation_input)
        self._timing.prepare_generation_seconds.append(timer.elapsed_seconds)
        observation.gen_state.input_ids = trace.conversation.gen_state.input_ids
        observation.gen_state.captured_routing_rows = trace.conversation.gen_state.captured_routing_rows
        return observation, trace, routing_preparation

    async def _finish(self) -> tuple[Conversation, float, bool, RolloutTrace | None, BlackBoxResponseMetric]:
        self.done = True
        await self.registry.close_session(self.session_id)
        with SessionTimer(self.session_id, "async", f"{self.runtime_name}: terminate runtime") as terminate_timer:
            await self.terminate_runtime()
        self._timing.terminate_runtime_seconds += terminate_timer.elapsed_seconds
        if self._status.forced_score is None:
            with SessionTimer(self.session_id, "async", f"{self.runtime_name}: verify result") as verifier_timer:
                try:
                    score = await self.calculate_trajectory_score()
                    if score is None:
                        verifier_output = await self.score_provider.generate(VerifierInput(label=self.label, output_text=self._model_output_text()))
                        score = verifier_output.score
                except Exception as exc:
                    if not _is_timeout_exception(exc):
                        raise
                    self._status.verifier_timeout = True
                    exception_text = _preview_text(str(exc).strip() or type(exc).__name__, limit=500)
                    reason = f"Verifier timed out for {self.runtime_name} session {self.session_id}: {exception_text}"
                    logger.warning(reason)
                    self._set_terminal_score(0.0, reason)
                    score = 0.0
            self._timing.verifier_seconds += verifier_timer.elapsed_seconds
        else:
            score = self._status.forced_score
        metric_output = self._merged_generation_output(self._model_output_text())
        with SessionTimer(self.session_id, "async", f"{self.runtime_name}: calculate response metric") as metric_timer:
            base_metric = await self.metric_calculator.generate(req=metric_output)
        self._timing.metric_seconds += metric_timer.elapsed_seconds
        metric = self.build_response_metric(base_metric, score)
        metric.score = score
        if self.trace is not None:
            self.conv = self.trace.conversation
        self.conv.extra["openai_io_events"] = list(self.model_io_events)
        self.conv.extra["blackbox_normal_finish"] = self._status.normal_finish
        if self._status.normal_finish_reason is not None:
            self.conv.extra["blackbox_normal_finish_reason"] = self._status.normal_finish_reason
        if self._status.forced_score_reason is not None:
            self.conv.extra["blackbox_forced_score_reason"] = self._status.forced_score_reason
        return self.conv, score, True, self.trace, metric

    def _model_output_text(self) -> str:
        return "\n\n".join(output.output_text for output in self.outputs if output.output_text)

    def _merged_generation_output(self, final_text: str) -> GenerationOutput:
        output_ids = np.concatenate([output.output_ids for output in self.outputs]) if self.outputs else np.empty(0, dtype=np.int32)
        output_logprobs = np.concatenate([output.output_logprobs for output in self.outputs]) if self.outputs else np.empty(0, dtype=np.float32)
        return GenerationOutput(
            session_id=self.session_id,
            output_ids=array_utils.as_i32(output_ids),
            output_logprobs=array_utils.as_f32(output_logprobs),
            output_text=final_text,
            output_text_with_special_tokens="\n".join(output.output_text_with_special_tokens for output in self.outputs),
            cached_tokens=sum(output.cached_tokens for output in self.outputs),
            finish_reason=self.outputs[-1].finish_reason if self.outputs else "abort",
            e2e_elapsed_seconds=sum(output.e2e_elapsed_seconds for output in self.outputs),
            stop_reason=self.outputs[-1].stop_reason if self.outputs else None,
            retry=sum(output.retry for output in self.outputs),
            event_timing=max(
                self.outputs,
                key=lambda output: output.event_timing.driver_worker_overhead_seconds or 0.0,
            ).event_timing
            if self.outputs
            else EventTiming(),
        )

    def _record_model_input(
        self,
        *,
        call_index: int,
        pending_request: OpenAIPendingRequest,
        converted: OpenAIChatConvertedRequest,
        routing_preparation: GenerationInputPreparation,
        timestamp: str | None = None,
    ) -> None:
        self.model_io_events.append(
            {
                "kind": "model_input",
                "timestamp": timestamp or _iso_from_epoch(float(pending_request.created_at)),
                "call_index": call_index,
                "request_id": pending_request.request_id,
                "session_id": self.session_id,
                "payload": {
                    "generation_input": converted.generation_input,
                    "generation_input_preparation": routing_preparation,
                    "response_context": converted.context,
                },
            }
        )

    def _record_model_output(
        self,
        *,
        call_index: int,
        generation_output: GenerationOutput,
        response_json: Mapping[str, object] | None = None,
        invalid_tool_call: BlackBoxInvalidToolCall | None = None,
    ) -> None:
        payload: Payload = {"generation_output": generation_output}
        if response_json is not None:
            payload["openai_response"] = response_json
        if invalid_tool_call is not None:
            payload["invalid_tool_call"] = invalid_tool_call
        self.model_io_events.append(
            {
                "kind": "model_output",
                "timestamp": _now_iso(),
                "call_index": call_index,
                "session_id": self.session_id,
                "payload": payload,
            }
        )

    def _attach_request_timeout_to_last_output(self, *, call_index: int, reason: str) -> None:
        for event in reversed(self.model_io_events):
            if event.get("kind") != "model_output" or event.get("call_index") != call_index:
                continue
            payload = event.get("payload")
            if isinstance(payload, dict):
                payload["request_timeout"] = reason
                return
        self.model_io_events.append(
            {
                "kind": "model_output",
                "timestamp": _now_iso(),
                "call_index": call_index,
                "session_id": self.session_id,
                "payload": {"request_timeout": reason},
            }
        )

    async def _convert_retry_request(self) -> OpenAIChatConvertedRequest:
        assert self.current_model_request is not None, "Retry conversion requires the current model request context."
        assert self.trace is not None
        request_json = copy.deepcopy(self.current_model_request.converted.context.request_json)
        request_json["messages"] = [message.to_dict() for message in self.trace.conversation.messages]
        return await self._build_model_request_from_body(
            request_json,
            timer_label="convert internal retry request",
        )

    async def _build_model_request_from_body(self, request_body: Mapping[str, object], *, timer_label: str) -> OpenAIChatConvertedRequest:
        with SessionTimer(self.session_id, "async", f"{self.runtime_name}: {timer_label}") as timer:
            result = await self.adapter.generate(
                OpenAIChatConvertRequest(
                    session_id=f"{self.session_id}:call:{len(self.outputs)}",
                    request_json=dict(request_body),
                )
            )
        self._timing.adapter_convert_seconds.append(timer.elapsed_seconds)
        assert isinstance(result, OpenAIChatConvertedRequest)
        return result

    async def _build_openai_response_from_generation_output(self, generation_output: GenerationOutput) -> OpenAIChatResponseResult:
        assert self.current_model_request is not None, "Runtime response building requires the current model request context."
        with SessionTimer(self.session_id, "async", f"{self.runtime_name}: build OpenAI response") as timer:
            result = await self.adapter.generate(
                OpenAIChatBuildResponseRequest(
                    context=self.current_model_request.converted.context,
                    generation_output=generation_output,
                )
            )
        self._timing.adapter_build_response_seconds.append(timer.elapsed_seconds)
        assert isinstance(result, OpenAIChatResponseResult)
        return result


def _trace_conversation_from_observation(observation: Conversation) -> Conversation:
    conv = Conversation(
        messages=copy.deepcopy(observation.messages),
        conversation_id=observation.conversation_id,
        source=observation.source,
        gen_state=copy.deepcopy(observation.gen_state),
    )
    conv.extra.update({key: copy.deepcopy(value) for key, value in observation.extra.items() if key != "openai_proxy_pending_request"})
    return conv


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _iso_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(value).astimezone().isoformat()


def _validate_generation_tool_calls(output: GenerationOutput, *, generation_input: GenerationInput) -> BlackBoxInvalidToolCall | None:
    expect_tool_calls = bool(generation_input.tools) and generation_input.tool_choice != "none"
    if not output.tool_calls:
        if expect_tool_calls and _looks_like_unparsed_tool_call(output):
            return BlackBoxInvalidToolCall(
                message="Model emitted tool-call syntax, but no valid tool call could be parsed.",
                tool_name=None,
                arguments_preview=_preview_text(output.output_text or output.output_text_with_special_tokens),
            )
        return None
    for tool_call in output.tool_calls:
        arguments = tool_call.arguments.strip()
        if not arguments:
            return BlackBoxInvalidToolCall(
                message="Tool call arguments must be a non-empty JSON object string.",
                tool_name=tool_call.name,
                arguments_preview=tool_call.arguments,
            )
        try:
            decoded = orjson.loads(arguments)
        except orjson.JSONDecodeError as exc:
            return BlackBoxInvalidToolCall(
                message=f"Tool call arguments must be valid JSON: {exc}",
                tool_name=tool_call.name,
                arguments_preview=_preview_text(tool_call.arguments),
            )
        if not isinstance(decoded, dict):
            return BlackBoxInvalidToolCall(
                message="Tool call arguments must decode to a JSON object.",
                tool_name=tool_call.name,
                arguments_preview=_preview_text(tool_call.arguments),
            )
    return None


def _looks_like_unparsed_tool_call(output: GenerationOutput) -> bool:
    if output.finish_reason == "tool_calls":
        return True
    text = f"{output.output_text}\n{output.output_text_with_special_tokens}".lower()
    return "<tool_call" in text or "</tool_call" in text


def _invalid_tool_call_feedback(result: BlackBoxInvalidToolCall) -> str:
    target = f" for `{result.tool_name}`" if result.tool_name else ""
    preview = f" Arguments preview: {result.arguments_preview}" if result.arguments_preview else ""
    return (
        f"<information>Warning: The previous tool call{target} had invalid JSON object arguments, "
        f"so no tool was executed. {result.message}{preview} "
        "Please issue the tool call again with valid JSON object arguments.</information>\n\n"
    )


def _request_timeout_feedback(reason: str) -> str:
    return (
        "<information>Warning: The previous assistant response caused the black-box runtime to stall. "
        f"{reason} The trajectory was terminated with score 0.</information>\n\n"
    )


def _preview_text(value: str, *, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}... [truncated]"


def _is_timeout_exception(exc: BaseException) -> bool:
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, TimeoutError):
            return True
        ray_cause = getattr(current, "cause", None)
        if isinstance(ray_cause, BaseException):
            pending.append(ray_cause)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return False


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _min_or_zero(values: list[float]) -> float:
    return float(np.min(values)) if values else 0.0


def _max_or_zero(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0
