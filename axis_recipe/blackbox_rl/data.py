from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from axrl.metrics.response_metric import ResponseMetric
from axrl.utils.timer import SessionTimer

if TYPE_CHECKING:
    from axrl.openai_proxy import OpenAIChatConvertedRequest, OpenAIPendingRequest


@dataclass
class BlackBoxResponseMetric(ResponseMetric):
    num_model_calls: int = 0
    normal_finish: int = 0
    initial_request_timeout: int = 0
    request_timeout: int = 0
    verifier_timeout: int = 0
    num_invalid_tool_calls: int = 0
    blackbox_total_seconds: float = 0.0
    blackbox_llm_total_seconds: float = 0.0
    blackbox_env_overhead_seconds: float = 0.0
    blackbox_env_overhead_ratio: float = 0.0
    blackbox_wait_request_total_seconds: float = 0.0
    blackbox_wait_request_mean_seconds: float = 0.0
    blackbox_adapter_convert_total_seconds: float = 0.0
    blackbox_adapter_convert_mean_seconds: float = 0.0
    blackbox_adapter_build_response_total_seconds: float = 0.0
    blackbox_adapter_build_response_mean_seconds: float = 0.0
    blackbox_prepare_generation_total_seconds: float = 0.0
    blackbox_prepare_generation_mean_seconds: float = 0.0
    blackbox_verifier_seconds: float = 0.0
    blackbox_metric_seconds: float = 0.0
    blackbox_drain_runtime_seconds: float = 0.0
    llm_turn_min_latency: float = 0.0
    llm_turn_mean_latency: float = 0.0
    llm_turn_max_latency: float = 0.0
    llm_turn_min_output_tokens: int = 0
    llm_turn_mean_output_tokens: float = 0.0
    llm_turn_max_output_tokens: int = 0


@dataclass
class BlackBoxInvalidToolCall:
    message: str
    tool_name: str | None = None
    arguments_preview: str | None = None


@dataclass
class BlackBoxModelRequest:
    pending_request: OpenAIPendingRequest
    converted: OpenAIChatConvertedRequest


@dataclass
class BlackBoxEnvStatus:
    initial_request_timeout: bool = False
    request_timeout: bool = False
    normal_finish: bool = False
    normal_finish_reason: str | None = None
    forced_score: float | None = None
    forced_score_reason: str | None = None
    invalid_tool_calls: int = 0
    verifier_timeout: bool = False


@dataclass
class BlackBoxEnvTiming:
    total_timer: SessionTimer
    total_timer_stopped: bool = False
    wait_request_seconds: list[float] = field(default_factory=list)
    adapter_convert_seconds: list[float] = field(default_factory=list)
    adapter_build_response_seconds: list[float] = field(default_factory=list)
    prepare_generation_seconds: list[float] = field(default_factory=list)
    verifier_seconds: float = 0.0
    metric_seconds: float = 0.0
    terminate_runtime_seconds: float = 0.0

    @classmethod
    def start(cls, *, session_id: str, runtime_name: str) -> BlackBoxEnvTiming:
        timing = cls(total_timer=SessionTimer(session_id, "async", f"{runtime_name}: total"))
        timing.total_timer.start()
        return timing

    def total_elapsed_seconds(self) -> float:
        if not self.total_timer_stopped:
            self.total_timer.stop()
            self.total_timer_stopped = True
        return max(self.total_timer.elapsed_seconds, 0.0)
