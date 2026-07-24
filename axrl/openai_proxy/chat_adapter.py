from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import orjson

from axrl.data import GenerationInput, GenerationOutput, Message, array_utils
from axrl.processor.base_processor import BaseProcessor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import torch
    from numpy.typing import NDArray
    from pydantic import BaseModel
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    from sglang.srt.managers.template_detection import ReasoningToggleConfig

    from axrl.configs import ModelConfig


@dataclass
class OpenAIChatResponseContext:
    request_json: dict[str, Any]
    prompt_tokens: int
    response_id: str
    resolved_tool_call_parser: str | None = None
    resolved_reasoning_parser: str | None = None
    force_reasoning: bool = False


@dataclass
class OpenAIChatAdapterConfig:
    # Valid parser names are SGLang registry keys:
    # - tool_call_parser: FunctionCallParser.ToolCallParserEnum
    #   https://github.com/sgl-project/sglang/blob/127b9e3283f7c2a43234b852ff5c9f1796d53624/python/sglang/srt/function_call/function_call_parser.py#L56-L84
    # - reasoning_parser: ReasoningParser.DetectorMap
    #   https://github.com/sgl-project/sglang/blob/127b9e3283f7c2a43234b852ff5c9f1796d53624/python/sglang/srt/parser/reasoning_parser.py#L610-L631
    model: ModelConfig
    tool_call_parser: str | None = None  # None, "auto", or SGLang tool parser, e.g. "qwen", "glm45".
    reasoning_parser: str | None = None  # None, "auto", or SGLang reasoning parser, e.g. "qwen3", "deepseek-r1".


@dataclass
class OpenAIChatConvertRequest:
    session_id: str
    request_json: dict[str, Any]


@dataclass
class OpenAIChatConvertedRequest:
    session_id: str
    generation_input: GenerationInput
    messages: list[Message]
    context: OpenAIChatResponseContext


@dataclass
class OpenAIChatBuildResponseRequest:
    context: OpenAIChatResponseContext
    generation_output: GenerationOutput


@dataclass
class OpenAIChatResponseResult:
    response_json: dict[str, Any]


OpenAIChatAdapterInput = OpenAIChatConvertRequest | OpenAIChatBuildResponseRequest
OpenAIChatAdapterOutput = OpenAIChatConvertedRequest | OpenAIChatResponseResult
ChatFinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call", "abort"]


class OpenAIChatAdapter(BaseProcessor[OpenAIChatAdapterInput, OpenAIChatAdapterOutput]):
    """Convert OpenAI chat requests to AXRL token generation and back.

    The adapter intentionally stays CPU-only. It uses the model tokenizer's
    chat template to render OpenAI messages into prompt token IDs. Sampling
    stays under the GRPO controller's global ``SamplingConfig``; OpenAI
    request sampling fields are intentionally ignored. The SGLang worker
    remains responsible only for token generation, logprobs, tool-call parsing,
    and routing capture.
    """

    def __init__(
        self,
        config: OpenAIChatAdapterConfig,
    ) -> None:
        super().__init__(config)
        self._adapter_config = config
        self._model_config = self._adapter_config.model
        self._tokenizer = self._load_tokenizer()
        self._chat_template = self._resolve_chat_template()
        self._template_content_format = self._detect_template_content_format()
        (
            self.force_reasoning,
            self.reasoning_config,
            self._reasoning_parser,
            self._tool_call_parser,
        ) = self._resolve_sglang_parsers()
        self._reasoning_detector_cache: dict[str, Any | None] = {}

    def process(self, item: OpenAIChatAdapterInput) -> OpenAIChatAdapterOutput:
        if isinstance(item, OpenAIChatConvertRequest):
            return self.convert_request(item)
        if isinstance(item, OpenAIChatBuildResponseRequest):
            return self.build_response(item)
        raise TypeError(f"Unsupported OpenAIChatAdapter input: {type(item)!r}")

    def convert_request(self, item: OpenAIChatConvertRequest) -> OpenAIChatConvertedRequest:
        request = self._chat_request_from_json(item.request_json)
        self._validate_request(request)
        input_ids = self._render_input_ids(request)
        tools = [tool.model_dump(mode="json", exclude_none=True) for tool in request.tools] if request.tools else None
        generation_input = GenerationInput(
            session_id=item.session_id,
            input_ids=input_ids,
            lora_path=request.lora_path if isinstance(request.lora_path, str) else None,
            tools=tools,
            tool_call_parser=self._tool_call_parser if tools else None,
        )
        context = OpenAIChatResponseContext(
            request_json=item.request_json,
            prompt_tokens=len(input_ids),
            response_id=f"chatcmpl-{uuid.uuid4().hex}",
            resolved_tool_call_parser=generation_input.tool_call_parser,
            resolved_reasoning_parser=self._reasoning_parser,
            force_reasoning=self.force_reasoning,
        )
        return OpenAIChatConvertedRequest(
            session_id=item.session_id,
            generation_input=generation_input,
            messages=self._axrl_messages_from_request(request),
            context=context,
        )

    def build_response(self, item: OpenAIChatBuildResponseRequest) -> OpenAIChatResponseResult:
        request = self._chat_request_from_json(item.context.request_json)
        response = self._build_chat_completion_response(
            request=request,
            context=item.context,
            output=item.generation_output,
        )
        return OpenAIChatResponseResult(response_json=response)

    def _load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self._model_path(),
            trust_remote_code=self._model_config.trust_remote_code,
            use_fast=True,
        )

    def _model_path(self) -> str:
        return str(self._model_config.get_full_path())

    def _resolve_chat_template(self) -> str | None:
        get_chat_template = getattr(self._tokenizer, "get_chat_template", None)
        if callable(get_chat_template):
            try:
                template = get_chat_template()
                if isinstance(template, str) and template:
                    return template
            except Exception as exc:
                logger.debug("Failed to read tokenizer chat template with get_chat_template: %s", exc)
        template = getattr(self._tokenizer, "chat_template", None)
        return template if isinstance(template, str) and template else None

    def _detect_template_content_format(self) -> str:
        from sglang.srt.parser.jinja_template_utils import detect_jinja_template_content_format

        if self._chat_template:
            return detect_jinja_template_content_format(self._chat_template)
        return "string"

    def _resolve_sglang_parsers(self) -> tuple[bool, ReasoningToggleConfig | None, str | None, str | None]:
        from sglang.srt.managers.template_detection import (
            detect_reasoning_parser,
            detect_reasoning_pattern,
            detect_tool_call_parser,
        )

        force_reasoning, reasoning_config = detect_reasoning_pattern(self._chat_template)
        reasoning_parser = self._resolve_parser_config(
            configured=self._adapter_config.reasoning_parser,
            label="reasoning parser",
            auto_detect=lambda: detect_reasoning_parser(
                self._chat_template,
                self._tokenizer,
                reasoning_config,
                force_reasoning,
            ),
            is_supported=self._is_supported_reasoning_parser,
        )
        tool_call_parser = self._resolve_parser_config(
            configured=self._adapter_config.tool_call_parser,
            label="tool-call parser",
            auto_detect=lambda: detect_tool_call_parser(
                self._chat_template,
                self._tokenizer,
                reasoning_config,
                force_reasoning,
            ),
            is_supported=self._is_supported_tool_call_parser,
        )
        return force_reasoning, reasoning_config, reasoning_parser, tool_call_parser

    @staticmethod
    def _resolve_parser_config(
        *,
        configured: str | None,
        label: str,
        auto_detect: Callable[[], str | None],
        is_supported: Callable[[str], bool],
    ) -> str | None:
        if configured is None:
            return None

        parser_name = configured.strip().lower()
        if not parser_name:
            return None
        if parser_name == "auto":
            detected = auto_detect()
            if detected is None:
                logger.warning("SGLang auto detection could not resolve %s; leaving it disabled.", label)
                return None
            return str(detected)

        if not is_supported(parser_name):
            raise ValueError(f"Unsupported SGLang {label}: {configured}")
        return parser_name

    @staticmethod
    def _is_supported_tool_call_parser(parser_name: str) -> bool:
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        return parser_name in FunctionCallParser.ToolCallParserEnum

    @staticmethod
    def _is_supported_reasoning_parser(parser_name: str) -> bool:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        return parser_name in ReasoningParser.DetectorMap

    def _chat_request_from_json(self, data: dict[str, Any]) -> ChatCompletionRequest:
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        return ChatCompletionRequest.model_validate(data)

    def _validate_request(self, request: ChatCompletionRequest) -> None:
        unsupported = self._unsupported_request_fields(request)
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(f"OpenAI chat request fields are not supported by GenerationInput yet: {joined}")

    @staticmethod
    def _axrl_messages_from_request(request: ChatCompletionRequest) -> list[Message]:
        return [OpenAIChatAdapter._axrl_message_from_sglang(message) for message in request.messages]

    @staticmethod
    def _axrl_message_from_sglang(message: BaseModel) -> Message:
        data = message.model_dump(mode="json", exclude_none=True)
        data.setdefault("content", "")
        return Message.from_dict(data)

    @staticmethod
    def _unsupported_request_fields(request: ChatCompletionRequest) -> list[str]:
        checks = [
            (not request.messages, "empty messages"),
            (
                isinstance(request.tool_choice, str) and request.tool_choice.lower() == "required" and not request.tools,
                "tool_choice=required without tools",
            ),
            (
                request.tool_choice is not None and not isinstance(request.tool_choice, str) and not request.tools,
                "specific tool_choice without tools",
            ),
            (request.stream, "stream=true"),
            (request.n != 1, "n != 1"),
            (request.logprobs or request.top_logprobs, "logprobs/top_logprobs"),
            (request.response_format is not None, "response_format"),
            (request.logit_bias is not None, "logit_bias"),
            (request.seed is not None, "seed"),
            (request.stop_token_ids is not None, "stop_token_ids"),
            (request.stop_regex is not None, "stop_regex"),
            (request.regex is not None, "regex"),
            (request.ebnf is not None, "ebnf"),
            (request.min_tokens, "min_tokens"),
            (request.ignore_eos, "ignore_eos"),
            (request.no_stop_trim, "no_stop_trim"),
            (request.custom_logit_processor is not None, "custom_logit_processor"),
            (request.custom_params is not None, "custom_params"),
            (request.return_hidden_states, "return_hidden_states"),
            (request.return_routed_experts, "return_routed_experts"),
            (isinstance(request.lora_path, list), "batched lora_path"),
        ]
        return [name for is_unsupported, name in checks if is_unsupported]

    def _render_input_ids(self, request: ChatCompletionRequest) -> NDArray[np.int32]:
        messages = self._messages_for_template(request)
        tools = self._tools_for_template(request)
        template_kwargs = self._chat_template_kwargs(request)
        input_ids_tensor = cast(
            "torch.Tensor",
            self._tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
                return_dict=False,
                tools=tools,
                **template_kwargs,
            ),
        )
        return array_utils.as_i32(input_ids_tensor.squeeze(0).tolist())

    def _messages_for_template(self, request: ChatCompletionRequest) -> list[dict[str, Any]]:
        from sglang.srt.entrypoints.openai.serving_chat import normalize_tool_content
        from sglang.srt.parser.jinja_template_utils import process_content_for_template_format

        messages: list[dict[str, Any]] = []
        image_data: list[Any] = []
        video_data: list[Any] = []
        audio_data: list[Any] = []
        modalities: list[Any] = []

        for message in request.messages:
            dumped = message.model_dump(mode="json", exclude_none=True)
            dumped.setdefault("content", "")

            # Mirrors SGLang OpenAI serving_chat._apply_jinja_template:
            # https://github.com/sgl-project/sglang/blob/127b9e3283f7c2a43234b852ff5c9f1796d53624/python/sglang/srt/entrypoints/openai/serving_chat.py#L608-L645
            processed = process_content_for_template_format(
                dumped,
                self._template_content_format,
                image_data,
                video_data,
                audio_data,
                modalities,
            )
            processed["content"] = normalize_tool_content(processed["role"], processed.get("content"))
            self._normalize_assistant_tool_calls(processed)
            messages.append(processed)

        if image_data or video_data or audio_data or modalities:
            raise ValueError("multimodal OpenAI chat content is not supported by GenerationInput")
        return messages

    @staticmethod
    def _normalize_assistant_tool_calls(message: dict[str, Any]) -> None:
        if message.get("role") != "assistant" or not isinstance(message.get("tool_calls"), list):
            return
        for tool_call in message["tool_calls"]:
            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                function["arguments"] = orjson.loads(function["arguments"])

    @staticmethod
    def _tools_for_template(request: ChatCompletionRequest) -> list[dict[str, Any]] | None:
        if not request.tools or request.tool_choice == "none":
            return None
        return [tool.model_dump(mode="json", exclude_none=True) for tool in request.tools]

    @staticmethod
    def _chat_template_kwargs(request: ChatCompletionRequest) -> dict[str, Any]:
        kwargs = dict(request.chat_template_kwargs or {})
        return kwargs

    def _build_chat_completion_response(
        self,
        *,
        request: ChatCompletionRequest,
        context: OpenAIChatResponseContext,
        output: GenerationOutput,
    ) -> dict[str, Any]:
        from sglang.srt.entrypoints.openai.protocol import (
            ChatCompletionResponse,
            ChatMessage,
        )
        from sglang.srt.entrypoints.openai.protocol import (
            ChatCompletionResponseChoice as SGLangChoice,
        )
        from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor

        text, reasoning_text = self._split_reasoning_text(request, context, output.output_text)
        text, tool_calls = self._resolve_tool_calls(request, context, text, output)

        finish_reason = self._finish_reason(output.finish_reason)
        matched_stop = output.stop_reason
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
            matched_stop = None
        if finish_reason == "tool_calls":
            matched_stop = None

        response = ChatCompletionResponse(
            id=context.response_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                SGLangChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=text or None,
                        tool_calls=tool_calls,
                        reasoning_content=reasoning_text or None,
                    ),
                    finish_reason=finish_reason,
                    matched_stop=matched_stop,
                )
            ],
            usage=UsageProcessor.calculate_token_usage(
                prompt_tokens=context.prompt_tokens,
                completion_tokens=len(output.output_ids),
                reasoning_tokens=0,
            ),
            metadata={"weight_version": "default"},
        )
        return response.model_dump(mode="json")

    def _split_reasoning_text(
        self,
        request: ChatCompletionRequest,
        context: OpenAIChatResponseContext,
        text: str,
    ) -> tuple[str, str | None]:
        if not context.resolved_reasoning_parser or not request.separate_reasoning:
            return text, None

        from sglang.srt.parser.reasoning_parser import ReasoningParser

        parser = ReasoningParser(
            model_type=context.resolved_reasoning_parser,
            stream_reasoning=False,
            force_reasoning=self._force_reasoning_for_request(request, context),
            request=request,
        )
        reasoning_text, normal_text = parser.parse_non_stream(text)
        return normal_text or "", reasoning_text or None

    def _force_reasoning_for_request(self, request: ChatCompletionRequest, context: OpenAIChatResponseContext) -> bool:
        if context.force_reasoning:
            return True
        if not context.resolved_reasoning_parser:
            return False

        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        serving_chat = cast("Any", OpenAIServingChat.__new__(OpenAIServingChat))
        serving_chat.reasoning_parser = context.resolved_reasoning_parser
        serving_chat.template_manager = self
        serving_chat._reasoning_detector = self._sglang_reasoning_detector(context.resolved_reasoning_parser)
        return bool(OpenAIServingChat._get_reasoning_from_request(serving_chat, request))

    def _sglang_reasoning_detector(self, parser_name: str) -> Any | None:
        if parser_name not in self._reasoning_detector_cache:
            from sglang.srt.parser.reasoning_parser import ReasoningParser

            try:
                self._reasoning_detector_cache[parser_name] = ReasoningParser(
                    model_type=parser_name,
                    stream_reasoning=True,
                ).detector
            except ValueError as exc:
                logger.warning("Failed to initialize SGLang reasoning detector for parser '%s': %s", parser_name, exc)
                self._reasoning_detector_cache[parser_name] = None
        return self._reasoning_detector_cache[parser_name]

    def _resolve_tool_calls(
        self,
        request: ChatCompletionRequest,
        context: OpenAIChatResponseContext,
        text: str,
        output: GenerationOutput,
    ) -> tuple[str, list[Any] | None]:
        if output.tool_calls:
            return text, self._sglang_tool_calls_from_generation(output.tool_calls)
        if not request.tools or request.tool_choice == "none" or not context.resolved_tool_call_parser:
            return text, None

        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        parser = FunctionCallParser(request.tools, context.resolved_tool_call_parser)
        if not parser.has_tool_call(text):
            return text, None
        normal_text, parsed_calls = parser.parse_non_stream(text)
        if not parsed_calls:
            return text, None
        return normal_text, self._sglang_tool_calls_from_parsed(parsed_calls, context.resolved_tool_call_parser)

    @staticmethod
    def _sglang_tool_calls_from_generation(tool_calls: list[Any]) -> list[Any]:
        from sglang.srt.entrypoints.openai.protocol import FunctionResponse, ToolCall

        return [
            ToolCall(
                id=tool_call.id,
                index=tool_call.index,
                function=FunctionResponse(name=tool_call.name, arguments=tool_call.arguments),
            )
            for tool_call in tool_calls
        ]

    @staticmethod
    def _sglang_tool_calls_from_parsed(parsed_calls: list[Any], tool_call_parser: str) -> list[Any]:
        from sglang.srt.entrypoints.openai.protocol import FunctionResponse, ToolCall

        tool_calls = []
        for call in parsed_calls:
            tool_index = getattr(call, "tool_index", None)
            name = getattr(call, "name", None)
            if tool_call_parser == "kimi_k2" and name is not None and tool_index is not None:
                tool_id = f"functions.{name}:{tool_index}"
            else:
                tool_id = f"call_{uuid.uuid4().hex[:24]}"
            tool_calls.append(
                ToolCall(
                    id=tool_id,
                    index=tool_index,
                    function=FunctionResponse(name=name, arguments=getattr(call, "parameters", None)),
                )
            )
        return tool_calls

    @staticmethod
    def _finish_reason(reason: str) -> ChatFinishReason:
        if reason in {"stop", "length", "tool_calls", "content_filter", "function_call", "abort"}:
            return cast("ChatFinishReason", reason)
        return "stop"
