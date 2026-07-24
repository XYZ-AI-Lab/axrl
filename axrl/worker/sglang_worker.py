import asyncio
import dataclasses
import json
import logging
import os
import time
import uuid
from typing import Any, Literal, override

import numpy as np
import torch
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.entrypoints.openai.protocol import Tool as OpenAITool
from sglang.srt.entrypoints.openai.protocol import ToolChoice as OpenAIToolChoice
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.utils import get_json_schema_constraint
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    InitWeightsUpdateGroupReqInput,
    MultimodalDataInputFormat,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
from sglang.srt.state_capturer.routed_experts import extract_routed_experts_from_meta_info
from sglang.utils import convert_json_schema_to_str
from transformers import AutoConfig, AutoProcessor

from axrl.configs import RolloutWorkerConfig, SamplingConfig
from axrl.data import GenerationInput, GenerationOutput, array_utils
from axrl.data.generation import TensorHandle, ToolCall
from axrl.processor.chat_template_utils import get_single_token_assistant_boundary_id
from axrl.utils import gpu_utils
from axrl.utils.moe_utils import get_routing_info_shape
from axrl.utils.sglang.sglang_scheduler_patch import install_scheduler_stream_output_patch
from axrl.utils.sglang.sglang_signal_patch import temporary_signal_patch
from axrl.utils.timer import SessionTimer, Timer
from axrl.worker.rollout_worker import RolloutWorker

logger = logging.getLogger(__name__)


install_scheduler_stream_output_patch()


class SGLangWorker(RolloutWorker):
    """SGLang-based model worker with partial rollout generation."""

    def __init__(self, config: RolloutWorkerConfig) -> None:
        super().__init__(config)
        self.config = config
        assert self.config.engine_type == "sglang"
        self.name = self.config.name
        self.working_rids: set[str] = set()  # Active request IDs
        self.generation_ready_event = asyncio.Event()
        self.generation_ready_event.set()
        self.finished_any_generation = False
        self._assistant_boundary_token_id: int | None = None

    @property
    def _engine_tokenizer_manager(self) -> TokenizerManager:
        tokenizer_manager = self.engine.tokenizer_manager
        assert tokenizer_manager is not None
        return tokenizer_manager

    def initialize(self) -> None:
        self._set_env()
        model_dir = self.config.model.get_full_path()
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory {model_dir} does not exist")

        if self.config.attention_backend is None and torch.cuda.get_device_capability()[0] < 9:
            # Use Triton backend for A100 and below
            self.config.attention_backend = "triton"

        # Initialize engine
        engine_args = ServerArgs(
            model_path=str(model_dir),
            mem_fraction_static=self.config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            tp_size=self.config.tp_size,
            pp_size=self.config.pp_size,
            dp_size=self.config.dp_size,
            ep_size=self.config.ep_size,
            moe_a2a_backend=self.config.moe_a2a_backend,
            decode_log_interval=8192,  # log interval for decoding, in number of tokens
            enable_metrics=self.config.enable_metrics,
            log_level=self.config.log_level,
            load_format="dummy" if self.config.load_dummy_weights else "auto",
            dist_init_addr=f"{self.config.master_addr}:{self.config.master_port}" if self.config.master_addr else None,
            nnodes=self.config.nnodes,
            node_rank=self.config.node_rank,
            enable_memory_saver=True,
            max_running_requests=self.config.max_running_requests,
            dtype=self.config.dtype,
            kv_cache_dtype=self.config.kv_cache_dtype,
            prefill_max_requests=self.config.prefill_max_requests,
            enable_fp32_lm_head=self.config.enable_fp32_lm_head,
            attention_backend=self.config.attention_backend,
            enable_return_routed_experts=self.config.enable_routing_replay,
            enforce_disable_flashinfer_allreduce_fusion=self._should_disable_flashinfer_allreduce_fusion(model_dir),
        )
        with temporary_signal_patch():
            self.engine = Engine(**dataclasses.asdict(engine_args))

        self.on_gpu: dict[str, bool] = {"weights": True, "kv_cache": True}

        self._init_routing_replay(model_dir)

        # Initialize processor
        self.processor: Any = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=model_dir,
            use_fast=True,
        )
        self._assistant_boundary_token_id = get_single_token_assistant_boundary_id(self.processor)

        logger.info(f"Worker {self.name} initialized")

    def _set_env(self) -> None:
        os.environ["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "1"

    def _should_disable_flashinfer_allreduce_fusion(self, model_dir: Any) -> bool:
        if not self._requires_flashinfer_allreduce_fusion_disable(model_dir):
            return False

        logger.warning(
            "Disabling SGLang FlashInfer all-reduce fusion for %s with tp=%s ep=%s. "
            "This avoids incorrect Qwen3-MoE outputs when expert parallelism is combined "
            "with MoE TP.",
            self.config.model.name,
            self.config.tp_size,
            self.config.ep_size,
        )
        return True

    def _requires_flashinfer_allreduce_fusion_disable(self, model_dir: Any) -> bool:
        """Return whether SGLang's Qwen3-MoE all-reduce fusion guard is needed."""
        if self.config.ep_size <= 1 or self.config.tp_size <= self.config.ep_size:
            return False
        if self.config.moe_a2a_backend != "none":
            return False

        hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        architectures = getattr(hf_config, "architectures", None) or []
        # SGLang currently auto-enables FlashInfer all-reduce fusion for Qwen3-MoE
        # on this path, but it produces incorrect outputs when rollout TP is
        # larger than rollout EP. Once SGLang fixes that fusion path, this guard
        # can be removed so the optimized fused all-reduce path is enabled again.
        return "Qwen3MoeForCausalLM" in architectures

    def _init_routing_replay(self, model_dir: Any) -> None:
        """Load MoE shape info from HF config when routing replay is enabled."""
        self._num_hidden_layers: int | None = None
        self._num_experts_per_tok: int | None = None
        if not self.config.enable_routing_replay:
            return

        hf_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self._num_hidden_layers, self._num_experts_per_tok = get_routing_info_shape(hf_config)
        logger.info(
            "Routing replay enabled: num_hidden_layers=%s, num_experts_per_tok=%s, routed_experts_start_len=True",
            self._num_hidden_layers,
            self._num_experts_per_tok,
        )

    @staticmethod
    def _resolve_tool_choice(tool_choice: str | dict[str, Any] | None, *, has_tools: bool) -> Literal["auto", "required", "none"] | OpenAIToolChoice:
        """Resolve tool_choice into a canonical form.

        Possible values (following the OpenAI API convention):
          - "auto"     — model decides whether to call a tool
          - "required" — model must call one or more tools
          - "none"     — model must not call any tool
          - {"type": "function", "function": {"name": "get_weather"}}
              — model must call the specific named function (OpenAIToolChoice)
        """
        if tool_choice is None:
            return "auto" if has_tools else "none"
        if isinstance(tool_choice, str):
            assert tool_choice in ("auto", "required", "none"), f"Unsupported tool_choice string: {tool_choice}"
            return tool_choice  # type: ignore[return-value]
        # Named function choice, e.g. {"type": "function", "function": {"name": "get_weather"}}
        return OpenAIToolChoice.model_validate(tool_choice)

    @staticmethod
    def _build_tool_models(tools: list[dict[str, Any]]) -> list[OpenAITool]:
        return [OpenAITool.model_validate(tool) for tool in tools]

    def _apply_tool_call_constraint(self, req: GenerationInput, sampling_params: dict[str, Any]) -> dict[str, Any]:
        if not req.tools:
            return sampling_params
        resolved = self._resolve_tool_choice(req.tool_choice, has_tools=True)
        if resolved == "none":
            return sampling_params

        tool_models = self._build_tool_models(req.tools)

        # "required" / named-function: uses json_schema constraint to force the model
        # to output **raw JSON** (a list of {name, parameters} objects). No parser
        # is needed — the output is parsed with json.loads() in parse_function_calls.
        if resolved == "required" or isinstance(resolved, OpenAIToolChoice):
            json_schema = get_json_schema_constraint(tool_models, resolved)
            assert json_schema is not None
            sampling_params = sampling_params.copy()
            sampling_params["json_schema"] = convert_json_schema_to_str(json_schema)
            return sampling_params

        # "auto": the model decides whether to call a tool. Output uses the model's
        # native format (e.g. <tool_call>...\n</tool_call> for Qwen), so a parser
        # is required to detect and extract tool calls from the generated text.
        if req.tool_call_parser is None:
            return sampling_params
        parser = FunctionCallParser(tool_models, req.tool_call_parser)
        # get_structure_constraint returns a (type, value) tuple describing how to
        # constrain the model's output during generation. For Qwen with tool_choice="auto":
        #   constraint_type = "structural_tag"
        #   constraint_value = LegacyStructuralTagResponseFormat(
        #       type="structural_tag",
        #       structures=[StructuresResponseFormat(
        #           begin='<tool_call>\n{"name":"get_weather", "arguments":',
        #           schema={"type": "object", "properties": {...}},
        #           end='}\n</tool_call>',
        #       ), ...],
        #       triggers=["<tool_call>"],
        #   )
        # This tells the engine: when the model generates the trigger token "<tool_call>",
        # constrain the function arguments to match the JSON schema until the end tag.
        constraint = parser.get_structure_constraint(resolved)
        if constraint is None:
            return sampling_params
        constraint_type, constraint_value = constraint
        sampling_params = sampling_params.copy()
        if constraint_type == "structural_tag":
            # Serialize the structural tag to a JSON string for the sampling params
            sampling_params[constraint_type] = convert_json_schema_to_str(constraint_value.model_dump(by_alias=True))
        elif constraint_type == "json_schema":
            sampling_params[constraint_type] = convert_json_schema_to_str(dict(constraint_value))
        else:
            # Other constraint types (e.g. "regex", "ebnf") are passed as-is
            sampling_params[constraint_type] = constraint_value
        return sampling_params

    @staticmethod
    def parse_function_calls(
        text: str,
        *,
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] | None = None,
        tool_call_parser: str | None = None,
        finish_reason: str = "stop",
    ) -> tuple[str, list[ToolCall] | None, str]:
        """Parse tool calls from generated text."""
        if not tools:
            return text, None, finish_reason

        resolved = SGLangWorker._resolve_tool_choice(tool_choice, has_tools=True)
        if resolved == "none":
            return text, None, finish_reason

        # "required" / named: json_schema constraint was applied, so the model
        # output is raw JSON — parse it directly without a tool_call_parser.
        if resolved == "required" or isinstance(resolved, OpenAIToolChoice):
            try:
                tool_call_data = json.loads(text)
            except json.JSONDecodeError:
                return text, None, finish_reason
            if isinstance(tool_call_data, dict):
                tool_call_data = [tool_call_data]
            tool_calls = [
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    index=i,
                    name=tc["name"],
                    arguments=json.dumps(tc["parameters"], ensure_ascii=False),
                )
                for i, tc in enumerate(tool_call_data)
            ]
            return "", tool_calls, "tool_calls" if finish_reason == "stop" else finish_reason

        # "auto": the model may or may not have called a tool, using its native
        # format (e.g. <tool_call>...</tool_call>). Need a parser to detect/extract.
        if tool_call_parser is None:
            return text, None, finish_reason
        tool_models = SGLangWorker._build_tool_models(tools)
        parser = FunctionCallParser(tool_models, tool_call_parser)
        if not parser.has_tool_call(text):
            return text, None, finish_reason
        try:
            normal_text, parsed_calls = parser.parse_non_stream(text)
        except Exception:
            logger.warning(f"Failed to parse tool calls from text: {text!r}", exc_info=True)
            return text, None, finish_reason
        if not parsed_calls:
            return text, None, finish_reason
        tool_calls = []
        for call in parsed_calls:
            assert call.name is not None, "Non-streaming parsing should always produce a function name"
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    index=call.tool_index,
                    name=call.name,
                    arguments=call.parameters,
                )
            )
        return normal_text, tool_calls, "tool_calls" if finish_reason == "stop" else finish_reason

    @override
    async def generate(self, req: GenerationInput) -> GenerationOutput:  # noqa: PLR0915
        """Generate with partial rollout, token-in-token-out."""
        session_id = req.session_id
        assert session_id, "GenerationInput.session_id must be set before SGLang generation."
        req.event_timing.mark_worker_received()
        total_timer = SessionTimer(session_id, "async", "SGLang: worker generate")
        total_timer.start()
        start_time = time.perf_counter()
        sampling_config = req.sampling_config or self.config.sampling_config
        self._validate_generation_request(req, sampling_config)
        request_input_ids = array_utils.to_int_list(req.input_ids)
        output_ids: list[int] = []
        output_logprobs: list[float] = []
        input_logprob_token_ids: list[int] | None = None
        input_logprobs: list[float] | None = None
        routed_experts: np.ndarray | None = None
        sampling_params = self._get_sampling_params(sampling_config)
        sampling_params = self._apply_tool_call_constraint(req, sampling_params)
        max_total_tokens = self._get_max_total_tokens(sampling_config)

        # Initialize generation state
        finish_reason: str | None = None
        matched_token: int | None = None
        cached_tokens: int | None = None
        retry: int = 0

        # Continue generation after model update aborts
        while True:
            if not self.generation_ready_event.is_set():
                logger.debug(f"Worker {self.name} updating, waiting to resume generation")
                await self.generation_ready_event.wait()

            new_max_tokens = self._compute_remaining_max_new_tokens(
                sampling_config,
                req,
                output_ids,
                max_total_tokens,
                allow_zero_new_tokens=req.capture_input_logprobs,
            )
            if new_max_tokens is None:
                finish_reason = "length"
                break
            sampling_params["max_new_tokens"] = new_max_tokens
            input_ids = [*array_utils.to_int_list(req.input_ids), *output_ids]
            logprob_start_len: int | None = None
            if req.capture_input_logprobs and input_logprobs is None:
                # SGLang returns a leading anchor row with a None logprob for
                # the first token in the requested scoring window. Request one
                # token earlier so the first user-requested token still has a
                # real logprob in the returned rows.
                logprob_start_len = req.input_logprob_start_index - 1
            with SessionTimer(session_id, "async", "SGLang: generate tokens"):
                outputs = await self._generate_tokens(
                    input_ids=input_ids,
                    sampling_params=sampling_params,
                    lora_path=req.lora_path,
                    return_logprob=True,
                    logprob_start_len=logprob_start_len,
                    routed_expert_start_index=req.routed_expert_start_index,
                    capture_routing=req.capture_routing,
                    session_id=session_id,
                )

            output_ids = output_ids + outputs["output_ids"]
            output_logprobs = output_logprobs + outputs["output_logprobs"]
            if req.capture_input_logprobs and input_logprobs is None:
                input_logprob_token_ids, input_logprobs = self._maybe_select_input_logprobs(
                    outputs,
                    request_input_ids=request_input_ids,
                    input_logprob_start_index=req.input_logprob_start_index,
                    session_id=session_id,
                )
            if self.config.enable_routing_replay:
                routed_experts = outputs.get("routed_experts", routed_experts)

            cached_tokens = outputs["cached_tokens"] if cached_tokens is None else cached_tokens
            finish_reason = outputs["finish_reason"]
            matched_token = outputs.get("matched_token")

            if finish_reason != "abort":
                # Finish reasons: "length", "stop", etc.
                break

            if not self.config.continue_generation_after_abort:
                break

            if self.config.clear_partial_outputs_after_abort:
                output_ids = []
                output_logprobs = []
                routed_experts = None
                cached_tokens = None

            logger.debug("Aborted, to re-submit the request.")
            retry += 1

        # Validate output
        assert finish_reason is not None
        assert cached_tokens is not None
        assert not req.capture_input_logprobs or input_logprobs is not None, "capture_input_logprobs=True but SGLang did not return input logprobs."
        self.finished_any_generation = True
        with SessionTimer(session_id, "sync", "SGLang: decode output"):
            output_text = self._decode(output_ids)
        tool_calls: list[ToolCall] | None = None
        if req.tools:
            with SessionTimer(session_id, "sync", "SGLang: parse function calls"):
                output_text, tool_calls, finish_reason = self.parse_function_calls(
                    output_text,
                    tools=req.tools,
                    tool_choice=req.tool_choice,
                    tool_call_parser=req.tool_call_parser,
                    finish_reason=finish_reason,
                )
        prior_rows = req.routed_expert_start_index
        if routed_experts is not None:
            with SessionTimer(session_id, "sync", "SGLang: validate routed experts"):
                total_seq_len = len(req.input_ids) + len(output_ids)
                expected_len = max(total_seq_len - 1 - prior_rows, 0)
                assert routed_experts.shape[0] == expected_len, (
                    f"Expected routed_experts length {expected_len} (total_seq_len={total_seq_len}, "
                    f"prior_rows={prior_rows}), got {routed_experts.shape[0]}"
                )

        routing_handle: TensorHandle | None = None
        if routed_experts is not None:
            assert req.capture_routing, "R3 enabled but req.capture_routing is False"
            from axrl.utils import tensor_store as store

            with SessionTimer(session_id, "sync", "SGLang: store routed experts"):
                routing_handle = store.put(routed_experts)

        assistant_boundary_token_id = self._assistant_boundary_token_to_append(
            output_ids,
            self._assistant_boundary_token_id,
        )

        with SessionTimer(session_id, "sync", "SGLang: build output"):
            output = GenerationOutput(
                session_id=session_id,
                output_ids=array_utils.as_i32(output_ids),
                output_logprobs=array_utils.as_f32(output_logprobs),
                output_text=output_text,
                output_text_with_special_tokens="",
                cached_tokens=cached_tokens,
                finish_reason=finish_reason,
                e2e_elapsed_seconds=time.perf_counter() - start_time,
                stop_reason=matched_token,
                retry=retry,
                assistant_boundary_token_id=assistant_boundary_token_id,
                tool_calls=tool_calls,
                input_logprobs=array_utils.as_f32(input_logprobs) if input_logprobs is not None else None,
                input_logprob_token_ids=array_utils.as_i32(input_logprob_token_ids) if input_logprob_token_ids is not None else None,
                input_logprob_start_index=req.input_logprob_start_index if req.capture_input_logprobs else None,
                routing_handle=routing_handle,
                event_timing=req.event_timing,
            )
        total_timer.stop()
        output.event_timing.mark_worker_returned()
        return output

    @staticmethod
    def _validate_generation_request(req: GenerationInput, sampling_config: SamplingConfig) -> None:
        assert req.input_logprob_start_index <= len(req.input_ids), (
            f"input_logprob_start_index={req.input_logprob_start_index} exceeds input length {len(req.input_ids)}."
        )
        if req.capture_input_logprobs:
            assert req.input_logprob_start_index > 0, "input_logprob_start_index must be greater than zero when capture_input_logprobs=True."
        if sampling_config.max_new_tokens == 0:
            assert req.capture_input_logprobs, "max_new_tokens=0 is only supported when capture_input_logprobs=True."

    @staticmethod
    def _assistant_boundary_token_to_append(output_ids: list[int], boundary_token_id: int | None) -> int | None:
        if boundary_token_id is None:
            return None
        if len(output_ids) > 0 and int(output_ids[-1]) == boundary_token_id:
            return None
        return boundary_token_id

    async def _generate_tokens(
        self,
        input_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: MultimodalDataInputFormat | None = None,
        video_data: MultimodalDataInputFormat | None = None,
        audio_data: MultimodalDataInputFormat | None = None,
        lora_path: str | None = None,
        *,
        return_logprob: bool = False,
        logprob_start_len: int | None = None,
        routed_expert_start_index: int = 0,
        capture_routing: bool = False,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate tokens, token-in-token-out."""
        timer_session_id = session_id or "unknown"
        if sampling_params["max_new_tokens"] <= 0 and logprob_start_len is None:
            return {
                "output_ids": [],
                "output_logprobs": [],
                "finish_reason": "length",
                "cached_tokens": 0,
                "routed_experts": None,
            }
        return_routed_experts = self.config.enable_routing_replay and capture_routing
        rid = uuid.uuid4().hex
        generation_request = GenerateReqInput(
            rid=rid,
            input_ids=input_ids,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            lora_path=lora_path,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            sampling_params=sampling_params,
            return_routed_experts=return_routed_experts,
            routed_experts_start_len=routed_expert_start_index if return_routed_experts else 0,
        )

        # Execute generation
        self.working_rids.add(rid)
        try:
            with SessionTimer(timer_session_id, "async", "SGLang: wait engine result"):
                generator = self._engine_tokenizer_manager.generate_request(generation_request, None)  # type: ignore
                results = [result async for result in generator]  # type: ignore
        finally:
            self.working_rids.remove(rid)

        # Parse results
        with SessionTimer(timer_session_id, "sync", "SGLang: parse engine result"):
            output: Any = results[0]
            finish_reason_info = output["meta_info"]["finish_reason"]

        with SessionTimer(timer_session_id, "sync", "SGLang: extract token/logprobs"):
            if "output_ids" not in output:
                assert finish_reason_info["type"] == "abort"
            output_ids, output_logprobs = self.get_token_ids_and_logprobs(output, return_logprob=return_logprob)
            input_logprob_token_ids, input_logprobs = self.get_input_token_ids_and_logprobs(
                output,
                logprob_start_len=logprob_start_len,
            )

        # Extract routed experts if enabled
        routed_experts: np.ndarray | None = None
        if return_routed_experts:
            with SessionTimer(timer_session_id, "sync", "SGLang: extract routed experts"):
                assert self._num_hidden_layers is not None and self._num_experts_per_tok is not None
                raw = extract_routed_experts_from_meta_info(output)
                routed_experts = raw.reshape(-1, self._num_hidden_layers, self._num_experts_per_tok).astype(np.int16)
                # SGLang returns seqlen-1 routing entries: the first token in the sequence
                # has no routing since it was never predicted by the model.
                assert len(input_ids) > 0
                full_row_count = len(input_ids) + len(output_ids) - 1
                expected_row_count = max(full_row_count - routed_expert_start_index, 0)
                assert routed_experts.shape[0] == expected_row_count, (
                    f"Expected routed_experts length {expected_row_count} after SGLang-side start "
                    f"(full_row_count={full_row_count}, routed_expert_start_index={routed_expert_start_index}), "
                    f"got {routed_experts.shape[0]}"
                )

        return {
            "output_ids": output_ids,
            "output_logprobs": output_logprobs,
            "finish_reason": finish_reason_info["type"],
            "matched_token": finish_reason_info.get("matched"),
            "cached_tokens": output["meta_info"].get("cached_tokens", 0),
            "input_logprob_token_ids": input_logprob_token_ids,
            "input_logprobs": input_logprobs,
            "routed_experts": routed_experts,
        }

    @staticmethod
    def get_token_ids_and_logprobs(output: dict, *, return_logprob: bool) -> tuple[list[int], list[float]]:
        if not return_logprob:
            token_ids = output.get("output_ids", [])
            return token_ids, []
        output_token_logprobs = output.get("meta_info", {}).get("output_token_logprobs", [])
        if not output_token_logprobs:
            return [], []
        log_probs = [x[0] for x in output_token_logprobs]
        token_ids = [x[1] for x in output_token_logprobs]
        return token_ids, log_probs

    @staticmethod
    def get_input_token_ids_and_logprobs(output: dict, *, logprob_start_len: int | None) -> tuple[list[int] | None, list[float] | None]:
        if logprob_start_len is None:
            return None, None
        meta_info = output.get("meta_info", {})
        if "input_token_logprobs" not in meta_info:
            return None, None
        input_token_logprobs = meta_info["input_token_logprobs"]
        if not input_token_logprobs:
            return [], []

        token_ids: list[int] = []
        logprobs: list[float] = []
        for index, entry in enumerate(input_token_logprobs):
            assert len(entry) >= 2, f"Unexpected input_token_logprobs[{index}] entry: {entry!r}"
            raw_logprob, token_id = entry[0], entry[1]
            if raw_logprob is None:
                assert index == 0, f"Unexpected None input logprob at index {index} for logprob_start_len={logprob_start_len}."
                continue
            token_ids.append(int(token_id))
            logprobs.append(float(raw_logprob))
        return token_ids, logprobs

    @staticmethod
    def _maybe_select_input_logprobs(
        outputs: dict[str, Any],
        *,
        request_input_ids: list[int],
        input_logprob_start_index: int,
        session_id: str,
    ) -> tuple[list[int] | None, list[float] | None]:
        token_ids = outputs.get("input_logprob_token_ids")
        logprobs = outputs.get("input_logprobs")
        if token_ids is None or logprobs is None:
            assert outputs["finish_reason"] == "abort", "capture_input_logprobs=True but SGLang did not return input logprobs."
            logger.debug(
                "SGLang abort did not return complete input logprobs for session %s; retry will request them again.",
                session_id,
            )
            return None, None
        return SGLangWorker._select_requested_input_logprobs(
            request_input_ids=request_input_ids,
            token_ids=token_ids,
            logprobs=logprobs,
            logprob_start_index=input_logprob_start_index,
        )

    @staticmethod
    def _select_requested_input_logprobs(
        *,
        request_input_ids: list[int],
        token_ids: list[int],
        logprobs: list[float],
        logprob_start_index: int,
    ) -> tuple[list[int], list[float]]:
        assert len(token_ids) == len(logprobs), f"input logprob token ids length {len(token_ids)} does not match logprobs length {len(logprobs)}."
        # SGLang returns a leading anchor row with a None logprob. The caller
        # requests one token before logprob_start_index, and
        # get_input_token_ids_and_logprobs drops that anchor row, so returned
        # token ids should start at the requested scored token.
        expected = request_input_ids[logprob_start_index:]
        selected_token_ids = token_ids[: len(expected)]
        selected_logprobs = logprobs[: len(expected)]
        assert selected_token_ids == expected, f"input logprob token ids {selected_token_ids} do not match input_ids slice {expected}."
        return selected_token_ids, selected_logprobs

    def _get_sampling_params(self, sampling_config: SamplingConfig) -> dict[str, Any]:
        config: dict[str, Any] = sampling_config.__dict__.copy()
        max_total_tokens = config.pop("max_total_tokens")
        if config.get("max_new_tokens") is None:
            config["max_new_tokens"] = max_total_tokens
        return config

    def _get_max_total_tokens(self, sampling_config: SamplingConfig) -> int:
        max_total_tokens = sampling_config.max_total_tokens
        if max_total_tokens <= 0:
            max_total_tokens = self.config.model.seq_length
        return max_total_tokens - 1

    def _compute_remaining_max_new_tokens(
        self,
        sampling_config: SamplingConfig,
        req: GenerationInput,
        output_ids: list[int],
        max_total_tokens: int,
        *,
        allow_zero_new_tokens: bool = False,
    ) -> int | None:
        new_max = max_total_tokens - len(req.input_ids) - len(output_ids) - 1
        if sampling_config.max_new_tokens is not None:
            new_max = min(sampling_config.max_new_tokens, new_max)
        if new_max == 0 and allow_zero_new_tokens:
            return 0
        if new_max > 0:
            return new_max
        assert output_ids, (
            f"Out of token budget on session {req.session_id} with no prior output: "
            f"max_total_tokens={max_total_tokens}, input_ids={len(req.input_ids)}, "
            f"output_ids=0. Env must budget-check before requesting generation."
        )
        logger.warning(
            f"Out of budget after abort-retry on session {req.session_id}: "
            f"max_total_tokens={max_total_tokens}, input_ids={len(req.input_ids)}, "
            f"output_ids={len(output_ids)}. Returning partial output as finish_reason='length'."
        )
        return None

    def _decode(self, output_ids: list[int]) -> str:
        return self.processor.decode(token_ids=output_ids, skip_special_tokens=True)

    async def warmup_tensor_store(self) -> TensorHandle:
        """Put a 1-element tensor and return its handle for consumer-side warmup."""
        from axrl.utils import tensor_store as store

        return store.put(torch.zeros(1, dtype=torch.int16))

    @override
    async def pause_generation(self) -> None:
        if not self.generation_ready_event.is_set():
            logger.info(f"{self.name} pause_generation called while already paused.")
            return

        self.generation_ready_event.clear()
        start_time = time.perf_counter()
        num_working_rids = len(self.working_rids)

        while self.working_rids:
            for rid in self.working_rids:
                self._engine_tokenizer_manager.abort_request(rid)
            await asyncio.sleep(0.5)

        seconds = time.perf_counter() - start_time
        logger.info(f"{self.name} aborted {num_working_rids} generations in {seconds:.2f}s.")

    @override
    async def resume_generation(self) -> None:
        if self.generation_ready_event.is_set():
            logger.info(f"{self.name} resume_generation called while already running.")
            return

        self.generation_ready_event.set()
        logger.info(f"{self.name} resumed generation.")

    @override
    async def init_weights_update_group(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None:
        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )

        await self._engine_tokenizer_manager.init_weights_update_group(obj, None)

    def is_paused(self) -> bool:
        return not self.generation_ready_event.is_set()

    @override
    async def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[bytes],
        load_format: str | None = None,
    ) -> None:
        """Update weights from serialized tensors.

        Args:
            serialized_named_tensors: One serialized entry per TP rank. Each entry
                is deserialized by the corresponding TP worker via MultiprocessingSerializer.
            load_format: Controls how SGLang's model_runner processes the deserialized data.
                - None (default): expects list[tuple[str, Tensor]], calls model.load_weights()
                - "flattened_bucket": expects dict with "flattened_tensor" + "metadata",
                  reconstructs tensors as zero-copy views from the flattened buffer.
                  This is the fast path used by the colocated FlattenedTensorBucket updater.
        """
        assert self.is_paused(), "Generation must be paused before updating weights"
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=serialized_named_tensors,  # type: ignore
            load_format=load_format,
            flush_cache=False,
        )
        await self._engine_tokenizer_manager.update_weights_from_tensor(obj, None)

    @override
    async def update_weights_from_distributed(self, names: list[str], dtypes: list[str], shapes: list[list[int]], group_name: str) -> None:
        assert self.is_paused(), "Generation must be paused before updating weights"
        obj = UpdateWeightsFromDistributedReqInput(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            flush_cache=False,
        )
        await self._engine_tokenizer_manager.update_weights_from_distributed(obj, None)

    @override
    def shutdown(self) -> None:
        with Timer(name=f"worker({self.name}) shutdown", verbose=True):
            self.engine.shutdown()
            del self.engine
            del self.processor
            gpu_utils.clear_cache()

    @override
    async def flush_cache(self) -> None:
        if not self.finished_any_generation:
            # Skip flush_cache if no generation has occurred yet to avoid SGLang deadlock
            logger.info(f"Worker {self.name}: skipping flush_cache (no prior generations)")
            return
        with Timer() as timer:
            await self._engine_tokenizer_manager.flush_cache()
        logger.info(f"Flushed cache for worker: {self.name} in {timer.elapsed_seconds:.2f} seconds")

    @override
    async def release_gpu_memory(self, *, backup_weights_on_cpu: bool = True) -> None:
        assert not backup_weights_on_cpu, "Backing up weights on CPU is not supported in SGLangWorker."
        valid_tags = {"kv_cache", "weights"}
        tags = list(valid_tags)
        tags = [tag for tag in tags if self.on_gpu[tag]]
        if not tags:
            logger.info(f"All specified tags are already on CPU for worker: {self.name}.")
            return
        with Timer() as timer:
            obj = ReleaseMemoryOccupationReqInput(tags=tags)
            await self._engine_tokenizer_manager.release_memory_occupation(obj)
        for tag in tags:
            self.on_gpu[tag] = False
        gpu_utils.log_gpu_memory_after_move(self.name, tags, "cpu", timer.elapsed_seconds)

    @override
    async def resume_gpu_memory(self, tags: list[str] | None = None) -> None:
        valid_tags = {"kv_cache", "weights"}
        tags = tags if tags is not None else list(valid_tags)
        assert all(tag in valid_tags for tag in tags), f"Invalid tags: {tags}"
        tags = [tag for tag in tags if not self.on_gpu[tag]]
        if not tags:
            logger.warning(f"All specified tags are already on GPU for worker: {self.name}.")
            return

        with Timer() as timer:
            obj = ResumeMemoryOccupationReqInput(tags=tags)
            await self._engine_tokenizer_manager.resume_memory_occupation(obj)
        for tag in tags:
            self.on_gpu[tag] = True
        gpu_utils.log_gpu_memory_after_move(self.name, tags, "gpu", timer.elapsed_seconds)

    @override
    async def is_gpu_memory_released(self) -> bool:
        return sum(self.on_gpu.values()) < len(self.on_gpu)
