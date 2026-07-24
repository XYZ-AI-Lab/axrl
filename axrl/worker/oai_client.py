from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, override

import httpx

from axrl.data import GenerationInput, GenerationOutput, array_utils
from axrl.worker.infer_worker import InferWorker

if TYPE_CHECKING:
    from axrl.configs import OAIClientConfig, SamplingConfig

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class OAICompatibleGenerationClient(InferWorker[GenerationInput, GenerationOutput]):
    """Small generation client for OpenAI-compatible services.

    The first backend intentionally uses SGLang's native ``/generate`` request
    shape, because some callers need token ids and input-token logprobs. The
    class stays small: send ``GenerationInput``, validate the response, and
    return ``GenerationOutput``. It does not implement routing replay, chat
    message conversion, or session management.
    """

    def __init__(self, config: OAIClientConfig) -> None:
        super().__init__()
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/generate"
        self._client: httpx.AsyncClient | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            async with self._lock:
                if self._client is None or self._client.is_closed:
                    limits = httpx.Limits(
                        max_connections=self.config.max_connections,
                        max_keepalive_connections=self.config.max_keepalive_connections,
                    )
                    timeout = httpx.Timeout(self.config.request_timeout_seconds)
                    self._client = httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False)
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @override
    def shutdown(self) -> None:
        if self._client is None or self._client.is_closed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.close())
        else:
            self._close_task = loop.create_task(self.close())

    @override
    async def generate(self, req: GenerationInput) -> GenerationOutput:
        """Generate, retrying forever until a valid response is returned."""
        req.event_timing.mark_worker_received()
        attempt = 0
        while True:
            payload = self._build_payload(req, backend_request_id=f"sglang-{uuid.uuid4().hex}")
            try:
                output = await self._request_once(req=req, payload=payload, retry=attempt)
                output.event_timing.mark_worker_returned()
                return output
            except Exception as exc:
                attempt += 1
                sleep_seconds = self._retry_sleep_seconds(attempt)
                logger.warning(
                    "Generation request failed; retrying forever: session=%s attempt=%d sleep=%.2fs error_type=%s error=%s",
                    req.session_id,
                    attempt,
                    sleep_seconds,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)

    async def _request_once(self, *, req: GenerationInput, payload: dict[str, Any], retry: int) -> GenerationOutput:
        start_time = time.perf_counter()
        client = await self._get_client()
        response = await client.post(self.generate_url, json=payload)
        response.raise_for_status()
        data = response.json()
        assert isinstance(data, dict), f"Unexpected generation response type: {type(data).__name__}"
        return self._parse_response(req=req, data=data, retry=retry, elapsed_seconds=time.perf_counter() - start_time)

    def _build_payload(self, req: GenerationInput, *, backend_request_id: str) -> dict[str, Any]:
        input_ids = array_utils.to_int_list(req.input_ids)
        sampling_params = self._sampling_params(req.sampling_config or self.config.sampling_config, input_len=len(input_ids))
        max_new_tokens = int(sampling_params["max_new_tokens"])
        self._validate_generation_request(req, input_len=len(input_ids), max_new_tokens=max_new_tokens)
        if max_new_tokens == 0:
            # SGLang's HTTP /generate path can hang on a literal zero-token
            # decode. Request one token from the backend and drop it when
            # parsing so callers still see a pure input-logprob score result.
            sampling_params["max_new_tokens"] = 1
        payload: dict[str, Any] = {
            "rid": backend_request_id,
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        if req.capture_input_logprobs:
            payload["logprob_start_len"] = req.input_logprob_start_index - 1
        if req.lora_path is not None:
            payload["lora_path"] = req.lora_path
        return payload

    @staticmethod
    def _validate_generation_request(req: GenerationInput, *, input_len: int, max_new_tokens: int) -> None:
        if req.input_logprob_start_index > input_len:
            raise ValueError(f"input_logprob_start_index={req.input_logprob_start_index} exceeds input length {input_len}.")
        if req.capture_input_logprobs and req.input_logprob_start_index <= 0:
            raise ValueError("input_logprob_start_index must be greater than zero when capture_input_logprobs=True.")
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}.")
        if max_new_tokens == 0 and not req.capture_input_logprobs:
            raise ValueError("max_new_tokens=0 is only supported when capture_input_logprobs=True.")

    @staticmethod
    def _sampling_params(config: SamplingConfig, *, input_len: int) -> dict[str, Any]:
        params = config.model_dump(exclude_none=True)
        max_total_tokens = int(params.pop("max_total_tokens"))
        if "max_new_tokens" not in params:
            params["max_new_tokens"] = max(0, max_total_tokens - input_len)
        return params

    def _parse_response(
        self,
        *,
        req: GenerationInput,
        data: dict[str, Any],
        retry: int,
        elapsed_seconds: float,
    ) -> GenerationOutput:
        meta_info = data.get("meta_info")
        if not isinstance(meta_info, dict):
            raise TypeError(f"Generation response missing dict meta_info: {meta_info!r}")

        requested_zero_new_tokens = self._requested_max_new_tokens(req) == 0
        if requested_zero_new_tokens:
            output_ids: list[int] = []
            output_logprobs: list[float] = []
            output_text = ""
        else:
            output_ids = _int_list(data.get("output_ids", []), field_name="output_ids")
            output_logprobs = self._parse_output_logprobs(output_ids=output_ids, meta_info=meta_info)
            output_text = str(data.get("text") or "")
        input_token_ids: list[int] | None = None
        input_logprobs: list[float] | None = None
        if req.capture_input_logprobs:
            input_token_ids, input_logprobs = self._parse_input_logprobs(req=req, meta_info=meta_info)

        finish_reason_info = meta_info.get("finish_reason") or {}
        if isinstance(finish_reason_info, dict):
            finish_reason = str(finish_reason_info.get("type", "unknown"))
            stop_reason = finish_reason_info.get("matched")
        else:
            finish_reason = str(finish_reason_info)
            stop_reason = None

        output = GenerationOutput(
            session_id=req.session_id,
            output_ids=array_utils.as_i32(output_ids),
            output_logprobs=array_utils.as_f32(output_logprobs),
            output_text=output_text,
            output_text_with_special_tokens=output_text,
            cached_tokens=int(meta_info.get("cached_tokens", 0)),
            finish_reason=finish_reason,
            e2e_elapsed_seconds=elapsed_seconds,
            stop_reason=stop_reason,
            retry=retry,
            input_logprobs=array_utils.as_f32(input_logprobs) if input_logprobs is not None else None,
            input_logprob_token_ids=array_utils.as_i32(input_token_ids) if input_token_ids is not None else None,
            input_logprob_start_index=req.input_logprob_start_index if req.capture_input_logprobs else None,
            event_timing=req.event_timing,
        )
        self._validate_output(req=req, output=output)
        return output

    def _requested_max_new_tokens(self, req: GenerationInput) -> int:
        sampling_params = self._sampling_params(
            req.sampling_config or self.config.sampling_config,
            input_len=len(req.input_ids),
        )
        return int(sampling_params["max_new_tokens"])

    @staticmethod
    def _parse_output_logprobs(*, output_ids: list[int], meta_info: dict[str, Any]) -> list[float]:
        if not output_ids:
            return []
        entries = meta_info.get("output_token_logprobs")
        if not isinstance(entries, list):
            raise TypeError("Generation response missing output_token_logprobs for non-empty output_ids.")
        token_ids, logprobs, _ = _parse_logprob_entries(
            entries,
            field_name="output_token_logprobs",
            skip_initial_none=False,
        )
        if token_ids != output_ids:
            raise ValueError(f"output_token_logprobs token ids {token_ids} do not match output_ids {output_ids}.")
        return logprobs

    @staticmethod
    def _parse_input_logprobs(*, req: GenerationInput, meta_info: dict[str, Any]) -> tuple[list[int], list[float]]:
        entries = meta_info.get("input_token_logprobs")
        if not isinstance(entries, list):
            raise TypeError("Generation response missing input_token_logprobs.")
        token_ids, logprobs, skipped_initial_none = _parse_logprob_entries(
            entries,
            field_name="input_token_logprobs",
            skip_initial_none=True,
        )
        del skipped_initial_none
        input_ids = array_utils.to_int_list(req.input_ids)
        expected_token_ids = input_ids[req.input_logprob_start_index : req.input_logprob_start_index + len(token_ids)]
        if token_ids != expected_token_ids:
            raise ValueError(f"input_token_logprobs token ids {token_ids} do not match input_ids slice {expected_token_ids}.")
        return token_ids, logprobs

    @staticmethod
    def _validate_output(*, req: GenerationInput, output: GenerationOutput) -> None:
        if len(output.output_ids) != len(output.output_logprobs):
            raise ValueError(f"output_ids length {len(output.output_ids)} does not match output_logprobs length {len(output.output_logprobs)}.")
        if not req.capture_input_logprobs:
            return
        if output.input_logprobs is None or output.input_logprob_token_ids is None:
            raise ValueError("capture_input_logprobs=True but input logprobs were not returned.")
        if len(output.input_logprobs) != len(output.input_logprob_token_ids):
            raise ValueError(
                f"input_logprobs length {len(output.input_logprobs)} does not match "
                f"input_logprob_token_ids length {len(output.input_logprob_token_ids)}."
            )
        if output.input_logprob_start_index != req.input_logprob_start_index:
            raise ValueError(f"input_logprob_start_index mismatch: {output.input_logprob_start_index} != {req.input_logprob_start_index}.")

    def _retry_sleep_seconds(self, attempt: int) -> float:
        base = self.config.retry_initial_sleep_seconds
        cap = self.config.retry_max_sleep_seconds
        if base <= 0 or cap <= 0:
            return 0.0
        return min(cap, base * (2 ** min(attempt - 1, 10)))


def _int_list(value: Any, *, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"Expected {field_name} to be a list, got {type(value).__name__}.")
    return [int(x) for x in value]


def _parse_logprob_entries(
    entries: list[Any],
    *,
    field_name: str,
    skip_initial_none: bool,
) -> tuple[list[int], list[float], bool]:
    token_ids: list[int] = []
    logprobs: list[float] = []
    skipped_initial_none = False
    for index, entry in enumerate(entries):
        if not isinstance(entry, list | tuple) or len(entry) < 2:
            raise TypeError(f"Unexpected {field_name}[{index}] entry: {entry!r}")
        logprob_raw = entry[0]
        if logprob_raw is None:
            if not (skip_initial_none and index == 0):
                raise ValueError(f"Unexpected None logprob in {field_name}[{index}].")
            skipped_initial_none = True
            continue
        token_ids.append(int(entry[1]))
        logprobs.append(float(logprob_raw))
    return token_ids, logprobs, skipped_initial_none
