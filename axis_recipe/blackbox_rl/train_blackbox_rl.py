from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast, override
from urllib.parse import urlparse

from axis_recipe.blackbox_rl.config import BlackBoxRLConfig
from axis_recipe.blackbox_rl.leetcode_dataset import register_leetcode_datasets
from axis_recipe.blackbox_rl.openhands_env import OpenHandsEnv
from axrl.agent.rollout_agent import RolloutAgent
from axrl.data import RolloutResult
from axrl.metrics.response_metric import ResponseMetricCalculator
from axrl.openai_proxy import OpenAIChatAdapter, OpenAIChatAdapterConfig, OpenAIProxyServer, OpenAIProxySessionRegistry
from axrl.pipeline.controller import PipelineController
from axrl.processor.conversation_tokenizer import ConversationTokenizer
from axrl.processor.processor_pool import ProcessorPool
from axrl.recipe.base_recipe import BaseRecipe
from axrl.utils import setup_logger
from axrl.utils.config_utils import load_and_validate_config
from axrl.utils.tunnel import Tunnel, allow_out_for_base_url, discover_local_ip, format_template, is_public_routable_host

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ray.actor import ActorHandle

    from axis_recipe.blackbox_rl.config import OpenAIProxyConfig
    from axrl.configs import ModelConfig
    from axrl.data import Conversation, GenerationOutput
    from axrl.metrics.response_metric import ResponseMetric
    from axrl.openai_proxy.chat_adapter import OpenAIChatAdapterInput, OpenAIChatAdapterOutput
    from axrl.pipeline.rollout_data import RolloutRuntime
    from axrl.ray.ray_infer_worker import RayInferWorker
    from axrl.ray.ray_rollout_worker import RayRolloutWorker
    from axrl.verifier.base_verifier import VerifierInput, VerifierOutput
    from axrl.worker.infer_worker import InferWorker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlackBoxOpenAIProxyHandle:
    registry_actor: ActorHandle
    base_url: str
    e2b_allow_out: tuple[str, ...]
    api_key: str
    pad_token_id: int

    def shutdown(self) -> None:
        """Runtime copies do not own the driver-side proxy service."""

    def session_base_url(self, session_id: str) -> str:
        return f"{self.base_url}/sessions/{session_id}/v1"


class BlackBoxOpenAIProxy:
    def __init__(
        self,
        *,
        registry: OpenAIProxySessionRegistry,
        server: OpenAIProxyServer,
        base_url: str,
        e2b_allow_out: tuple[str, ...],
        api_key: str,
        tunnel: Tunnel | None = None,
    ) -> None:
        self.registry = registry
        self.server = server
        self.base_url = base_url.rstrip("/")
        self.e2b_allow_out = e2b_allow_out
        self.api_key = api_key
        self.tunnel = tunnel
        self._stopped = False

    @classmethod
    async def start(cls, config: BlackBoxRLConfig) -> BlackBoxOpenAIProxy:
        proxy_config = config.openai_proxy
        registry = OpenAIProxySessionRegistry(request_timeout_seconds=proxy_config.request_timeout_seconds)
        api_key = secrets.token_urlsafe(32)
        server = OpenAIProxyServer(
            host=proxy_config.host,
            port=proxy_config.port,
            registry=registry,
            request_timeout_seconds=proxy_config.request_timeout_seconds,
            auth_token=api_key,
        )
        await server.start()
        tunnel: Tunnel | None = None
        try:
            base_url, e2b_allow_out, tunnel = await _resolve_proxy_exposure(proxy_config, server)
            proxy = cls(
                registry=registry,
                server=server,
                base_url=base_url,
                e2b_allow_out=e2b_allow_out,
                api_key=api_key,
                tunnel=tunnel,
            )
            logger.info("OpenAI proxy started at %s; exposed base URL=%s; e2b_allow_out=%s.", server.base_url, proxy.base_url, proxy.e2b_allow_out)
            return proxy
        except Exception:
            if tunnel is not None:
                await tunnel.stop()
            await server.stop()
            registry.shutdown()
            raise

    def handle(self, *, pad_token_id: int) -> BlackBoxOpenAIProxyHandle:
        return BlackBoxOpenAIProxyHandle(
            registry_actor=self.registry.get_actor_handle(),
            base_url=self.base_url,
            e2b_allow_out=self.e2b_allow_out,
            api_key=self.api_key,
            pad_token_id=pad_token_id,
        )

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.tunnel is not None:
            await self.tunnel.stop()
        await self.server.stop()
        self.registry.shutdown()


class BlackBoxRLRecipe(BaseRecipe):
    def __init__(self, config: BlackBoxRLConfig) -> None:
        super().__init__(config)
        self.config: BlackBoxRLConfig = config

    @override
    async def register_datasets(self) -> None:
        register_leetcode_datasets()

    async def start_services(self) -> dict[str, Any]:
        return {"openai_proxy": await BlackBoxOpenAIProxy.start(self.config)}

    async def stop_services(self, services: Mapping[str, Any]) -> None:
        proxy = cast("BlackBoxOpenAIProxy | None", services.get("openai_proxy"))
        if proxy is not None:
            await proxy.stop()

    def initialize_shared_workers(self, services: Mapping[str, Any]) -> dict[str, RayRolloutWorker | RayInferWorker[Any, Any]]:
        proxy = cast("BlackBoxOpenAIProxy", services["openai_proxy"])
        return cast(
            "dict[str, RayRolloutWorker | RayInferWorker[Any, Any]]",
            {"openai_proxy": proxy.handle(pad_token_id=_get_pad_token_id(self.config.rollout_worker.model))},
        )

    def initialize_local_processors(self, worker_id: str) -> dict[str, InferWorker[Any, Any]]:
        assert worker_id, "worker_id must be non-empty when initializing local blackbox workers."
        from axis_recipe.blackbox_rl.leetcode_verifier import LeetCodeVerifier

        verifier_config = {
            "e2b": self.config.verifier_e2b.model_dump(),
            "timeout": self.config.verifier_timeout_seconds,
            "memory_limit_bytes": self.config.verifier_memory_limit_gib * 1024**3,
        }
        return {
            "openai_adapter": ProcessorPool(
                OpenAIChatAdapter,
                config=OpenAIChatAdapterConfig(
                    model=self.config.rollout_worker.model,
                    tool_call_parser=self.config.openai_proxy.tool_call_parser,
                    reasoning_parser=self.config.openai_proxy.reasoning_parser,
                ),
                num_processors=self.config.openai_proxy.adapter_num_processors,
                timeout_seconds=self.config.openai_proxy.adapter_timeout_seconds,
            ),
            "metric": ProcessorPool(ResponseMetricCalculator, config=None, num_processors=1),
            "conv_tokenizer": ProcessorPool(ConversationTokenizer, config=self.config.rollout_worker.model, num_processors=1),
            "verifier:newfacade/LeetCodeDataset": ProcessorPool(
                LeetCodeVerifier,
                config=verifier_config,
                num_processors=self.config.verifier_num_processors,
                timeout_seconds=self.config.verifier_timeout_seconds,
            ),
        }

    async def run_rollout(self, conversation: Conversation, runtime: RolloutRuntime) -> RolloutResult:
        session_id = conversation.gen_state.session_id or conversation.conversation_id
        assert session_id, "Blackbox rollout conversation must have a session id or conversation id."
        sampling_config = conversation.gen_state.sampling_config
        assert sampling_config is not None, f"Blackbox rollout conversation {session_id!r} is missing gen_state.sampling_config."
        assert "answer" in conversation.extra, f"Blackbox rollout conversation {session_id!r} is missing extra['answer']."

        proxy_handle = cast("BlackBoxOpenAIProxyHandle", runtime.get_shared_worker("openai_proxy"))
        registry = OpenAIProxySessionRegistry.from_remote_actor(proxy_handle.registry_actor)
        adapter = cast("ProcessorPool[OpenAIChatAdapterInput, OpenAIChatAdapterOutput]", runtime.get_local_worker("openai_adapter"))
        metric_calculator = cast("InferWorker[GenerationOutput, ResponseMetric]", runtime.get_local_worker("metric"))
        score_provider = cast("InferWorker[VerifierInput, VerifierOutput]", runtime.get_local_worker(f"verifier:{conversation.source}"))
        env = OpenHandsEnv(
            conv=conversation,
            label=conversation.extra["answer"],
            registry=registry,
            adapter=adapter,
            openhands_config=self.config.openhands.model_copy(update={"api_key": proxy_handle.api_key}),
            llm_base_url=proxy_handle.session_base_url(session_id),
            e2b_allow_out=proxy_handle.e2b_allow_out,
            llm_model=self.config.openai_proxy.served_model_name or self.config.rollout_worker.model.name,
            score_provider=score_provider,
            metric_calculator=metric_calculator,
            config=self.config.openhands_env,
            max_length=self.config.rollout_worker.model.seq_length,
            pad_token_id=proxy_handle.pad_token_id,
        )
        agent = RolloutAgent(runtime.rollout_worker)
        observation = await env.start()
        assert not env.done, "OpenHandsEnv.start() must return a live first observation."
        while True:
            generation_output = await agent.act(observation, sampling_config)
            observation, _, done, sample, step_metric = await env.step(generation_output)
            if done:
                assert step_metric is not None
                return RolloutResult(conversation=observation, trace=sample, metric=step_metric)


def _load_tokenizer(model_config: ModelConfig) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_config.get_full_path(),
        trust_remote_code=model_config.trust_remote_code,
    )


def _get_pad_token_id(model_config: ModelConfig) -> int:
    tokenizer = _load_tokenizer(model_config)
    pad_token_id = tokenizer.pad_token_id
    assert pad_token_id is not None, f"pad_token_id should not be None for model {model_config.name}."
    logger.info("Pad token ID for model %s: %s", model_config.name, pad_token_id)
    return int(pad_token_id)


def _proxy_public_host(proxy_config: OpenAIProxyConfig) -> str:
    if proxy_config.public_host:
        return proxy_config.public_host
    if proxy_config.host and proxy_config.host not in {"0.0.0.0", "::"}:  # noqa: S104 - compare against bind addresses.
        return proxy_config.host
    return discover_local_ip()


async def _resolve_proxy_exposure(
    proxy_config: OpenAIProxyConfig,
    server: OpenAIProxyServer,
) -> tuple[str, tuple[str, ...], Tunnel | None]:
    public_host = _proxy_public_host(proxy_config)
    direct_base_url = f"http://{public_host}:{server.port}"
    template_vars = {"base_url": direct_base_url, "host": server.host, "port": server.port}
    exposure = proxy_config.exposure
    if exposure.exposed_base_url is not None and exposure.tunnel is not None:
        raise ValueError("Set only one of openai_proxy.exposure.exposed_base_url or openai_proxy.exposure.tunnel.")
    tunnel: Tunnel | None = None
    if exposure.tunnel is not None:
        tunnel = await Tunnel.start(
            exposure.tunnel,
            template_vars=template_vars,
            drain_task_name="blackbox-openai-proxy-tunnel-drain",
        )
        base_url = tunnel.base_url
    elif exposure.exposed_base_url is not None:
        base_url = format_template(exposure.exposed_base_url, template_vars)
    else:
        base_url = direct_base_url
        host = urlparse(base_url).hostname
        if not is_public_routable_host(host):
            raise ValueError(
                "OpenHands now runs in E2B, but the OpenAI proxy has no E2B-routable exposure. "
                "Set openai_proxy.exposure.exposed_base_url or openai_proxy.exposure.tunnel in the config."
            )
    return base_url.rstrip("/"), allow_out_for_base_url(base_url, exposure.allow_out, context="openai_proxy.exposure.allow_out"), tunnel


async def run_controller() -> None:
    setup_logger("info")
    config = load_and_validate_config(
        BlackBoxRLConfig,
        config_path="axis_recipe/blackbox_rl/blackbox-rl-config.yaml",
        print_configs=True,
    )
    controller = PipelineController(config, BlackBoxRLRecipe(config))
    await controller.start()


if __name__ == "__main__":
    asyncio.run(run_controller())
