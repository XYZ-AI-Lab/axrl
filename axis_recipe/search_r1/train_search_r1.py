from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any, cast

from transformers import AutoTokenizer

from axis_recipe.search_r1.search_client import SearchClient
from axis_recipe.search_r1.search_env import SearchEnv
from axis_recipe.search_r1.search_r1_config import SearchR1Config
from axis_recipe.search_r1.search_r1_verifier import SearchR1Verifier
from axrl.agent.rollout_agent import RolloutAgent
from axrl.metrics.response_metric import ResponseMetricCalculator
from axrl.pipeline.controller import PipelineController
from axrl.pipeline.utils import rollout_from_env
from axrl.processor.appended_message_tokenizer import AppendedMessageTokenizer
from axrl.processor.conversation_tokenizer import ConversationTokenizer
from axrl.processor.processor_pool import ProcessorPool
from axrl.recipe.base_recipe import BaseRecipe
from axrl.utils import setup_logger
from axrl.utils.config_utils import load_and_validate_config

if TYPE_CHECKING:
    from axrl.data import Conversation, RolloutResult
    from axrl.pipeline.rollout_data import RolloutRuntime
    from axrl.worker.infer_worker import InferWorker


class SearchR1PipelineRecipe(BaseRecipe):
    def __init__(self, config: SearchR1Config) -> None:
        super().__init__(config)
        self.config: SearchR1Config = config
        self.pad_token_id: int | None = None

    def initialize_local_processors(self, worker_id: str) -> dict[str, InferWorker[Any, Any]]:
        assert worker_id, "worker_id must be non-empty when initializing Search-R1 workers."
        self.pad_token_id = _get_pad_token_id(self.config)
        return cast(
            "dict[str, InferWorker[Any, Any]]",
            {
                "verifier": ProcessorPool(SearchR1Verifier, config=self.config.verifier, num_processors=1),
                "metric": ProcessorPool(ResponseMetricCalculator, config=None, num_processors=1),
                "conv_tokenizer": ProcessorPool(ConversationTokenizer, config=self.config.rollout_worker.model, num_processors=1),
                "appended_message_tokenizer": ProcessorPool(AppendedMessageTokenizer, config=self.config.rollout_worker.model, num_processors=1),
                "search_client": SearchClient(
                    base_urls=_search_base_urls(),
                    request_timeout=self.config.search_client.request_timeout,
                    max_connections=self.config.search_client.max_connections,
                    max_keepalive_connections=self.config.search_client.max_keepalive_connections,
                    max_retries=self.config.search_client.max_retries,
                    retry_backoff_seconds=self.config.search_client.retry_backoff_seconds,
                ),
            },
        )

    async def run_rollout(self, conversation: Conversation, runtime: RolloutRuntime) -> RolloutResult:
        sampling_config = conversation.gen_state.sampling_config
        assert sampling_config is not None, f"Rollout conversation {conversation.conversation_id!r} is missing sampling config."
        assert "answer" in conversation.extra, f"Rollout conversation {conversation.conversation_id!r} is missing extra['answer']."

        conv_tokenizer = runtime.get_local_worker("conv_tokenizer")
        gen_input = await conv_tokenizer.generate(conversation)
        conversation.gen_state.input_ids = gen_input.input_ids

        assert self.pad_token_id is not None, "Search-R1 local processors must be initialized before rollout."
        env = SearchEnv(
            conv=conversation,
            label=conversation.extra["answer"],
            search_client=runtime.get_local_worker("search_client"),
            score_provider=runtime.get_local_worker("verifier"),
            metric_calculator=runtime.get_local_worker("metric"),
            max_length=self.config.rollout_worker.model.seq_length,
            pad_token_id=self.pad_token_id,
            appended_message_tokenizer=runtime.get_local_worker("appended_message_tokenizer"),
            search_limit=3,
        )
        agent = RolloutAgent(rollout_worker=runtime.rollout_worker)
        return await rollout_from_env(env, agent, sampling_config)


def _search_base_urls() -> list[str]:
    configured_urls = os.environ.get("AXRL_SEARCH_BASE_URLS")
    if configured_urls:
        return [url.strip() for url in configured_urls.split(",") if url.strip()]
    port = os.environ.get("AXRL_SEARCH_PORT", "18000")
    return [f"http://127.0.0.1:{port}"]


def _get_pad_token_id(config: SearchR1Config) -> int:
    tokenizer = AutoTokenizer.from_pretrained(
        config.rollout_worker.model.get_full_path(),
        trust_remote_code=config.rollout_worker.model.trust_remote_code,
    )
    pad_token_id = tokenizer.pad_token_id
    assert pad_token_id is not None, f"pad_token_id should not be None for model {config.rollout_worker.model.name}."
    return int(pad_token_id)


async def run_controller() -> None:
    setup_logger("info")
    config = load_and_validate_config(
        SearchR1Config,
        config_path="axis_recipe/search_r1/search-r1-config.yaml",
        print_configs=True,
    )
    controller = PipelineController(config, SearchR1PipelineRecipe(config))
    await controller.start()


if __name__ == "__main__":
    asyncio.run(run_controller())
