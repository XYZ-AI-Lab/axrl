import asyncio
import logging
from typing import TYPE_CHECKING

import pytest
import torch
from rich.pretty import pprint
from transformers import AutoTokenizer

from axrl.configs import EngineType, ModelConfig, RolloutWorkerConfig, SamplingConfig
from axrl.data import SampleTensorDict
from axrl.data.conversation import Conversation
from axrl.data.token_trace import TokenTrace
from axrl.example.message_examples import prompts
from axrl.processor.conversation_tokenizer import ConversationTokenizer
from axrl.utils import kl_utils
from axrl.utils.hf.download_model_from_hf import download_model
from axrl.worker.hf_worker import HFWorker
from tests.test_configs import all_engine_types, default_engine_type, make_worker

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("engine_type", all_engine_types)
def test_consistency(engine_type: EngineType) -> None:
    max_len = 4096
    model_config = ModelConfig(name="Qwen/Qwen3-0.6B", seq_length=max_len)
    download_model(model_config)

    worker_config = RolloutWorkerConfig(
        engine_type=engine_type,
        model=model_config,
        tp_size=2,
        num_workers=2,
        gpu_memory_utilization=0.6,
        sampling_config=SamplingConfig(max_total_tokens=max_len),
    )

    # create input ids from example conversation
    messages = prompts["short_lm_messages"]
    conversation = Conversation.from_dict({"messages": messages})
    conversation_tokenizer = ConversationTokenizer(worker_config.model)
    input_ids = conversation_tokenizer.process(conversation).input_ids

    # inference with rollout worker
    worker = make_worker(worker_config, use_ray_worker=True)
    worker.initialize()
    num_samples: int = 10
    from axrl.data.generation import GenerationInput

    rollout_inputs = [GenerationInput(session_id=f"{i}", input_ids=input_ids) for i in range(num_samples)]
    rollout_outputs: Sequence = asyncio.run(worker.batch_generate(rollout_inputs))

    # build samples from rollout outputs
    hf_tokenizer = AutoTokenizer.from_pretrained(model_config.get_full_path(), trust_remote_code=True)
    pad_token_id = hf_tokenizer.pad_token_id
    assert pad_token_id is not None, "pad_token_id must be set for padding"

    samples = []
    for out in rollout_outputs:
        trace = TokenTrace()
        trace.extend_tokens(input_ids, token_type="init")
        trace.extend_tokens(out.output_ids, logprobs=out.output_logprobs, token_type="assistant")
        samples.append(trace.to_sample(max_length=model_config.seq_length, pad_token_id=pad_token_id))

    logger.info(f"Example output text: {rollout_outputs[0].output_text}")

    # inference with HF
    hf_worker = HFWorker(model_path=worker_config.model.get_full_path())
    hf_worker.initialize()
    hf_logprobs, _ = hf_worker.compute_logprobs(SampleTensorDict.from_samples(samples), batch_size=2)
    hf_worker.shutdown()

    worker.shutdown()

    # compare logprobs from rollout and HF
    loss_mask = torch.tensor([sample.loss_mask for sample in samples])
    rollout_logprobs = torch.tensor([sample.rollout_logprobs for sample in samples], dtype=torch.float32)
    diff_result = kl_utils.compare_logprobs(loss_mask, hf_logprobs, rollout_logprobs)
    logger.info("Difference between HF and rollout logprobs:")
    pprint(diff_result)
    assert diff_result.cosine_similarity > 0.999, f"Cosine similarity is too low: {diff_result.cosine_similarity}"


if __name__ == "__main__":
    test_consistency(default_engine_type)
    print("Consistency test completed.")
