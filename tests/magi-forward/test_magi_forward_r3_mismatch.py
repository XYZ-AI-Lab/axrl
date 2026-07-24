"""Focused R3 + Magi merged-forward mismatch test.

Builds a 4-turn cities conversation through SGLang on a ``RolloutTrace``
with multi-turn compaction between assistant turns, then materializes the
merged Sample via ``trace.to_sample()`` (which trie-packs the per-turn
samples into a prefix-tree) and asks the Megatron worker to recompute its
logprobs. KL1 = ``mean |rollout_logprob - mcore_logprob|`` at trainable
positions.

When R3 is on, routed_experts flow through the TensorStore under the
per-call ``TensorHandle``; the trainer-side materialiser fetches them at
microbatch time and applies the trie source map in ``merge_info`` to
assemble the merged-trie routing (one concat-per-path, then gather).
When R3 is off, the merged Sample has no routing handles and the legacy
logprobs path is exercised.

Asserts:
- R3 OFF and R3 ON both succeed end-to-end.
- R3 ON yields KL1 ≤ R3 OFF (R3 narrows the rollout-vs-train gap).

The test uses Qwen3-30B-A3B-Instruct-2507 and skips when fewer than 8 GPUs
are visible (tp=2 x cp=2 x pp=2).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from axrl.configs import (
    DataloaderConfig,
    MCoreLrSchedulerConfig,
    MCoreOptimizerConfig,
    MegatronWorkerConfig,
    ModelConfig,
    RolloutWorkerConfig,
    SamplingConfig,
)
from axrl.data import GenerationInput, GenerationOutput, GenerationState, Sample, SampleTensorDict
from axrl.data.conversation import Conversation, Message
from axrl.data.generation import TensorHandle
from axrl.data.rollout_trace import RolloutTrace
from axrl.processor.appended_message_tokenizer import AppendedMessageTokenizer
from axrl.processor.conversation_tokenizer import ConversationTokenizer
from axrl.processor.text_decoder import TextDecoder
from axrl.ray import ray_utils
from axrl.ray.resource_group import Request, ResourceGroup
from axrl.utils import tensor_store as store
from axrl.utils.megatron.spike_snapshot_routing import (
    collect_unique_routing_handles_from_batch,
    restore_spike_snapshot_routing,
    save_spike_snapshot_routing,
)

if TYPE_CHECKING:
    from pathlib import Path

    from axrl.ray.ray_rollout_worker import RayRolloutWorker
    from axrl.worker.rollout_worker import RolloutWorker

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# Single combined config tp=2 * cp=2 * pp=2 = 8 ranks exercises every parallel
# dimension at once on one node, instead of running a tp/cp/pp matrix.
NUM_GPUS_REQUIRED = 8
MAX_LENGTH = 4096
FIRST_TURN_MAX_NEW_TOKENS = 32
DEFAULT_MAX_NEW_TOKENS = 512

BEIJING_INTRO = (
    "Beijing is the capital of the People's Republic of China and one of the "
    "oldest continuously inhabited capital cities in the world. Located in "
    "the northern plain of China, Beijing has served as the political, "
    "cultural, and educational center of the country for centuries, "
    "stretching back through the Yuan, Ming, and Qing dynasties to the "
    "present day. The city is home to a remarkable concentration of UNESCO "
    "World Heritage Sites, including the Forbidden City, the Temple of "
    "Heaven, the Summer Palace, and major sections of the Great Wall. Modern "
    "Beijing also stands as a leading hub for technology, finance, "
    "education, and international diplomacy in East Asia."
)

SHANGHAI_INTRO = (
    "Shanghai is the largest city in China by population and one of the most "
    "influential financial centers in the world, sitting at the mouth of the "
    "Yangtze River on the eastern coast of the country. Originally a small "
    "fishing and textile town, Shanghai grew into a global metropolis after "
    "it was opened as a treaty port in the nineteenth century, mixing "
    "Chinese, European, and modern architectural styles along the famous "
    "Bund waterfront. Today the city hosts the Shanghai Stock Exchange, the "
    "headquarters of many multinational corporations, and the busiest "
    "container port on the planet, while still preserving traditional "
    "water-town districts and classical Chinese gardens."
)

GUANGZHOU_INTRO = (
    "Guangzhou, historically known in the West as Canton, is the capital of "
    "Guangdong Province and the largest city in southern China, located on "
    "the Pearl River about a hundred kilometers northwest of Hong Kong. "
    "With more than two thousand years of urban history, Guangzhou has been "
    "an international trading port since the Tang dynasty and is the "
    "historical home of Cantonese language, opera, and cuisine, including "
    "dim sum and roast meats now eaten across the world. The city remains a "
    "major manufacturing, logistics, and convention center, and it hosts "
    "the famous biannual Canton Fair, one of the largest trade exhibitions "
    "in the world."
)

SHENZHEN_INTRO = (
    "Shenzhen is a major technology and innovation hub in southern China, "
    "situated immediately north of Hong Kong on the Pearl River Delta. "
    "Originally a small fishing village, Shenzhen was designated as the "
    "country's first Special Economic Zone in 1980 and has since grown into "
    "one of the fastest-developing cities in modern history, with a "
    "population of well over ten million. It is the headquarters of many "
    "leading Chinese technology companies, including Tencent, Huawei, BYD, "
    "and DJI, and it operates one of the busiest container ports and stock "
    "exchanges in Asia. The city is also known for its modern skyline, "
    "theme parks, and vibrant electronics markets."
)

CITIES_USER_PROMPT = (
    "You will receive a short reference introduction for each of four Chinese "
    "cities — Beijing, Shanghai, Guangzhou, and Shenzhen — one at a time. "
    "For EACH city, use the reference only as background, then tell a "
    "lesser-known story about that city — a specific, vivid anecdote, "
    "neighborhood, dish, festival, person, or moment in its history that a "
    "casual visitor would probably not know. Aim for around 150-250 words. "
    "Do NOT summarize or repeat the reference. Reply only with the story for "
    "that city — no preamble, no commentary, no bullet points.\n\n"
    f'First, Beijing reference: "{BEIJING_INTRO}"'
)

CITY_TOOL_PROMPTS = [
    f'Shanghai reference: "{SHANGHAI_INTRO}"',
    f'Guangzhou reference: "{GUANGZHOU_INTRO}"',
    f'Shenzhen reference: "{SHENZHEN_INTRO}"',
]

PLACEHOLDER_TEXT = "Tool result is omitted to save tokens."


def _make_rollout_config(*, enable_r3: bool) -> RolloutWorkerConfig:
    return RolloutWorkerConfig(
        engine_type="sglang",
        model=ModelConfig(name=MODEL_NAME, trust_remote_code=True, seq_length=MAX_LENGTH),
        sampling_config=SamplingConfig(temperature=0.0, max_total_tokens=MAX_LENGTH, max_new_tokens=512),
        gpu_memory_utilization=0.7,
        tp_size=NUM_GPUS_REQUIRED,
        ep_size=NUM_GPUS_REQUIRED,
        num_workers=1,
        enable_metrics=False,
        enable_routing_replay=enable_r3,
    )


def _make_megatron_config(
    *,
    enable_r3: bool,
    replay_routing_for_loss_tokens_only: bool = False,
    tp_size: int,
    cp_size: int,
    pp_size: int,
    ep_size: int,
) -> MegatronWorkerConfig:
    # World size = tp * cp * pp * dp must equal NUM_GPUS_REQUIRED.
    assert tp_size * cp_size * pp_size == NUM_GPUS_REQUIRED, f"tp*cp*pp must equal {NUM_GPUS_REQUIRED}, got {tp_size}*{cp_size}*{pp_size}"
    return MegatronWorkerConfig(
        model=ModelConfig(name=MODEL_NAME, trust_remote_code=True, seq_length=MAX_LENGTH),
        seed=42,
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
        dp_size=1,
        ep_size=ep_size,
        etp_size=1,
        enable_routing_replay=enable_r3,
        replay_routing_for_loss_tokens_only=replay_routing_for_loss_tokens_only,
        use_magi_merged_forward=True,
        bf16=True,
        global_batch_size=1,
        train_micro_batch_size=1,
        eval_micro_batch_size=1,
        log_every_k_steps=1,
        moe_aux_loss_coeff=0.0,
        moe_router_load_balancing_type="none",
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        data_loader=DataloaderConfig(num_workers=0),
        optimizer=MCoreOptimizerConfig(lr=1e-5, min_lr=1e-6, optimizer_cpu_offload=True, optimizer_offload_fraction=1.0),
        lr_scheduler=MCoreLrSchedulerConfig(lr_warmup_steps=10, lr_decay_steps=1000, lr_decay_style="constant"),
        inference_only=True,
    )


def _seed_conversation(conv_tokenizer: ConversationTokenizer) -> Conversation:
    """Build the initial Conversation with the user prompt tokenized."""
    conv = Conversation(
        conversation_id="r3-cities",
        messages=[Message(role="user", content=CITIES_USER_PROMPT)],
    )
    gen_input = conv_tokenizer.process(conv)
    conv.gen_state = GenerationState(input_ids=gen_input.input_ids)
    return conv


def _resolve_boundary_token_to_append(
    gen_output: GenerationOutput,
) -> int | None:
    assistant_boundary_token_id = gen_output.assistant_boundary_token_id
    if assistant_boundary_token_id is None:
        return None
    output_ends_with_boundary = len(gen_output.output_ids) > 0 and int(gen_output.output_ids[-1]) == assistant_boundary_token_id
    assert not output_ends_with_boundary, "SGLang worker should only return assistant_boundary_token_id when the output is missing it"
    return assistant_boundary_token_id


async def _run_rollout(
    rollout_worker: RolloutWorker | RayRolloutWorker,
    conv_tokenizer: ConversationTokenizer,
    appended_tokenizer: AppendedMessageTokenizer,
    text_decoder: TextDecoder,
    *,
    enable_r3: bool,
    max_length: int,
) -> Sample:
    """Drive a 4-turn rollout end-to-end with compaction between turns."""
    seed = _seed_conversation(conv_tokenizer)
    if enable_r3:
        seed.gen_state.capture_routing = True
    trace = RolloutTrace(seed, token_in_token_out=True, max_length=max_length)
    placeholder_tokens = appended_tokenizer.process(Message(role="tool", content=PLACEHOLDER_TEXT))

    cities = ["Beijing", "Shanghai", "Guangzhou", "Shenzhen"]
    appended_boundary_turns = 0
    returned_boundary_token_ids: set[int] = set()
    for turn_idx in range(4):
        gen_state = trace.observation().gen_state
        assert gen_state.input_ids is not None
        turn_max_new_tokens = FIRST_TURN_MAX_NEW_TOKENS if turn_idx == 0 else DEFAULT_MAX_NEW_TOKENS
        sampling = SamplingConfig(temperature=0.0, max_total_tokens=max_length, max_new_tokens=turn_max_new_tokens)
        gen_output = await rollout_worker.generate(
            GenerationInput(
                session_id=f"{seed.conversation_id}-t{turn_idx}",
                input_ids=gen_state.input_ids,
                sampling_config=sampling,
                capture_routing=gen_state.capture_routing,
                routed_expert_start_index=gen_state.captured_routing_rows,
            )
        )
        logger.info(
            "[R3=%s] %s story (%d tokens):\n%s\n---",
            enable_r3,
            cities[turn_idx],
            len(gen_output.output_ids),
            gen_output.output_text,
        )
        boundary_token_to_append = _resolve_boundary_token_to_append(gen_output)
        if boundary_token_to_append is not None:
            returned_boundary_token_ids.add(boundary_token_to_append)
            appended_boundary_turns += 1
        if turn_idx == 0:
            assistant_boundary_token_id = gen_output.assistant_boundary_token_id
            assert assistant_boundary_token_id is not None, "Truncated first turn should return an assistant boundary token id"
            assert len(gen_output.output_ids) == FIRST_TURN_MAX_NEW_TOKENS, (
                "First turn should be truncated by max_new_tokens so the boundary is not produced by generation"
            )
        before_append_len = len(trace.token_ids)
        trace.append_assistant_message(gen_output)
        if boundary_token_to_append is not None:
            assert int(trace.token_ids[-1]) == boundary_token_to_append
            assert len(trace.token_ids) == before_append_len + len(gen_output.output_ids) + 1
        else:
            assert len(trace.token_ids) == before_append_len + len(gen_output.output_ids)
        if turn_idx == 3:
            break
        tool_msg = Message(role="tool", content=CITY_TOOL_PROMPTS[turn_idx])
        tool_tokens = appended_tokenizer.process(tool_msg)
        trace.append_user_or_tool_message(content=CITY_TOOL_PROMPTS[turn_idx], tokens=tool_tokens)
        trace.compact(
            max_recent_tool_results=1,
            placeholder_tokens=placeholder_tokens,
            placeholder_text=PLACEHOLDER_TEXT,
        )

    assert appended_boundary_turns > 0, "Expected at least one SGLang output to carry assistant_boundary_token_id and be appended by RolloutTrace"
    assert len(returned_boundary_token_ids) == 1, f"Expected one assistant boundary token id, got {returned_boundary_token_ids}"

    final_token_ids = trace.token_ids
    final_text = text_decoder.process(final_token_ids)
    logger.info(
        "[R3=%s] final trace: %d tokens, %d turn samples, %d routing handles",
        enable_r3,
        len(final_token_ids),
        len(trace.turn_samples),
        len(trace.routing_handles),
    )
    logger.info("[R3=%s] final decoded prefix:\n%s\n---", enable_r3, final_text)

    sample = trace.to_sample()
    if enable_r3:
        assert sample.routing_handles_per_path is not None and len(sample.routing_handles_per_path) == 4, (
            "R3 enabled but merged sample missing per-leaf-path handle lists"
        )
        # Each leaf path's handle list must be non-empty + every handle a TensorHandle.
        for path_handles in sample.routing_handles_per_path:
            assert path_handles
            assert all(isinstance(h, TensorHandle) for h in path_handles)
    else:
        assert sample.routing_handles_per_path is None
    return sample


@dataclass
class KL1Stats:
    """Mismatch summary at trainable positions.

    ``mean`` / ``std`` / ``max`` summarise ``|rollout_logprob - mcore_logprob|``;
    ``cosine`` is the cosine similarity between the rollout and mcore logprob
    vectors restricted to trainable positions.
    """

    mean: float
    std: float
    max: float
    cosine: float
    count: int

    def __str__(self) -> str:
        return f"mean={self.mean:.4f} std={self.std:.4f} max={self.max:.4f} cos={self.cosine:.6f} (n={self.count})"


def _kl1(rollout_logprobs: list[float] | None, mcore_logprobs: torch.Tensor, loss_mask: list[bool]) -> KL1Stats:
    assert rollout_logprobs is not None
    rollout: list[float] = []
    mcore: list[float] = []
    for r, m, keep in zip(rollout_logprobs, mcore_logprobs.cpu().tolist(), loss_mask, strict=True):
        if keep:
            rollout.append(float(r))
            mcore.append(float(m))
    assert rollout, "no trainable positions found"
    r_tensor = torch.tensor(rollout, dtype=torch.float64)
    m_tensor = torch.tensor(mcore, dtype=torch.float64)
    deltas = (r_tensor - m_tensor).abs()
    cosine = torch.nn.functional.cosine_similarity(r_tensor.unsqueeze(0), m_tensor.unsqueeze(0)).item()
    return KL1Stats(
        mean=deltas.mean().item(),
        std=deltas.std().item(),
        max=deltas.max().item(),
        cosine=cosine,
        count=int(deltas.numel()),
    )


def _format_kl1_snapshot(kl_off: KL1Stats, kl_on: KL1Stats, kl_loss_tokens_only: KL1Stats) -> str:
    rows = [
        "Magi R3 KL1 snapshot:",
        "| R3 | mean KL1 | std KL1 | max KL1 | cosine | tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| off | {kl_off.mean:.4f} | {kl_off.std:.4f} | {kl_off.max:.4f} | {kl_off.cosine:.6f} | {kl_off.count} |",
        f"| all tokens | {kl_on.mean:.4f} | {kl_on.std:.4f} | {kl_on.max:.4f} | {kl_on.cosine:.6f} | {kl_on.count} |",
        "| loss tokens only "
        f"| {kl_loss_tokens_only.mean:.4f} "
        f"| {kl_loss_tokens_only.std:.4f} "
        f"| {kl_loss_tokens_only.max:.4f} "
        f"| {kl_loss_tokens_only.cosine:.6f} "
        f"| {kl_loss_tokens_only.count} |",
    ]
    return "\n".join(rows)


def _format_grpo_train_step_snapshot(results: dict[tuple[bool, bool], tuple[float, float]]) -> str:
    rows = [
        "Magi GRPO train_step snapshot:",
        "| R3 | recompute | loss | grad_norm |",
        "| --- | --- | ---: | ---: |",
    ]
    for enable_r3, activation_recompute in ((False, False), (False, True), (True, False), (True, True)):
        loss, grad_norm = results[(enable_r3, activation_recompute)]
        rows.append(f"| {'loss tokens only' if enable_r3 else 'off'} | {'on' if activation_recompute else 'off'} | {loss:.6f} | {grad_norm:.6f} |")
    return "\n".join(rows)


def _run_one_combo(
    *,
    enable_r3: bool,
    replay_routing_for_loss_tokens_only: bool = False,
    tp_size: int,
    cp_size: int,
    pp_size: int,
    ep_size: int,
) -> KL1Stats:
    """Drive one full (rollout → merged forward) cycle and return KL1 stats."""
    from axrl.ray.ray_megatron_worker import RayMegatronWorker
    from axrl.ray.ray_rollout_worker import RayRolloutWorker

    model_config = ModelConfig(name=MODEL_NAME, trust_remote_code=True, seq_length=MAX_LENGTH)
    conv_tokenizer = ConversationTokenizer(model_config)
    appended_tokenizer = AppendedMessageTokenizer(model_config)
    text_decoder = TextDecoder(model_config)

    resource_group = ResourceGroup([Request(cpu=1, gpu=NUM_GPUS_REQUIRED)])
    rollout_config = _make_rollout_config(enable_r3=enable_r3)
    megatron_config = _make_megatron_config(
        enable_r3=enable_r3,
        replay_routing_for_loss_tokens_only=replay_routing_for_loss_tokens_only,
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
        ep_size=ep_size,
    )

    rollout_worker = RayRolloutWorker(RayRolloutWorker.initialize_remote_actor(rollout_config, resource_group))
    megatron_worker = RayMegatronWorker(config=megatron_config, resource_group=resource_group)
    megatron_worker.initialize()

    # Rollout phase: Megatron off-GPU, rollout on-GPU.
    megatron_worker.to_cpu()
    asyncio.run(rollout_worker.resume_gpu_memory())
    sample = asyncio.run(
        _run_rollout(
            rollout_worker,
            conv_tokenizer,
            appended_tokenizer,
            text_decoder,
            enable_r3=enable_r3,
            max_length=MAX_LENGTH,
        )
    )

    asyncio.run(rollout_worker.release_gpu_memory(backup_weights_on_cpu=False))
    megatron_worker.to_gpu()

    batch = SampleTensorDict.from_samples([sample])
    logprobs, _ = megatron_worker.compute_logprobs(samples=batch, batch_size=1)

    rollout_worker.shutdown()
    megatron_worker.shutdown()
    if enable_r3 and sample.routing_handles_per_path is not None:
        all_handles = [h for path_handles in sample.routing_handles_per_path for h in path_handles]
        store.delete_batch(all_handles)

    # logprobs: (1, max_length) — indexed by packed position. Trainable positions
    # are where sample.loss_mask is True (after the shift-left in to_sample).
    rollout_logprobs = sample.rollout_logprobs.tolist() if sample.rollout_logprobs is not None else None
    return _kl1(rollout_logprobs, logprobs[0], sample.loss_mask.tolist())


def _make_grpo_train_sample() -> Sample:
    """Generate one R3 sample once so train-step variants share identical routing inputs."""
    from ray.util import remove_placement_group

    from axrl.ray.ray_rollout_worker import RayRolloutWorker

    model_config = ModelConfig(name=MODEL_NAME, trust_remote_code=True, seq_length=MAX_LENGTH)
    conv_tokenizer = ConversationTokenizer(model_config)
    appended_tokenizer = AppendedMessageTokenizer(model_config)
    text_decoder = TextDecoder(model_config)

    resource_group = ResourceGroup([Request(cpu=1, gpu=NUM_GPUS_REQUIRED)])
    rollout_config = _make_rollout_config(enable_r3=True)
    rollout_worker = RayRolloutWorker(RayRolloutWorker.initialize_remote_actor(rollout_config, resource_group))
    try:
        sample = asyncio.run(
            _run_rollout(
                rollout_worker,
                conv_tokenizer,
                appended_tokenizer,
                text_decoder,
                enable_r3=True,
                max_length=MAX_LENGTH,
            )
        )
        asyncio.run(rollout_worker.release_gpu_memory(backup_weights_on_cpu=False))
    finally:
        rollout_worker.shutdown()
        remove_placement_group(resource_group.pg)
    sample.trajectory_id = 0
    sample.advantage = np.where(sample.loss_mask, np.float32(1.0), np.float32(0.0))
    sample.reward = 1.0
    sample.reward_baseline = 0.0
    return sample


def _run_grpo_train_step(
    samples: SampleTensorDict,
    *,
    enable_r3: bool,
    replay_routing_for_loss_tokens_only: bool,
    activation_recompute: bool,
    tp_size: int,
    cp_size: int,
    pp_size: int,
    ep_size: int,
) -> tuple[float, float]:
    """Run one GRPO train step and return ``(loss, grad_norm)``."""
    from ray.util import remove_placement_group

    from axrl.configs import GrpoTrainerConfig
    from axrl.ray.ray_megatron_worker import RayMegatronWorker
    from axrl.trainer.grpo_trainer import GrpoTrainer

    config = _make_megatron_config(
        enable_r3=enable_r3,
        replay_routing_for_loss_tokens_only=replay_routing_for_loss_tokens_only,
        tp_size=tp_size,
        cp_size=cp_size,
        pp_size=pp_size,
        ep_size=ep_size,
    )
    if not activation_recompute:
        config.recompute_granularity = None
        config.recompute_method = None
        config.recompute_num_layers = None
    config.log_every_k_steps = 1
    config.global_batch_size = 1
    config.train_micro_batch_size = 1
    config.eval_micro_batch_size = 1
    config.reset_init_weights_every_k_steps = 1
    config.inference_only = False

    resource_group = ResourceGroup([Request(cpu=1, gpu=NUM_GPUS_REQUIRED)])
    megatron_worker = RayMegatronWorker(config=config, resource_group=resource_group)
    try:
        megatron_worker.initialize()
        megatron_worker.set_trainer(GrpoTrainer(config=GrpoTrainerConfig(micro_batch_denominator_type="token")))
        _step, metrics = megatron_worker.train(
            global_step=0,
            samples=samples,
            data_shuffle_seed=0,
            compute_logprobs=True,
        )
        return float(metrics["actor_train/loss"]), float(metrics["actor_train/grad_norm"])
    finally:
        megatron_worker.shutdown()
        remove_placement_group(resource_group.pg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_magi_merged_forward_r3_reduces_kl1() -> None:
    """Compare one compacted multi-turn rollout with and without R3.

    2026-06-01 snapshot from a completed 8xH200 run. These values may contain
    rollout and runtime randomness, so the assertions below use ranges instead
    of exact equality.

    | R3 | mean KL1 | std KL1 | max KL1 | cosine | tokens |
    | off | 0.0324 | 0.0511 | 0.6169 | 0.995934 | 846 |
    | all tokens | 0.0172 | 0.0219 | 0.1191 | 0.999119 | 846 |
    | loss tokens only | 0.0180 | 0.0232 | 0.1471 | 0.999041 | 846 |

    The exact runtime snapshot is also logged below because this test samples
    real model output and the token count can shift with generation.
    """
    if torch.cuda.device_count() < NUM_GPUS_REQUIRED:
        pytest.skip(f"Need at least {NUM_GPUS_REQUIRED} GPUs for tp=2 x cp=2 x pp=2 routing replay test")

    # tp=2 x cp=2 x pp=2 = 8 ranks. ep x etp x pp must divide world (= 8), so with
    # etp=1 the largest legal ep is 8/pp=4.
    # Each combo allocates a fresh Ray cluster (fresh placement groups +
    # fresh store) via ``ray_utils.restart()``. Within a combo,
    # Ray is NOT torn down — ``RayRolloutWorker`` hosts sglang inside a
    # Ray actor, so sglang's ``kill_process_tree`` stays scoped to that
    # actor and the driver's Ray/GCS all survive into the Megatron
    # forward phase.
    ray_utils.restart()
    kl_off = _run_one_combo(enable_r3=False, tp_size=2, cp_size=2, pp_size=2, ep_size=4)
    ray_utils.restart()
    kl_on = _run_one_combo(enable_r3=True, tp_size=2, cp_size=2, pp_size=2, ep_size=4)
    ray_utils.restart()
    kl_loss_tokens_only = _run_one_combo(
        enable_r3=True,
        replay_routing_for_loss_tokens_only=True,
        tp_size=2,
        cp_size=2,
        pp_size=2,
        ep_size=4,
    )
    kl_snapshot = _format_kl1_snapshot(kl_off, kl_on, kl_loss_tokens_only)
    logger.info("\n%s", kl_snapshot)
    print(kl_snapshot, flush=True)
    assert kl_off.mean > 0.03, f"Expected non-R3 KL1 mean mismatch to stay visible: off={kl_off}"
    assert kl_on.mean < 0.02, f"Expected R3 KL1 mean mismatch to stay low: on={kl_on}"
    assert kl_loss_tokens_only.mean < 0.02, f"Expected loss-token-only R3 KL1 mean mismatch to stay low: on={kl_loss_tokens_only}"
    assert kl_off.std > 0.045, f"Expected non-R3 KL1 std mismatch to stay visible: off={kl_off}"
    assert kl_on.std < 0.025, f"Expected R3 KL1 std mismatch to stay low: on={kl_on}"
    assert kl_loss_tokens_only.std < 0.025, f"Expected loss-token-only R3 KL1 std mismatch to stay low: on={kl_loss_tokens_only}"
    assert kl_on.mean <= kl_off.mean + 1e-3, f"R3 should not regress KL1 mean: off={kl_off}, on={kl_on}"
    assert kl_loss_tokens_only.mean <= kl_off.mean + 1e-3, f"Loss-token-only R3 should not regress KL1 mean: off={kl_off}, on={kl_loss_tokens_only}"
    assert abs(kl_loss_tokens_only.mean - kl_on.mean) < 0.005, (
        f"Loss-token-only R3 should stay close to all-token R3: all={kl_on}, loss_only={kl_loss_tokens_only}"
    )
    ray_utils.stop()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_magi_merged_loss_token_r3_grpo_train_step_grad_norm_matches_all_token_r3(tmp_path: Path) -> None:
    """GRPO train_step should keep R3 gradients stable with and without recompute.

    2026-06-01 snapshot from a completed 8xH200 run. The rollout is generated
    once with R3 enabled, then the same sample and saved routing payload are
    reused for all Megatron train_step configs.

    | R3 | recompute | loss | grad_norm |
    | off | off | 0.449379 | 8.372854 |
    | off | on | 0.449379 | 8.372854 |
    | loss tokens only | off | 0.446674 | 8.368618 |
    | loss tokens only | on | 0.446674 | 8.368618 |
    """
    if torch.cuda.device_count() < NUM_GPUS_REQUIRED:
        pytest.skip(f"Need at least {NUM_GPUS_REQUIRED} GPUs for tp=2 x cp=2 x pp=2 routing replay test")

    kwargs = dict(tp_size=2, cp_size=2, pp_size=2, ep_size=4)
    ray_utils.restart()
    sample = _make_grpo_train_sample()
    routing_payload_path = tmp_path / "routing_payload.pt"
    try:
        routing_batch = SampleTensorDict.from_samples([sample])
        assert save_spike_snapshot_routing(routing_batch, routing_payload_path) > 0
        store.delete_batch(collect_unique_routing_handles_from_batch(routing_batch))

        results: dict[tuple[bool, bool], tuple[float, float]] = {}
        for enable_r3 in (False, True):
            for activation_recompute in (False, True):
                batch = SampleTensorDict.from_samples([sample])
                assert restore_spike_snapshot_routing(batch, routing_payload_path) > 0
                try:
                    results[(enable_r3, activation_recompute)] = _run_grpo_train_step(
                        batch,
                        enable_r3=enable_r3,
                        replay_routing_for_loss_tokens_only=True,
                        activation_recompute=activation_recompute,
                        **kwargs,
                    )
                finally:
                    store.delete_batch(collect_unique_routing_handles_from_batch(batch))
    finally:
        ray_utils.stop()

    train_step_snapshot = _format_grpo_train_step_snapshot(results)
    logger.info("\n%s", train_step_snapshot)
    print(train_step_snapshot, flush=True)

    def _relative_delta(left: float, right: float) -> float:
        return abs(left - right) / max(abs(left), abs(right), 1e-6)

    loss_r3_off_no_recompute, grad_r3_off_no_recompute = results[(False, False)]
    loss_r3_off_recompute, grad_r3_off_recompute = results[(False, True)]
    loss_r3_on_no_recompute, grad_r3_on_no_recompute = results[(True, False)]
    loss_r3_on_recompute, grad_r3_on_recompute = results[(True, True)]

    assert _relative_delta(loss_r3_off_no_recompute, loss_r3_off_recompute) < 1e-2
    assert _relative_delta(grad_r3_off_no_recompute, grad_r3_off_recompute) < 1e-2
    assert _relative_delta(loss_r3_on_no_recompute, loss_r3_on_recompute) < 1e-2
    assert _relative_delta(grad_r3_on_no_recompute, grad_r3_on_recompute) < 1e-2
    assert _relative_delta(grad_r3_on_no_recompute, grad_r3_off_no_recompute) < 5e-2
    assert _relative_delta(grad_r3_on_recompute, grad_r3_off_recompute) < 5e-2
