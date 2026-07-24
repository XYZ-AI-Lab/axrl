"""Shared fixture for context-management / prefix-tree-merging tests.

Both ``test_magi_attention.py`` and ``test_rollout_trace.py`` build
the same 4-turn tool-using conversation (a query about A100/H100/H200
energy efficiency) — the canonical Hide-Tool-Result context-management
example. This module is the single source of truth; the test files
import the conversation tokens, parallel-case helpers, and worker-spawn
utilities from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from axrl.configs import MegatronWorkerConfig, ModelConfig
from axrl.data import Conversation, Message, Sample, SampleTensorDict
from axrl.data.token_trace import TokenTrace
from axrl.example.config_examples import get_megatron_trainer_config
from axrl.processor.appended_message_tokenizer import AppendedMessageTokenizer
from axrl.processor.conversation_tokenizer import ConversationTokenizer
from axrl.ray import ray_utils
from axrl.ray.ray_megatron_worker import RayMegatronWorker
from axrl.ray.resource_group import Request, ResourceGroup

if TYPE_CHECKING:
    import numpy as np
    import torch
    from numpy.typing import NDArray

# =====================================================================
# Conversation text (system + user + 4 assistant turns + 3 tool results)
# =====================================================================


_SYSTEM_PROMPT = (
    "You are an expert research assistant with access to web search "
    "tools. When the user asks a question, think carefully about what "
    "information is needed, issue tool calls to gather that information, "
    "and then synthesize a clear, well-cited answer. Always explain your "
    "reasoning before invoking a tool, and cite sources when summarizing."
)

_USER_QUERY = (
    "I'd like a detailed comparison of the energy efficiency of large "
    "language model inference across three different GPU generations "
    "(A100, H100, and H200) for a 70B-parameter model served at a "
    "batch size of 64 with 1024-token prompts. Please include tokens "
    "per second per watt, total power draw at load, and typical "
    "utilization, then tell me which option gives the best dollars-"
    "per-million-tokens under published cloud prices. Cite specific "
    "benchmarks where you can."
)


def _think_toolcall(turn: int) -> str:
    return (
        f"<think>Turn {turn}: Let me think carefully about how to "
        f"approach this benchmarking question step by step. The user "
        f"is asking for a detailed per-GPU comparison covering three "
        f"specific quantitative metrics (tokens per second per watt, "
        f"total power draw under load, and typical utilization) as "
        f"well as a price-per-million-tokens analysis using published "
        f"on-demand cloud rates. To give a rigorous answer I need "
        f"current benchmark figures from credible third-party sources, "
        f"not just vendor marketing materials, and I should cross-"
        f"reference MLPerf Inference v4.0 submissions against the "
        f"relevant vendor datasheets for power and memory-bandwidth "
        f"numbers. My plan: first perform a targeted web search for "
        f"published 70-billion-parameter inference benchmarks at the "
        f"requested batch size and prompt length on A100, H100, and "
        f"H200 hardware; then, on the next turn, narrow the search to "
        f"cloud-provider price sheets so I can compute the dollars-per-"
        f"million-tokens figure. I will track the tokens-per-watt "
        f"metric explicitly because that is the efficiency quantity "
        f"the user cares about most. If the first search returns sparse "
        f"data I will refine the query on my next turn by specifying "
        f"the exact GPU variant and quantization regime.</think>\n"
        f"Let me look that up right now so I can give you well-sourced "
        f"numbers rather than rough estimates.\n"
        f'<tool_call>{{"name": "web_search", "arguments": '
        f'{{"query": "MLPerf Inference v4.0 70B language model '
        f"NVIDIA A100 H100 H200 tokens-per-second-per-watt throughput "
        f"benchmark batch 64 prompt 1024 turn {turn}, include cloud "
        f"on-demand dollars-per-million-tokens and sustained power "
        f'draw in watts plus typical SM utilization percentages"}}}}'
        f"</tool_call>"
    )


def _tool_result(turn: int) -> str:
    return (
        f"Top result for turn {turn}: The official MLPerf Inference "
        f"v4.0 result table and the corresponding vendor whitepapers "
        f"report the following sustained figures for a 70-billion-"
        f"parameter language model served at batch size 64 with 1024-"
        f"token prompts. On NVIDIA A100 80GB the model sustains "
        f"approximately 180 tokens/second at around 350 watts under "
        f"load, yielding roughly 0.51 tokens per watt; SM utilization "
        f"averages about 62 percent during the steady-state window. "
        f"On NVIDIA H100 80GB SXM the same workload achieves about "
        f"520 tokens/second at around 600 watts, producing roughly "
        f"0.87 tokens per watt with SM utilization near 78 percent. "
        f"On NVIDIA H200 141GB — which benefits from the larger HBM3e "
        f"capacity and higher memory bandwidth — throughput reaches "
        f"about 780 tokens/second at around 700 watts, for an "
        f"approximate 1.11 tokens per watt, and SM utilization climbs "
        f"to roughly 86 percent because larger effective batches fit "
        f"in device memory. Translating those numbers to on-demand "
        f"cloud pricing, recent price sheets from major providers "
        f"imply roughly $2.10 per million tokens on A100, about $1.45 "
        f"per million tokens on H100, and approximately $1.15 per "
        f"million tokens on H200; spot-market prices are typically "
        f"30-45 percent below those on-demand rates depending on "
        f"region and availability. Multiple independent v4.0 MLPerf "
        f"Inference submissions corroborate the relative ordering."
    )


def _final_answer() -> str:
    return (
        "Here is the detailed comparison you asked for. On a 70-"
        "billion-parameter language model served at batch size 64 "
        "with 1024-token prompts, recent MLPerf Inference v4.0 "
        "submissions and vendor whitepapers give approximately the "
        "following figures: on A100 80GB the throughput is about "
        "180 tokens/second at ~350 watts, roughly 62 percent SM "
        "utilization, for ~0.51 tokens per watt; on H100 80GB SXM "
        "we see about 520 tokens/second at ~600 watts, roughly 78 "
        "percent SM utilization, for ~0.87 tokens per watt; and on "
        "H200 141GB the model reaches about 780 tokens/second at "
        "~700 watts, roughly 86 percent SM utilization, for ~1.11 "
        "tokens per watt — the best energy efficiency of the three. "
        "Translating those throughput and power numbers to dollars "
        "per million tokens at on-demand published cloud rates: A100 "
        "around $2.10, H100 around $1.45, and H200 around $1.15 per "
        "million tokens. The H200 therefore offers the best cost "
        "efficiency at load, with H100 a strong second choice if "
        "H200 capacity is constrained. Sources: MLPerf Inference "
        "v4.0 result tables and the corresponding NVIDIA datasheet "
        "power and memory-bandwidth curves. If you need a deeper "
        "break-down by quantization regime or by cloud provider, I "
        "can follow up with another targeted search."
    )


# =====================================================================
# Tokenized fixture
# =====================================================================


@dataclass(frozen=True)
class RealisticTokens:
    """Per-message token lists for the realistic 4-turn conversation."""

    prompt_tokens: NDArray[np.int32]  # [system, user] tokens
    a1: NDArray[np.int32]
    tr1: NDArray[np.int32]
    a2: NDArray[np.int32]
    tr2: NDArray[np.int32]
    a3: NDArray[np.int32]
    tr3: NDArray[np.int32]
    a4: NDArray[np.int32]
    pad_id: int


_MIN_TOKEN_PER_MESSAGE = 100


def make_realistic_tokens(max_length: int = 2048) -> RealisticTokens:
    """Tokenize the canonical realistic 4-turn tool-using conversation.

    Each per-message token sequence is asserted to exceed
    ``_MIN_TOKEN_PER_MESSAGE`` so the trie has substantive content per
    node and the realistic test bands stay meaningful.
    """
    model_config = ModelConfig(name="Qwen/Qwen3-0.6B", seq_length=max_length)

    sys_msg = Message(role="system", content=_SYSTEM_PROMPT)
    user_msg = Message(role="user", content=_USER_QUERY)
    a1_msg = Message(role="assistant", content=_think_toolcall(1))
    tr1_msg = Message(role="tool", content=_tool_result(1), tool_call_id="c1")
    a2_msg = Message(role="assistant", content=_think_toolcall(2))
    tr2_msg = Message(role="tool", content=_tool_result(2), tool_call_id="c2")
    a3_msg = Message(role="assistant", content=_think_toolcall(3))
    tr3_msg = Message(role="tool", content=_tool_result(3), tool_call_id="c3")
    a4_msg = Message(role="assistant", content=_final_answer())

    conv_prompt = Conversation(messages=[sys_msg, user_msg])
    conv_tok = ConversationTokenizer(model_config)
    prompt_tokens = conv_tok.process(conv_prompt).input_ids

    append_tok = AppendedMessageTokenizer(model_config)
    a1 = append_tok.process(a1_msg)
    tr1 = append_tok.process(tr1_msg)
    a2 = append_tok.process(a2_msg)
    tr2 = append_tok.process(tr2_msg)
    a3 = append_tok.process(a3_msg)
    tr3 = append_tok.process(tr3_msg)
    a4 = append_tok.process(a4_msg)

    for name, toks in [("a1", a1), ("tr1", tr1), ("a2", a2), ("tr2", tr2), ("a3", a3), ("tr3", tr3), ("a4", a4)]:
        assert len(toks) > _MIN_TOKEN_PER_MESSAGE, f"{name} has only {len(toks)} tokens (<{_MIN_TOKEN_PER_MESSAGE} required)"

    pad_id = getattr(append_tok._processor, "pad_token_id", None) or getattr(append_tok._processor, "eos_token_id", 0) or 0

    return RealisticTokens(
        prompt_tokens=prompt_tokens,
        a1=a1,
        tr1=tr1,
        a2=a2,
        tr2=tr2,
        a3=a3,
        tr3=tr3,
        a4=a4,
        pad_id=pad_id,
    )


# =====================================================================
# Hide-Tool-Result samples (used by test_magi_attention.py)
# =====================================================================


def make_hide_tool_result_samples(toks: RealisticTokens, max_length: int) -> list[Sample]:
    """Build 4 Hide-Tool-Result SFT samples from a tokenized fixture.

    Each later sample drops earlier tool_results that are no longer
    needed once their assistant turn has already been generated::

        sample 1: [s, u] + a1(T)
        sample 2: [s, u] + a1 + tr1 + a2(T)
        sample 3: [s, u] + a1 + a2 + tr2 + a3(T)            # tr1 dropped
        sample 4: [s, u] + a1 + a2 + a3 + tr3 + a4(T)       # tr1, tr2 dropped

    Having 4 samples makes the batch divisible by DP ∈ {1, 2, 4} which
    is required by Megatron's batch-splitter.
    """

    def _build(segments: list[tuple[NDArray[np.int32], bool]]) -> Sample:
        trace = TokenTrace()
        for idx, (seg_toks, trainable) in enumerate(segments):
            if trainable:
                token_type = "assistant"
            else:
                token_type = "init" if idx == 0 else "tool_result"
            trace.extend_tokens(seg_toks, token_type=token_type)
        return trace.to_sample(max_length=max_length, pad_token_id=toks.pad_id)

    return [
        _build([(toks.prompt_tokens, False), (toks.a1, True)]),
        _build([(toks.prompt_tokens, False), (toks.a1, False), (toks.tr1, False), (toks.a2, True)]),
        _build([(toks.prompt_tokens, False), (toks.a1, False), (toks.a2, False), (toks.tr2, False), (toks.a3, True)]),
        _build(
            [
                (toks.prompt_tokens, False),
                (toks.a1, False),
                (toks.a2, False),
                (toks.a3, False),
                (toks.tr3, False),
                (toks.a4, True),
            ],
        ),
    ]


# =====================================================================
# Parallel case dataclass + worker config helpers
# =====================================================================


@dataclass(frozen=True)
class ParallelCase:
    """One parallelism configuration for parametrizing GPU tests."""

    name: str
    tp: int = 1
    pp: int = 1
    cp: int = 1
    dp: int = 1
    deterministic: bool = False
    batch_invariant: bool = False

    def world_size(self) -> int:
        return self.tp * self.pp * self.cp * self.dp


def make_megatron_worker_config(case: ParallelCase, *, seq_length: int = 1024) -> MegatronWorkerConfig:
    """Megatron worker config for the given parallelism case.

    Determinism flags are propagated to the worker config only; the Ray
    worker's ``__init__`` calls the corresponding ``apply_*_flags()``
    inside its own process — the test driver intentionally does not set
    any global torch / env state.
    """
    model_config = ModelConfig(name="Qwen/Qwen3-0.6B", seq_length=seq_length)
    config = get_megatron_trainer_config(
        tp_size=case.tp,
        pp_size=case.pp,
        cp_size=case.cp,
        dp_size=case.dp,
        model_config=model_config,
    )
    config.global_batch_size = max(1, case.dp)
    config.train_micro_batch_size = 1
    config.eval_micro_batch_size = 1
    config.inference_only = False
    config.apply_rope_fusion = True
    config.log_every_k_steps = 1
    # batch_invariant is the heavy superset of deterministic — when bi is on,
    # deterministic is also on (so Magi's atomic-reduction kernel is replaced
    # by the ordered one, in addition to the torch global flag flips).
    if case.batch_invariant:
        config.batch_invariant_mode = True
        config.deterministic_mode = True
    elif case.deterministic:
        config.deterministic_mode = True
    else:
        config.deterministic_mode = False
        config.batch_invariant_mode = False
    return config


# =====================================================================
# Worker spawn helpers (run a single train step / compute logprobs)
# =====================================================================


def train_step_loss_gn(config: MegatronWorkerConfig, samples: list[Sample], world_size: int = 1) -> tuple[float, float]:
    """Spawn a Ray Megatron worker, run one ``train`` step, return ``(loss, grad_norm)``.

    Each input sample is assigned its own ``trajectory_id`` so the new
    trajectory-grouped iterator interprets ``config.global_batch_size`` as the
    number of trajectories per gradient update.
    """
    samples = [Sample(**s.__dict__) for s in samples]
    for trajectory_id, sample in enumerate(samples):
        sample.trajectory_id = trajectory_id
    ray_utils.restart()
    rg = ResourceGroup([Request(cpu=1, gpu=world_size)])
    worker = RayMegatronWorker(config=config, resource_group=rg)
    worker.initialize()
    tensor_dict = SampleTensorDict.from_samples(samples)
    _step, metrics = worker.train(samples=tensor_dict, global_step=0, data_shuffle_seed=0, compute_logprobs=False)
    worker.shutdown()
    ray_utils.stop()
    return float(metrics["actor_train/loss"]), float(metrics["actor_train/grad_norm"])


def compute_logprobs_via_worker(config: MegatronWorkerConfig, samples: list[Sample], rg: ResourceGroup) -> torch.Tensor:
    """Spawn a Ray Megatron worker via ``rg`` and return ``compute_logprobs`` output."""
    samples = [Sample(**s.__dict__) for s in samples]
    for trajectory_id, sample in enumerate(samples):
        sample.trajectory_id = trajectory_id
    worker = RayMegatronWorker(config=config, resource_group=rg)
    worker.initialize()
    tensor_dict = SampleTensorDict.from_samples(samples)
    logprobs, _ = worker.compute_logprobs(samples=tensor_dict, batch_size=config.global_batch_size)
    worker.shutdown()
    return logprobs.cpu()


__all__ = [
    "ParallelCase",
    "RealisticTokens",
    "compute_logprobs_via_worker",
    "make_hide_tool_result_samples",
    "make_megatron_worker_config",
    "make_realistic_tokens",
    "train_step_loss_gn",
]
