from __future__ import annotations

import subprocess
import time
from typing import Any

import pytest
import torch
from transformers import AutoProcessor

from axrl.configs import ModelConfig, RolloutWorkerConfig, SamplingConfig
from axrl.data.conversation import Conversation, Message
from axrl.runner import CgroupRunner

TEST_MODEL_NAME = "Qwen/Qwen3-0.6B"
TEST_SEQ_LENGTH = 2048
TEST_TP_SIZE = 1
TEST_GPU_MEMORY_UTILIZATION = 0.45

SGLANG_PROCESS_PATTERNS = (
    "sglang.launch_server",
    "sglang_router",
    "RemoteSGLangWorker",
    "axrl.worker.sglang_worker",
)


def sglang_test_config(*, num_workers: int = 1) -> RolloutWorkerConfig:
    return RolloutWorkerConfig(
        model=ModelConfig(name=TEST_MODEL_NAME, seq_length=TEST_SEQ_LENGTH, trust_remote_code=True),
        sampling_config=SamplingConfig(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_total_tokens=512,
            max_new_tokens=8,
        ),
        tp_size=TEST_TP_SIZE,
        pp_size=1,
        dp_size=1,
        num_workers=num_workers,
        gpu_memory_utilization=TEST_GPU_MEMORY_UTILIZATION,
        max_running_requests=8,
        enable_metrics=False,
        log_level="warning",
    )


def skip_if_real_sglang_cannot_run(config: RolloutWorkerConfig, *, min_gpus: int | None = None) -> None:
    required_gpus = min_gpus if min_gpus is not None else config.gpus_per_worker()
    if not torch.cuda.is_available() or torch.cuda.device_count() < required_gpus:
        pytest.skip(f"Need >= {required_gpus} GPU(s), found {torch.cuda.device_count()}.")
    if not config.model.get_full_path().exists():
        pytest.skip(f"Model path does not exist: {config.model.get_full_path()}.")
    if not CgroupRunner.is_supported():
        pytest.skip("CgroupRunner is not supported on this host.")


def prompt_input_ids(config: RolloutWorkerConfig) -> list[int]:
    processor: Any = AutoProcessor.from_pretrained(config.model.get_full_path(), use_fast=True)
    conversation = Conversation(messages=[Message(role="user", content="Write one short sentence about Beijing.")])
    prompt = processor.apply_chat_template(conversation.to_dict()["messages"], add_generation_prompt=True, tokenize=False)
    return processor(text=[prompt], return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()


def standalone_server_gpus(config: RolloutWorkerConfig) -> list[int]:
    return list(range(config.tp_size))


def sglang_processes() -> list[str]:
    proc = subprocess.run(["/usr/bin/ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True)
    lines = [line.strip() for line in proc.stdout.splitlines()]
    return [line for line in lines if any(pattern in line for pattern in SGLANG_PROCESS_PATTERNS)]


def assert_no_sglang_processes(timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    leaked = sglang_processes()
    while leaked and time.monotonic() < deadline:
        time.sleep(1.0)
        leaked = sglang_processes()
    assert not leaked, "SGLang server/worker processes leaked after test:\n" + "\n".join(leaked)
