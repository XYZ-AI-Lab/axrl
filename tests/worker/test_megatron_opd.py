from __future__ import annotations

from typing import Any

import pytest
import torch
from tensordict import TensorDict

from axrl.configs import GrpoTrainerConfig, ModelConfig, OPDConfig, RolloutWorkerConfig
from axrl.trainer.grpo_trainer import GrpoTrainer
from axrl.worker.megatron_worker import MegatronWorker


def _worker_with_trainer(config: GrpoTrainerConfig) -> MegatronWorker:
    worker = MegatronWorker.__new__(MegatronWorker)
    worker.trainer = GrpoTrainer(config)
    return worker


def test_megatron_teacher_logprob_gate() -> None:
    disabled_worker = _worker_with_trainer(GrpoTrainerConfig())
    teacher_model = ModelConfig(name="teacher")
    sglang_worker = _worker_with_trainer(
        GrpoTrainerConfig(
            opd=OPDConfig(
                enabled=True,
                backend="sglang",
                teacher_model=teacher_model,
                sglang_worker=RolloutWorkerConfig(model=teacher_model),
                sglang_port=31080,
            )
        )
    )
    enabled_worker = _worker_with_trainer(
        GrpoTrainerConfig(
            opd=OPDConfig(
                enabled=True,
                backend="megatron",
                teacher_model=ModelConfig(name="teacher"),
            )
        )
    )

    assert not disabled_worker._should_compute_megatron_teacher_logprobs()
    assert not sglang_worker._should_compute_megatron_teacher_logprobs()
    assert enabled_worker._should_compute_megatron_teacher_logprobs()


def test_megatron_teacher_logprob_update_is_noop_when_gate_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _worker_with_trainer(GrpoTrainerConfig())
    calls: list[str] = []

    def record_call(name: str) -> None:
        calls.append(name)

    monkeypatch.setattr(worker, "copy_weights_to_cpu", record_call)
    monkeypatch.setattr(worker, "apply_weights_from_cpu", record_call)
    batches = [TensorDict({"input_ids": torch.ones((1, 4), dtype=torch.long)}, batch_size=1)]

    updated = worker._update_teacher_logprobs_for_batches(batches)

    assert updated is batches
    assert "teacher_logprobs" not in batches[0]
    assert calls == []


def test_megatron_teacher_logprob_update_uses_named_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _worker_with_trainer(
        GrpoTrainerConfig(
            opd=OPDConfig(
                enabled=True,
                backend="megatron",
                teacher_model=ModelConfig(name="teacher"),
                teacher_weight_name="teacher_snapshot",
            )
        )
    )
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(worker, "copy_weights_to_cpu", lambda name: calls.append(("copy", name)))
    monkeypatch.setattr(worker, "apply_weights_from_cpu", lambda name: calls.append(("apply", name)))

    def compute_logprobs(batches: list[Any]) -> tuple[list[torch.Tensor], object]:
        calls.append(("compute", "logprobs"))
        return [torch.full_like(batch["input_ids"], -3.0, dtype=torch.float32) for batch in batches], object()

    monkeypatch.setattr(worker, "_compute_logprobs_from_local_batches", compute_logprobs)
    batches = [TensorDict({"input_ids": torch.ones((1, 4), dtype=torch.long)}, batch_size=1)]

    updated = worker._update_teacher_logprobs_for_batches(batches)

    assert updated is batches
    assert calls == [
        ("copy", "cur_weights"),
        ("apply", "teacher_snapshot"),
        ("compute", "logprobs"),
        ("apply", "cur_weights"),
    ]
    assert torch.allclose(batches[0]["teacher_logprobs"], torch.full((1, 4), -3.0))


def test_megatron_teacher_logprob_update_restores_current_weights_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _worker_with_trainer(
        GrpoTrainerConfig(
            opd=OPDConfig(
                enabled=True,
                backend="megatron",
                teacher_model=ModelConfig(name="teacher"),
                teacher_weight_name="teacher_snapshot",
            )
        )
    )
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(worker, "copy_weights_to_cpu", lambda name: calls.append(("copy", name)))
    monkeypatch.setattr(worker, "apply_weights_from_cpu", lambda name: calls.append(("apply", name)))

    def fail_compute_logprobs(_batches: list[Any]) -> tuple[list[torch.Tensor], object]:
        calls.append(("compute", "logprobs"))
        raise RuntimeError("teacher failed")

    monkeypatch.setattr(worker, "_compute_logprobs_from_local_batches", fail_compute_logprobs)
    batches = [TensorDict({"input_ids": torch.ones((1, 4), dtype=torch.long)}, batch_size=1)]

    with pytest.raises(RuntimeError, match="teacher failed"):
        worker._update_teacher_logprobs_for_batches(batches)

    assert calls == [
        ("copy", "cur_weights"),
        ("apply", "teacher_snapshot"),
        ("compute", "logprobs"),
        ("apply", "cur_weights"),
    ]
    assert "teacher_logprobs" not in batches[0]


def test_logprob_lookup_accepts_sparse_indices_from_model_parallel_gather() -> None:
    worker = _worker_with_trainer(GrpoTrainerConfig())
    outputs = [
        TensorDict(
            {
                "index": torch.tensor([32, 0]),
                "log_prob": torch.tensor([[32.0, 32.1, 32.2], [0.0, 0.1, 0.2]]),
            },
            batch_size=2,
        ),
        TensorDict(
            {
                "index": torch.tensor([16]),
                "log_prob": torch.tensor([[16.0, 16.1, 16.2]]),
            },
            batch_size=1,
        ),
    ]
    index_to_pos, logprobs_by_pos = worker._cat_logprobs_by_index(outputs)
    batch = TensorDict(
        {
            "input_ids": torch.ones((4, 3), dtype=torch.long),
            "index": torch.tensor([0, 16, -1, 32]),
        },
        batch_size=4,
    )

    logprobs = worker._logprobs_for_batch(batch, index_to_pos, logprobs_by_pos)

    expected = torch.tensor(
        [
            [0.0, 0.1, 0.2],
            [16.0, 16.1, 16.2],
            [0.0, 0.0, 0.0],
            [32.0, 32.1, 32.2],
        ]
    )
    assert torch.allclose(logprobs, expected)
