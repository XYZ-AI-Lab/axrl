import logging
from pathlib import Path
from typing import override

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from axrl.data import SampleTensorDict
from axrl.utils import gpu_utils
from axrl.utils.gpu_utils import GpuUsageInfo, GpuUsageTracker
from axrl.worker.trainer_worker import TrainerWorker

logger = logging.getLogger(__name__)


class HFWorker(TrainerWorker):
    """Worker for HuggingFace model training tasks, mainly to test consistency of other implementations.."""

    def __init__(self, model_path: Path) -> None:
        super().__init__()
        self.model_path = model_path

    @override
    def initialize(self) -> None:
        # SDPA bypasses transformers/integrations/flash_attention.py, which in
        # transformers 5.6.0 unconditionally calls `s_aux.to(query.dtype)` and
        # crashes for models without a learnable attention sink (s_aux=None).
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).cuda()  # type: ignore
        super().initialize()

    @override
    def compute_logprobs(self, samples: SampleTensorDict, batch_size: int) -> tuple[Tensor, list[GpuUsageInfo]]:
        """Reference single-GPU logprobs. Iterates ``samples`` in chunks of ``batch_size``."""
        self.model.eval()
        num_samples = len(samples)
        attention_mask_full: Tensor = samples["attention_mask"]
        seq_lengths: list[int] = attention_mask_full.sum(dim=1).tolist()
        logprobs: list[torch.Tensor | None] = [None] * num_samples

        with GpuUsageTracker() as usage_tracker, torch.no_grad():
            for start in tqdm(range(0, num_samples, batch_size), desc="Computing logprobs"):
                end = min(start + batch_size, num_samples)
                input_ids = samples["input_ids"][start:end].cuda(non_blocking=True)
                attention_mask = samples["attention_mask"][start:end].cuda(non_blocking=True)
                labels = samples["labels"][start:end].cuda(non_blocking=True)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits: torch.Tensor = outputs.logits  # (B, S, V)
                all_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [B, S, V]
                safe_labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
                log_probs = all_log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, S]
                assert log_probs.shape == input_ids.shape
                batch_seq_length = attention_mask.sum(dim=1).tolist()
                for i, pos in enumerate(range(start, end)):
                    assert seq_lengths[pos] == batch_seq_length[i]
                    logprobs[pos] = log_probs[i].cpu()

        gpu_usage_info = usage_tracker.usage_info
        assert gpu_usage_info is not None
        results: list[torch.Tensor] = [x for x in logprobs if x is not None]
        assert len(logprobs) == len(results)
        tensor = torch.stack(results, dim=0)
        return tensor, [gpu_usage_info]

    @override
    def shutdown(self) -> None:
        del self.model
        gpu_utils.clear_cache()
        super().shutdown()
