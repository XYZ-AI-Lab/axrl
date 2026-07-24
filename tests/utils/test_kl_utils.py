import pytest
import torch

from axrl.utils.kl_utils import compare_logprobs, kl_divergence


def test_kl_divergence_uses_current_over_reference_log_ratio() -> None:
    logprobs = torch.tensor([-1.0, -2.5, -0.3])
    logprobs_ref = torch.tensor([-1.2, -2.0, -0.1])
    log_ratio = logprobs - logprobs_ref

    torch.testing.assert_close(kl_divergence(logprobs, logprobs_ref, "k1"), log_ratio)
    torch.testing.assert_close(kl_divergence(logprobs, logprobs_ref, "k2"), log_ratio.square() / 2.0)

    ref_to_current_log_ratio = -log_ratio
    expected_k3 = ref_to_current_log_ratio.exp() - 1 - ref_to_current_log_ratio
    torch.testing.assert_close(kl_divergence(logprobs, logprobs_ref, "k3"), expected_k3)


def test_compare_logprobs_uses_shared_k3_estimator() -> None:
    loss_mask = torch.tensor([[True, True, False]])
    base_logprobs = torch.tensor([[-2.0, -1.0, -4.0]])
    test_logprobs = torch.tensor([[-1.5, -1.25, -4.0]])

    result = compare_logprobs(loss_mask, base_logprobs, test_logprobs)

    log_ratio = base_logprobs[0, :2] - test_logprobs[0, :2]
    expected_k3 = (log_ratio.exp() - log_ratio - 1).mean().item()
    assert result.k3 == pytest.approx(expected_k3)
