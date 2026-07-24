from __future__ import annotations

import numpy as np
import torch

from axrl.configs import IGNORE_INDEX
from axrl.data import array_utils
from axrl.data.sample import (
    Sample,
    SampleTensorDict,
    _concat_sample_tensor_dicts,
    pad_sample_tensor_dict_to_multiple,
    samples_from_tensor_dict,
    select_sample_tensor_dict_rows,
)
from axrl.utils.megatron.prefix_tree import merge_trajectory_samples, unpack_tensor_from_merged


def _sample(
    input_ids: list[int],
    *,
    trainable_start: int,
    old_value_base: float,
    return_base: float,
) -> Sample:
    n = len(input_ids)
    loss_mask = [False] * n
    for idx in range(trainable_start, n - 1):
        loss_mask[idx] = True
    return Sample(
        input_ids=array_utils.as_i32(input_ids),
        labels=array_utils.as_i32([*input_ids[1:], IGNORE_INDEX]),
        loss_mask=array_utils.as_bool(loss_mask),
        attention_mask=array_utils.as_bool([True] * n),
        position_ids=array_utils.as_i32(list(range(n))),
        reward=1.0,
        reward_baseline=0.0,
        advantage=array_utils.as_f32([float(idx) if keep else 0.0 for idx, keep in enumerate(loss_mask)]),
        rollout_logprobs=array_utils.as_f32([0.0] * n),
        old_values=array_utils.as_f32([old_value_base + idx for idx in range(n)]),
        returns=array_utils.as_f32([return_base + idx for idx in range(n)]),
    )


def test_sample_tensor_dict_preserves_old_values_and_returns() -> None:
    samples = [
        _sample([1, 2, 3], trainable_start=1, old_value_base=10.0, return_base=20.0),
        _sample([4, 5], trainable_start=1, old_value_base=30.0, return_base=40.0),
    ]

    td = SampleTensorDict.from_samples(samples, max_length=4)

    assert td["old_values"].dtype == torch.float32
    assert td["returns"].dtype == torch.float32
    torch.testing.assert_close(td["old_values"][0], torch.tensor([10.0, 11.0, 12.0, 0.0]))
    torch.testing.assert_close(td["returns"][1], torch.tensor([40.0, 41.0, 0.0, 0.0]))

    restored = samples_from_tensor_dict(td)
    assert restored[0].old_values is not None
    assert restored[0].returns is not None
    assert restored[1].returns is not None
    np.testing.assert_allclose(restored[0].old_values, np.array([10.0, 11.0, 12.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(restored[1].returns, np.array([40.0, 41.0, 0.0, 0.0], dtype=np.float32))


def test_padding_concat_and_row_select_preserve_ppo_value_fields() -> None:
    td_a = SampleTensorDict.from_samples([_sample([1, 2, 3], trainable_start=1, old_value_base=1.0, return_base=2.0)], max_length=4)
    td_b = SampleTensorDict.from_samples([_sample([4, 5], trainable_start=1, old_value_base=3.0, return_base=4.0)], max_length=4)

    concatenated = _concat_sample_tensor_dicts([td_a, td_b])
    selected = select_sample_tensor_dict_rows(concatenated, [1, 0])

    torch.testing.assert_close(selected["old_values"][0], td_b["old_values"][0])
    torch.testing.assert_close(selected["returns"][1], td_a["returns"][0])

    padded, original_len = pad_sample_tensor_dict_to_multiple(concatenated, 4, padding_sample_length=2)
    assert original_len == 2
    assert padded["old_values"].shape[0] == 4
    assert padded["returns"].shape[0] == 4
    torch.testing.assert_close(padded["old_values"][2:], torch.zeros_like(padded["old_values"][2:]))
    torch.testing.assert_close(padded["returns"][2:], torch.zeros_like(padded["returns"][2:]))


def test_merge_trajectory_samples_scatters_old_values_and_returns_on_trainable_slots() -> None:
    s0 = _sample([1, 2, 10, 11], trainable_start=2, old_value_base=10.0, return_base=20.0)
    s1 = _sample([1, 2, 30, 31, 32], trainable_start=2, old_value_base=30.0, return_base=40.0)

    merged = merge_trajectory_samples([s0, s1], align_size=1)

    assert merged.merge_info is not None
    assert merged.old_values is not None
    assert merged.returns is not None
    old_values_by_path = unpack_tensor_from_merged(torch.from_numpy(merged.old_values), merged.merge_info)
    returns_by_path = unpack_tensor_from_merged(torch.from_numpy(merged.returns), merged.merge_info)

    for sample, old_values, returns in zip([s0, s1], old_values_by_path, returns_by_path, strict=True):
        assert sample.old_values is not None
        assert sample.returns is not None
        for idx, is_trainable in enumerate(sample.loss_mask.tolist()):
            if is_trainable:
                assert old_values[idx] == float(sample.old_values[idx])
                assert returns[idx] == float(sample.returns[idx])
            else:
                assert old_values[idx] == 0.0
                assert returns[idx] == 0.0
