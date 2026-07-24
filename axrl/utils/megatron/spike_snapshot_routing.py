from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from pathlib import Path

    from tensordict import TensorDict

    from axrl.utils.tensor_store import TensorHandle


def collect_unique_routing_handles_from_batch(batch: TensorDict) -> list[TensorHandle]:
    """Collect effective routing handles from a training batch in first-seen order."""
    if "routing_handles_per_path" not in batch.keys():  # noqa: SIM118
        return []
    raw_rows = _unwrap_non_tensor(batch.get_non_tensor("routing_handles_per_path", default=None))
    if raw_rows is None:
        return []

    seen: set[TensorHandle] = set()
    ordered: list[TensorHandle] = []
    for row in list(raw_rows):
        unwrapped_row = _unwrap_non_tensor(row)
        if unwrapped_row is None:
            continue
        for path_handles in list(unwrapped_row):
            unwrapped_path = _unwrap_non_tensor(path_handles)
            if unwrapped_path is None:
                continue
            for raw_handle in list(unwrapped_path):
                handle = _unwrap_non_tensor(raw_handle)
                if handle not in seen:
                    seen.add(handle)
                    ordered.append(handle)
    return ordered


def save_spike_snapshot_routing(batch: TensorDict, path: Path) -> int:
    """Save the routing payloads referenced by ``batch`` to ``path``.

    The spike batch itself only contains ``TensorHandle``s. Those handles point
    at Ray object-store data that may be gone by the time the spike is replayed,
    so the snapshot must persist the payload bytes too.
    """
    from axrl.utils import tensor_store as store

    handles = collect_unique_routing_handles_from_batch(batch)
    if not handles:
        return 0

    fetched = store.get_batch(handles)
    payloads = [np.asarray(fetched[handle]).copy() for handle in handles]
    torch.save(payloads, path)
    return len(payloads)


def restore_spike_snapshot_routing(batch: TensorDict, path: Path) -> int:
    """Restore saved routing payloads and rewrite ``batch`` to their new handles."""
    from axrl.utils import tensor_store as store

    handles = collect_unique_routing_handles_from_batch(batch)
    if not handles:
        return 0

    payloads = torch.load(path, weights_only=False)
    if not isinstance(payloads, list):
        msg = f"Malformed spike routing payload file: {path}"
        raise TypeError(msg)
    if len(payloads) != len(handles):
        msg = f"Spike routing payload count mismatch: batch has {len(handles)} handles, file has {len(payloads)} payloads"
        raise RuntimeError(msg)

    replacements = {old: store.put(np.asarray(payload).copy()) for old, payload in zip(handles, payloads, strict=True)}
    _replace_routing_handles_in_batch(batch, replacements)
    return len(replacements)


def _replace_routing_handles_in_batch(batch: TensorDict, replacements: dict[TensorHandle, TensorHandle]) -> None:
    raw_rows = _unwrap_non_tensor(batch.get_non_tensor("routing_handles_per_path", default=None))
    if raw_rows is None:
        return

    restored_rows: list[list[list[TensorHandle]] | None] = []
    for row in list(raw_rows):
        unwrapped_row = _unwrap_non_tensor(row)
        if unwrapped_row is None:
            restored_rows.append(None)
            continue
        restored_paths: list[list[TensorHandle]] = []
        for path_handles in list(unwrapped_row):
            unwrapped_path = _unwrap_non_tensor(path_handles)
            if unwrapped_path is None:
                restored_paths.append([])
                continue
            restored_paths.append([replacements[_unwrap_non_tensor(handle)] for handle in list(unwrapped_path)])
        restored_rows.append(restored_paths)
    batch.set_non_tensor("routing_handles_per_path", restored_rows)


def _unwrap_non_tensor(value: Any) -> Any:
    if hasattr(value, "data") and not isinstance(value, (list, tuple)):
        try:
            return value.data
        except AttributeError:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, type(None))):
        try:
            return value.tolist()
        except (AttributeError, RuntimeError):
            pass
    return value
