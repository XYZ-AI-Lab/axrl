from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_i32(values: ArrayLike) -> NDArray[np.int32]:
    return np.ascontiguousarray(values, dtype=np.int32)


def as_f32(values: ArrayLike) -> NDArray[np.float32]:
    return np.ascontiguousarray(values, dtype=np.float32)


def as_bool(values: ArrayLike) -> NDArray[np.bool_]:
    return np.ascontiguousarray(values, dtype=np.bool_)


def optional_as_f32(values: ArrayLike | None) -> NDArray[np.float32] | None:
    return None if values is None else as_f32(values)


def optional_as_i32(values: ArrayLike | None) -> NDArray[np.int32] | None:
    return None if values is None else as_i32(values)


def to_int_list(values: NDArray[Any] | Iterable[int]) -> list[int]:
    if isinstance(values, np.ndarray):
        return values.tolist()
    return [int(value) for value in values]


def to_float_list(values: NDArray[Any] | Iterable[float]) -> list[float]:
    if isinstance(values, np.ndarray):
        return values.tolist()
    return [float(value) for value in values]
