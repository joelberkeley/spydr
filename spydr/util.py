from __future__ import annotations
import jax.numpy as np


def assert_shape(x: np.ndarray, shape: list[int | None] | tuple[int | None, ...]) -> np.ndarray:
    shape = tuple(shape)
    assert all(dim is None or dim > 0 for dim in shape)
    assert len(x.shape) == len(shape), f"expected {shape}, got {x.shape}"
    full_shape = tuple(a if e is None else e for (a, e) in zip(x.shape, shape))
    assert x.shape == full_shape, f"expected {shape}, got {x.shape}"
    return x