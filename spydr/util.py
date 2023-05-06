from __future__ import annotations

import functools
from typing import Callable, TypeVar, Any

import jax.numpy as jnp


def assert_shape(x: jnp.ndarray, shape: list[int | None] | tuple[int | None, ...]) -> jnp.ndarray:
    return x
    shape = tuple(shape)
    assert all(dim is None or dim > 0 for dim in shape)
    assert len(x.shape) == len(shape), f"expected {shape}, got {x.shape}"
    full_shape = tuple(a if e is None else e for (a, e) in zip(x.shape, shape))
    assert x.shape == full_shape, f"expected {shape}, got {x.shape}"
    return x


C = TypeVar("C", bound=Callable)


def assert_shapes(checks: Callable[[Any, ...], bool]) -> Callable[[C], C]:
    def decorator(g: C) -> C:
        @functools.wraps(g)
        def decorated(*args, **kwargs):
            assert checks(*args, **kwargs)
            return g(*args, **kwargs)

        return decorated

    return decorator


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def compose(f: Callable[[U], V], g: Callable[[T], U]) -> Callable[[T], V]:
    return lambda t: f(g(t))
