from collections.abc import Iterator
from typing import TypeVar

T = TypeVar("T")


def take(n: int, xs: Iterator[T]) -> list[T]:
    if n < 0:
        raise ValueError(f"n must be positive or zero, got {n}")

    return [next(xs) for _ in range(n)]
