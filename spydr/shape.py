from __future__ import annotations

from typing import Callable, TypeVar, TypeAlias


Shape = tuple[int | None, ...] | list[int | None]

StaticShape: TypeAlias = list[int] | tuple[int, ...]


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def compose(f: Callable[[U], V], g: Callable[[T], U]) -> Callable[[T], V]:
    return lambda t: f(g(t))
