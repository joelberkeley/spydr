from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable

S = TypeVar("S")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


@dataclass(frozen=True, eq=False)
class Reader(Generic[T, U]):
    run: Callable[[T], U]

    def map(self, f: Callable[[U], V]) -> Reader[T, V]:
        return Reader(lambda t: f(self.run(t)))

    def contramap(self, f: Callable[[S], T]) -> Reader[S, V]:
        return Reader(lambda s: self.run(f(s)))

    def apply(self, other: Reader[T, Callable[[U], V]]) -> Reader[T, V]:
        return Reader(lambda t: other.run(t)(self.run(t)))

    def bind(self, f: Callable[[U], Reader[T, V]]) -> Reader[T, V]:
        return Reader(lambda t: f(self.run(t)).run(t))
