# Copyright 2021 Joel Berkeley
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
from __future__ import annotations

from typing import Callable, Generic, TypeVar, final

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
AA = TypeVar("AA")
BB = TypeVar("BB")
CC = TypeVar("CC")


@final
class Binary(Generic[A, B, C]):
    def __init__(self, f: Callable[[A, B], C]):
        self._f = f

    def map(self, f: Callable[[C], CC]) -> Binary[A, B, CC]:
        return Binary(lambda a, b: f(self(a, b)))

    def apply(self, f: Binary[A, B, Callable[[C], CC]]) -> Binary[A, B, CC]:
        return Binary(lambda a, b: f(a, b)(self(a, b)))

    def bind(self, other: Callable[[C], Binary[A, B, CC]]) -> Binary[A, B, CC]:
        return Binary(lambda a, b: other(self(a, b))(a, b))

    def __call__(self, a: A, b: B) -> C:
        return self._f(a, b)

    def __lshift__(self, f: Callable[[AA], A]) -> Binary[AA, B, C]:
        return Binary(lambda aa, b: self(f(aa), b))

    def __rshift__(self, f: Callable[[BB], B]) -> Binary[A, BB, C]:
        return Binary(lambda a, bb: self(a, f(bb)))


def unit(c: C) -> Binary[A, B, C]:
    return Binary(lambda *_: c)
