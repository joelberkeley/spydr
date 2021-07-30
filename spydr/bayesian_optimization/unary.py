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
from collections.abc import Callable
from typing import TypeVar

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
T = TypeVar("T")


def unit(a: A) -> Callable[[T], A]:
    return lambda _: a


def map(f: Callable[[B], C], unary: Callable[[T], B]) -> Callable[[T], C]:
    return lambda t: f(unary(t))


def apply(unary_f: Callable[[T], Callable[[A], B]], unary_x: Callable[[T], A]) -> Callable[[T], B]:
    return lambda t: unary_f(t)(unary_x(t))


def bind(mk_unary: Callable[[A], Callable[[T], B]], unary: Callable[[T], A]) -> Callable[[T], B]:
    return lambda t: mk_unary(unary(t))(t)
