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

from collections.abc import Iterator
from typing import Callable, TypeVar

import jax.numpy as jnp
from spydr.bayesian_optimization.binary import Binary

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def loop(
        tactic: Binary[T, U, jnp.ndarray],
        observer: Callable[[jnp.ndarray, T, U], tuple[T, U]],
        initial: tuple[T, U],
) -> Iterator[tuple[T, U]]:
    yield initial
    current = observer(tactic(*initial), *initial)
    yield from loop(tactic, observer, current)
