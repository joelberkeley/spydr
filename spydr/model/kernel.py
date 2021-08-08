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
from typing import Callable

import jax.numpy as jnp

from spydr.util import assert_shape

Kernel = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def rbf(length_scale: jnp.ndarray) -> Kernel:
    def kernel(x: jnp.ndarray, x_: jnp.ndarray) -> jnp.ndarray:
        assert_shape(x, (None, 1))
        assert_shape(x_, (None, 1))
        l2_norm = jnp.sum((x[:, None] - x_[None]) ** 2, axis=2)
        return assert_shape(jnp.exp(- l2_norm / (2 * length_scale ** 2)), (len(x), len(x_)))

    return kernel


def matern52(amplitude: jnp.ndarray, length_scale: jnp.ndarray) -> Kernel:
    def kernel(x: jnp.ndarray, x_: jnp.ndarray) -> jnp.ndarray:
        assert_shape(x, (None, 1))
        assert_shape(x_, (None, 1))
        l2_norm = jnp.sum((x[:, None] - x_[None]) ** 2, axis=2)
        d = l2_norm / length_scale
        res = amplitude ** 2 * (1 + jnp.sqrt(5) * d + 5 * d ** 2 / 3) * jnp.exp(- jnp.sqrt(5) * d)
        return assert_shape(res, (len(x), len(x_)))

    return kernel
