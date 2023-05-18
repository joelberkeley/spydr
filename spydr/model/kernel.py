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
import functools
from typing import Callable

import chex
import jax.numpy as jnp

from spydr.shape import Shape

Kernel = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def kernel_shapes(feature_shapes: Shape, kernel: Kernel) -> Kernel:
    @functools.wraps(kernel)
    def checked_kernel(x: jnp.ndarray, x_: jnp.ndarray) -> jnp.ndarray:
        chex.assert_shape([x, x_], [None] + list(feature_shapes))
        res = kernel(x, x_)
        chex.assert_shape(res, [len(x), len(x_)])
        return res

    return checked_kernel


def rbf(length_scale: jnp.ndarray) -> Kernel:
    chex.assert_rank(length_scale, 0)

    def kernel(x: jnp.ndarray, x_: jnp.ndarray) -> jnp.ndarray:
        l2_norm = jnp.sum((x[:, None] - x_[None]) ** 2, axis=2)
        return jnp.exp(- l2_norm / (2 * length_scale ** 2))

    return kernel


def matern52(amplitude: jnp.ndarray, length_scale: jnp.ndarray) -> Kernel:
    chex.assert_rank(amplitude, 0)
    chex.assert_rank(length_scale, 0)

    def kernel(x: jnp.ndarray, x_: jnp.ndarray) -> jnp.ndarray:
        l2_norm = jnp.sum((x[:, None] - x_[None]) ** 2, axis=2)
        d = jnp.sqrt(5) * l2_norm / length_scale
        return amplitude ** 2 * (1 + d + d ** 2 / 3) * jnp.exp(- d)

    return kernel
