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
from collections.abc import Callable

import chex
from jax import numpy as jnp

from spydr.shape import Shape

MeanFunction = Callable[[jnp.ndarray], jnp.ndarray]


def mean_function_shapes(feature_shape: Shape, mean_function: MeanFunction) -> MeanFunction:
    @functools.wraps(mean_function)
    def checked_mean_function(features: jnp.ndarray) -> jnp.ndarray:
        chex.assert_shape(features, [None] + feature_shape)
        targets = mean_function(features)
        chex.assert_shape(targets, [len(features)])
        return targets

    return checked_mean_function


def zero(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros([len(x)])
