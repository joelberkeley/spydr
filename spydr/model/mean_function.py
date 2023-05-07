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

from jax import numpy as jnp

from spydr.shape import assert_shape

MeanFunction = Callable[[jnp.ndarray], jnp.ndarray]


def zero(x: jnp.ndarray) -> jnp.ndarray:
    assert_shape(x, (None, 1))
    res = jnp.zeros([len(x)])
    assert_shape(res, (len(x),))
    return res
