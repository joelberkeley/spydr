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
from typing import Callable, TypeVar

import jax.numpy as jnp
from jax.scipy.optimize import minimize

T = TypeVar("T")

Optimizer = Callable[[Callable[[T], jnp.ndarray]], T]


def bfgs(initial_guess: jnp.ndarray) -> Optimizer[jnp.ndarray]:
    def optimizer(f: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        res = minimize(lambda x: -f(x), initial_guess, method="BFGS")#, options={"maxiter": 100000})

        if not res.success:
            print("WARN: bfgs failed to converge with error:", res.status)

        return res.x

    return optimizer
