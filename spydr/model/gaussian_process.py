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

from collections import Callable
from dataclasses import dataclass

from jax.scipy.linalg import solve_triangular
from jax import numpy as jnp
from mypy.nodes import TypeVar

from spydr.distribution import Gaussian
from spydr.model import ProbabilisticModel, Data, Trainable
from spydr.model.kernel import Kernel
from spydr.model.mean_function import MeanFunction
from spydr.optimize import Optimizer
from spydr.util import assert_shape


@dataclass
class GaussianProcess:
    mean_function: MeanFunction
    kernel: Kernel


def marginalise(gp: GaussianProcess, x: jnp.ndarray) -> Gaussian:
    assert_shape(x, (None, 1))
    return Gaussian(gp.mean_function(x), gp.kernel(x, x))


def _posterior(prior: GaussianProcess, noise: jnp.ndarray, training_data: Data) -> Gaussiajnprocess:
    x_train, y_train = training_data
    assert_shape(x_train, (None, 1))
    assert_shape(y_train, (len(x_train),))

    l = jnp.linalg.cholesky(prior.kernel(x_train, x_train) + noise * jnp.eye(len(x_train)))
    alpha = solve_triangular(l.transpose(), solve_triangular(l, y_train))

    def posterior_meanf(x: jnp.ndarray) -> jnp.ndarray:
        assert_shape(x, (None, 1))
        res = prior.mean_function(x) + prior.kernel(x, x_train) @ alpha
        return assert_shape(res, (len(x),))

    def posterior_kernel(x: jnp.ndarray, x_: jnp.ndarray) -> jnp.ndarray:
        assert_shape(x, (None, 1))
        assert_shape(x_, (None, 1))
        res = (
            prior.kernel(x, x_)
            - (solve_triangular(l, prior.kernel(x_train, x))).transpose()
            @ solve_triangular(l, prior.kernel(x_train, x_))
        )
        return assert_shape(res, (len(x), len(x_)))

    return GaussianProcess(posterior_meanf, posterior_kernel)


def _log_marginal_likelihood(
        gp: GaussianProcess, noise: jnp.ndarray, data: Data
) -> jnp.ndarray:
    x, y = data
    assert_shape(x, (None, 1))
    assert_shape(y, (len(x),))
    l = jnp.linalg.cholesky(gp.kernel(x, x) + noise * jnp.eye(len(x)))
    alpha = solve_triangular(l.transpose(), solve_triangular(l, y))
    res = - (y.transpose() @ alpha).reshape([]) / 2 - jnp.trace(jnp.log(l)) - len(x) * jnp.log(2 * jnp.pi) / 2
    return assert_shape(res, ())


T = TypeVar("T")


class ConjugateGPRegression(ProbabilisticModel[Gaussian], Trainable):
    def __init__(
            self,
            mk_gp: Callable[[jnp.ndarray], GaussianProcess],
            gp_params: jnp.ndarray,
            noise: jnp.ndarray
    ):
        self._mk_gp = mk_gp
        self._gp_params = gp_params
        self._noise = noise

    def marginalise(self, x: jnp.ndarray) -> Gaussian:
        gp = self._mk_gp(self._gp_params)
        return Gaussian(gp.mean_function(x)[..., None], gp.kernel(x, x)[..., None])

    def fit(
            self,
            optimizer: Callable[[jnp.ndarray], Optimizer[jnp.ndarray]],
            data: Data
    ) -> ConjugateGPRegression:
        def objective(params: jnp.ndarray) -> jnp.ndarray:
            (x, y) = data
            noise, *gp_params = params
            return _log_marginal_likelihood(
                self._mk_gp(gp_params), noise, (x, jnp.squeeze(y, axis=-1))
            )

        initial_params = jnp.concatenate([self._noise[None], self._gp_params])
        new_noise, *new_gp_params = optimizer(initial_params)(objective)

        def mk_posterior(gp_params: jnp.ndarray) -> GaussianProcess:
            (x, y) = data
            return _posterior(self._mk_gp(gp_params), new_noise, (x, jnp.squeeze(y, axis=-1)))

        return ConjugateGPRegression(mk_posterior, new_gp_params, new_noise)
