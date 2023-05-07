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

from collections.abc import Callable
from dataclasses import dataclass
from typing import final

from jax.scipy.linalg import solve_triangular
from jax import numpy as jnp

from spydr.distribution import Gaussian
from spydr.data import Dataset
from spydr.model import ProbabilisticModel
from spydr.model.kernel import Kernel
from spydr.model.mean_function import MeanFunction
from spydr.optimize import Optimizer
from spydr.shape import assert_shape


@final
@dataclass(frozen=True, eq=False)
class GaussianProcess:
    mean_function: MeanFunction
    kernel: Kernel


def _posterior(
        prior: GaussianProcess, noise: jnp.ndarray, training_data: Dataset
) -> GaussianProcess:
    x_train, y_train = training_data.as_tuple()
    assert_shape(x_train, (None, 1))
    assert_shape(y_train, (len(x_train),))

    l = jnp.linalg.cholesky(prior.kernel(x_train, x_train) + noise * jnp.eye(len(x_train)))
    alpha = solve_triangular(l.transpose(), solve_triangular(l, y_train, lower=True))

    def posterior_meanf(x: jnp.ndarray) -> jnp.ndarray:
        assert_shape(x, (None, 1))
        res = prior.mean_function(x) + prior.kernel(x, x_train) @ alpha
        return assert_shape(res, (len(x),))

    def posterior_kernel(x: jnp.ndarray, x_: jnp.ndarray) -> jnp.ndarray:
        assert_shape(x, (None, 1))
        assert_shape(x_, (None, 1))
        res = (
            prior.kernel(x, x_)
            - (solve_triangular(l, prior.kernel(x_train, x), lower=True)).transpose()
            @ solve_triangular(l, prior.kernel(x_train, x_), lower=True)
        )
        return assert_shape(res, (len(x), len(x_)))

    return GaussianProcess(posterior_meanf, posterior_kernel)


def _log_marginal_likelihood(
        gp: GaussianProcess, noise: jnp.ndarray, data: Dataset
) -> jnp.ndarray:
    x, y = data.as_tuple()
    assert_shape(x, (None, 1))
    assert_shape(y, (len(x),))
    l = jnp.linalg.cholesky(gp.kernel(x, x) + noise * jnp.eye(len(x)))
    alpha = solve_triangular(l.transpose(), solve_triangular(l, y, lower=True))
    res = - (y.transpose() @ alpha).reshape([]) / 2 - jnp.trace(jnp.log(l)) - len(x) * jnp.log(2 * jnp.pi) / 2
    return assert_shape(res, ())


@final
@dataclass(frozen=True, eq=False)
class ConjugateGPRegression:
    data: Dataset
    mk_gp: Callable[[jnp.ndarray], GaussianProcess]
    gp_params: jnp.ndarray
    noise: jnp.ndarray

    def __repr__(self) -> str:
        return f"{self.gp_params}, {self.noise}"


def predict_latent(gpr: ConjugateGPRegression) -> ProbabilisticModel[Gaussian]:
    x, y = gpr.data.as_tuple()
    gp = _posterior(gpr.mk_gp(gpr.gp_params), gpr.noise, Dataset(x, jnp.squeeze(y, axis=-1)))
    return lambda x: Gaussian(gp.mean_function(x)[..., None], gp.kernel(x, x)[..., None])


def predict_observations(gpr: ConjugateGPRegression) -> ProbabilisticModel[Gaussian]:
    x, y = gpr.data.as_tuple()
    gp = _posterior(gpr.mk_gp(gpr.gp_params), gpr.noise, Dataset(x, jnp.squeeze(y, axis=-1)))
    return lambda x: Gaussian(
        gp.mean_function(x)[..., None],
        gp.kernel(x, x)[..., None] + jnp.diag(jnp.broadcast_to(gpr.noise, [len(x)])[..., None])
    )


def fit(
        gpr: ConjugateGPRegression,
        optimizer: Callable[[jnp.ndarray], Optimizer[jnp.ndarray]],
        data: Dataset
) -> ConjugateGPRegression:
    def objective(params: jnp.ndarray) -> jnp.ndarray:
        x, y = data.as_tuple()
        return _log_marginal_likelihood(
            gpr.mk_gp(params[1:]), params[0], Dataset(x, jnp.squeeze(y, axis=-1))
        )

    initial_params = jnp.concatenate([gpr.noise[None], gpr.gp_params])
    new_params = optimizer(initial_params)(objective)

    return ConjugateGPRegression(gpr.data.op(data), gpr.mk_gp, new_params[1:], new_params[0])
