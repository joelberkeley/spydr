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
from typing import final, Final

import chex
from jax.scipy.linalg import solve_triangular
from jax import numpy as jnp

from spydr.distribution import Gaussian
from spydr.data import Dataset
from spydr.model import ProbabilisticModel
from spydr.model.kernel import Kernel, kernel_shapes
from spydr.model.mean_function import (
    MeanFunction,
    mean_function_shapes,
)
from spydr.optimize import Optimizer
from spydr.shape import Shape


@final
class GaussianProcess:
    def __init__(
            self,
            mean_function: MeanFunction,
            kernel: Kernel,
            *,
            feature_shape: Shape,
            target_shape: Shape,
    ):
        self.mean_function: Final = mean_function_shapes(feature_shape, mean_function)
        self.kernel: Final = kernel_shapes(feature_shape, kernel)
        self.feature_shape: Final = feature_shape
        self.target_shape: Final = target_shape


def _posterior(
        prior: GaussianProcess, noise: jnp.ndarray, training_data: Dataset
) -> GaussianProcess:
    chex.assert_rank(noise, 0)
    chex.assert_shape(training_data.targets, [len(training_data.features)])

    x_train, y_train = training_data.as_tuple()

    l = jnp.linalg.cholesky(prior.kernel(x_train, x_train) + noise * jnp.eye(len(x_train)))
    alpha = solve_triangular(l.transpose(), solve_triangular(l, y_train, lower=True))

    def posterior_meanf(x: jnp.ndarray) -> jnp.ndarray:
        return prior.mean_function(x) + prior.kernel(x, x_train) @ alpha

    def posterior_kernel(x: jnp.ndarray, x_: jnp.ndarray) -> jnp.ndarray:
        return (
            prior.kernel(x, x_)
            - (solve_triangular(l, prior.kernel(x_train, x), lower=True)).transpose()
            @ solve_triangular(l, prior.kernel(x_train, x_), lower=True)
        )

    return GaussianProcess(
        posterior_meanf,
        posterior_kernel,
        feature_shape=prior.feature_shape,
        target_shape=prior.target_shape,
    )


def _log_marginal_likelihood(
        gp: GaussianProcess, noise: jnp.ndarray, data: Dataset
) -> jnp.ndarray:
    chex.assert_rank(noise, 0)
    x, y = data.as_tuple()
    l = jnp.linalg.cholesky(gp.kernel(x, x) + noise * jnp.eye(len(x)))
    alpha = solve_triangular(l.transpose(), solve_triangular(l, y, lower=True))
    lml = (
        - (y.transpose() @ alpha).reshape([]) / 2
        - jnp.trace(jnp.log(l)) - len(x) * jnp.log(2 * jnp.pi) / 2
    )
    chex.assert_rank(lml, 0)
    return lml


@final
@dataclass(frozen=True, eq=False)
class ConjugateGPRegression:
    data: Dataset
    mk_gp: Callable[[jnp.ndarray], GaussianProcess]
    gp_params: jnp.ndarray
    noise: jnp.ndarray

    def __post_init__(self) -> None:
        chex.assert_rank(self.noise, 0)

    def __repr__(self) -> str:
        return f"{self.gp_params}, {self.noise}"


def predict_latent(gpr: ConjugateGPRegression) -> ProbabilisticModel[Gaussian]:
    x, y = gpr.data.as_tuple()
    gp = _posterior(gpr.mk_gp(gpr.gp_params), gpr.noise, Dataset(x, jnp.squeeze(y, axis=-1)))

    def prob_model(x: jnp.ndarray) -> Gaussian:
        chex.assert_shape(x, [None] + list(gp.feature_shape))
        return Gaussian(gp.mean_function(x)[..., None], gp.kernel(x, x)[..., None])

    return prob_model


def predict_observations(gpr: ConjugateGPRegression) -> ProbabilisticModel[Gaussian]:
    x, y = gpr.data.as_tuple()
    gp = _posterior(gpr.mk_gp(gpr.gp_params), gpr.noise, Dataset(x, jnp.squeeze(y, axis=-1)))

    def prob_model(x: jnp.ndarray) -> Gaussian:
        chex.assert_shape(x, [None] + list(gp.feature_shape))
        return Gaussian(
            gp.mean_function(x)[..., None],
            gp.kernel(x, x)[..., None] + jnp.diag(jnp.broadcast_to(gpr.noise, [len(x)])[..., None])
        )

    return prob_model


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
