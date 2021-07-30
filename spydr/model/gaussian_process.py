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
from collections import Callable
from dataclasses import dataclass

from jax.scipy.linalg import solve_triangular
from jax import numpy as np
from mypy.nodes import TypeVar

from spydr.distribution import Distribution, Gaussian
from spydr.model.kernel import Kernel
from spydr.model.mean_function import MeanFunction
from spydr.optimize import Optimizer


@dataclass
class GaussianProcess:
    mean_function: MeanFunction
    kernel: Kernel


def marginalise(gp: GaussianProcess, x: np.ndarray) -> Distribution:
    return Gaussian(gp.mean_function(x), gp.kernel(x, x))


def _posterior(
        prior: GaussianProcess,
        likelihood: Gaussian,
        training_data: tuple[np.ndarray, np.ndarray]
) -> GaussianProcess:
    x_train, y_train = training_data
    l = np.linalg.cholesky(prior.kernel(x_train, x_train) + likelihood.cov)
    alpha = solve_triangular(l.transpose(), solve_triangular(l, y_train))

    def posterior_meanf(x: np.ndarray) -> np.ndarray:
        return prior.mean_function(x) + (prior.kernel(x, x_train) @ alpha)

    def posterior_kernel(x: np.ndarray, x_: np.ndarray) -> np.ndarray:
        return (
            prior.kernel(x, x_)
            - (solve_triangular(l, prior.kernel(x_train, x))).transpose()
            @ solve_triangular(l, prior.kernel(x_train, x_))
        )

    return GaussianProcess(posterior_meanf, posterior_kernel)


def _log_marginal_likelihood(
        gp: GaussianProcess, likelihood: Gaussian, data: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    x, y = data
    l = np.linalg.cholesky(gp.kernel(x, x) + likelihood.cov)
    alpha = solve_triangular(l.transpose(), solve_triangular(l, y))
    return - y @ alpha / 2 - np.trace(np.log(l)) - len(x) * np.log(2 * np.pi) / 2


T = TypeVar("T")


def fit(
        optimizer: Optimizer[T],
        mk_prior: Callable[[T], GaussianProcess],
        mk_likelihood: Callable[[T], Gaussian],
        data: tuple[np.ndarray, np.ndarray]
) -> GaussianProcess:
    def objective(hp: T) -> np.ndarray:
        return _log_marginal_likelihood(mk_prior(hp), mk_likelihood(hp), data)

    params = optimizer(objective)

    return _posterior(mk_prior(params), mk_likelihood(params), data)