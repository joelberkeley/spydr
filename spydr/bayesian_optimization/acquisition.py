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

from typing import Callable, TypeVar

from jax import numpy as jnp

from spydr.distribution import ClosedFormDistribution, variance, Gaussian
from spydr.model import Data, ProbabilisticModel, DistributionType_co
from spydr.util import assert_shape

T_co = TypeVar("T_co", covariant=True)

Empiric = Callable[[Data, ProbabilisticModel[DistributionType_co]], T_co]

Acquisition = Callable[[jnp.ndarray], jnp.ndarray]

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def use_pair(f: Callable[[T], tuple[U, V]], g: Callable[[U, V], T_co]) -> Callable[[T], T_co]:
    return lambda t: g(*f(t))


def expected_improvement(
        model: ProbabilisticModel[ClosedFormDistribution], best: jnp.ndarray
) -> Acquisition:
    def acquisition(x: jnp.ndarray) -> jnp.ndarray:
        assert_shape(x, [None, 1])
        marginal = model.marginalise(x)
        res = (best - marginal.mean) * marginal.cdf(best) + variance(marginal) * marginal.pdf(best)
        return assert_shape(res, ())

    return acquisition


def expected_improvement_by_model(
        data: Data, model: ProbabilisticModel[ClosedFormDistribution]
) -> Acquisition:
    qp, obs = data
    assert_shape(qp, (None, 1))
    assert_shape(obs, (len(qp), 1))
    return expected_improvement(model, model.marginalise(data[0]).mean.min(0))


def probability_of_feasibility(limit: jnp.ndarray) -> Empiric[ClosedFormDistribution, Acquisition]:
    return lambda _, model: lambda x: model.marginalise(x).cdf(limit)


def negative_lower_confidence_bound(beta: jnp.ndarray) -> Empiric[Gaussian, Acquisition]:
    if beta < 0:
        raise ValueError

    def empiric(_: Data, model: ProbabilisticModel[Gaussian]) -> Acquisition:
        def acquisition(x: jnp.ndarray) -> jnp.ndarray:
            assert_shape(x, (None, 1))
            marginal = model.marginalise(x)
            res = jnp.squeeze(marginal.mean - beta * variance(marginal))
            return assert_shape(res, ())

        return acquisition

    return empiric


def expected_constrained_improvement(
        limit: jnp.ndarray
) -> Empiric[Gaussian, Callable[[Acquisition], Acquisition]]:
    ...
