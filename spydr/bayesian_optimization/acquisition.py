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

import jax.numpy as jnp
from jax import jit

from spydr.bayesian_optimization.binary import Binary
from spydr.distribution import ClosedFormDistribution, variance, Gaussian
from spydr.model import ProbabilisticModel, DistributionType_co
from spydr.data import Dataset
from spydr.util import assert_shape

T_co = TypeVar("T_co", covariant=True)

Empiric = Binary[Dataset, ProbabilisticModel[DistributionType_co], T_co]

Acquisition = Callable[[jnp.ndarray], jnp.ndarray]

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def expected_improvement(
        predict: ProbabilisticModel[ClosedFormDistribution], best: jnp.ndarray
) -> Acquisition:
    # @jit
    def acquisition(x: jnp.ndarray) -> jnp.ndarray:
        assert_shape(x, [None, 1])
        marginal = predict(x)
        res = jnp.squeeze(
            (best - marginal.mean) * marginal.cdf(best) + variance(marginal) * marginal.pdf(best)
        )
        return assert_shape(res, ())

    return acquisition


def expected_improvement_by_model() -> Empiric[ClosedFormDistribution, Acquisition]:
    def binary(
            data: Dataset, predict: ProbabilisticModel[ClosedFormDistribution]
    ) -> Acquisition:
        qp, obs = data
        assert_shape(qp, (None, 1))
        assert_shape(obs, (len(qp), 1))
        return expected_improvement(predict, predict(qp).mean.min(0))

    return Binary(binary)


def probability_of_feasibility(limit: jnp.ndarray) -> Empiric[ClosedFormDistribution, Acquisition]:
    return Binary(lambda _, predict: jit(lambda x: jnp.squeeze(predict(x).cdf(limit))))


def negative_lower_confidence_bound(beta: jnp.ndarray) -> Empiric[Gaussian, Acquisition]:
    if beta < 0:
        raise ValueError

    def empiric(_: Dataset, predict: ProbabilisticModel[Gaussian]) -> Acquisition:
        @jit
        def acquisition(x: jnp.ndarray) -> jnp.ndarray:
            assert_shape(x, (None, 1))
            # marginal = predict(x)
            # res = jnp.squeeze(marginal.mean - beta * variance(marginal))
            return assert_shape(jnp.squeeze(x ** 2), ())

        return acquisition

    return Binary(empiric)


def expected_constrained_improvement(
        limit: jnp.ndarray
) -> Empiric[Gaussian, Callable[[Acquisition], Acquisition]]:
    def empiric(
            data: Dataset, predict: ProbabilisticModel[Gaussian]
    ) -> Callable[[Acquisition], Acquisition]:
        query_points, observations = data

        def inner(constraint_fn: Acquisition) -> Acquisition:
            # todo we don't support broadcasting dimensions on acquisition fns yet
            #
            # ^^^ don't need to - can use vmap/pmap
            pof = constraint_fn(query_points[:, None, ...])
            is_feasible = jnp.squeeze(pof >= limit, axis=-1)

            if not jnp.any(is_feasible):
                return constraint_fn

            feasible_query_points = query_points[is_feasible]
            eta = jnp.min(predict(feasible_query_points).mean, axis=0)
            ei = expected_improvement(predict, eta)

            return jit(lambda at: ei(at) * constraint_fn(at))  # type: ignore
        return inner
    return Binary(empiric)
