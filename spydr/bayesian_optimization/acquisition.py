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

from dataclasses import dataclass
from typing import Callable, TypeVar, Generic

from jax import vmap
import jax.numpy as jnp

from spydr.distribution import ClosedFormDistribution, variance, Gaussian
from spydr.model import ProbabilisticModel, DistributionType_co
from spydr.data import Dataset
from spydr.reader import Reader
from spydr.util import assert_shape


@dataclass(frozen=True, eq=False)
class DataAndModel(Generic[DistributionType_co]):
    data: Dataset
    model: ProbabilisticModel[DistributionType_co]


Acquisition = Callable[[jnp.ndarray], jnp.ndarray]

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def expected_improvement(
        predict: ProbabilisticModel[ClosedFormDistribution], best: jnp.ndarray
) -> Acquisition:
    def acquisition(x: jnp.ndarray) -> jnp.ndarray:
        assert_shape(x, [None, 1])
        marginal = predict(x)
        res = jnp.squeeze(
            (best - marginal.mean) * marginal.cdf(best) + variance(marginal) * marginal.pdf(best)
        )
        return assert_shape(res, ())

    return acquisition


def expected_improvement_by_model() -> Reader[DataAndModel[ClosedFormDistribution], Acquisition]:
    def binary(env: DataAndModel[ClosedFormDistribution]) -> Acquisition:
        qp, obs = env.data
        assert_shape(qp, (None, 1))
        assert_shape(obs, (len(qp), 1))
        return expected_improvement(env.model, env.model(qp).mean.min(0))

    return Reader(binary)


def probability_of_feasibility(
        limit: jnp.ndarray
) -> Reader[DataAndModel[ClosedFormDistribution], Acquisition]:
    return Reader(lambda env: lambda x: jnp.squeeze(env.model(assert_shape(x, [1, 2])).cdf(limit)))


def negative_lower_confidence_bound(
        beta: jnp.ndarray
) -> Reader[DataAndModel[Gaussian], Acquisition]:
    if beta < 0:
        raise ValueError

    def empiric(env: DataAndModel[Gaussian]) -> Acquisition:
        def acquisition(x: jnp.ndarray) -> jnp.ndarray:
            assert_shape(x, (None, 1))
            marginal = env.model(x)
            return - assert_shape(jnp.squeeze(marginal.mean - beta * variance(marginal)), ())

        return acquisition

    return Reader(empiric)


def expected_constrained_improvement(
        limit: jnp.ndarray
) -> Reader[DataAndModel[Gaussian], Callable[[Acquisition], Acquisition]]:
    def empiric(env: DataAndModel[Gaussian]) -> Callable[[Acquisition], Acquisition]:
        query_points, observations = env.data

        def inner(constraint_fn: Acquisition) -> Acquisition:
            pof = assert_shape(vmap(constraint_fn)(query_points[:, None, ...]), [None, 1])
            is_feasible = pof >= limit

            if not jnp.any(is_feasible):
                return constraint_fn

            feasible_query_points = query_points[is_feasible]
            eta = jnp.min(env.model(feasible_query_points).mean, axis=0)
            ei = expected_improvement(env.model, eta)

            return lambda at: assert_shape(ei(at) * constraint_fn(at), [])  # type: ignore
        return inner
    return Reader(empiric)
