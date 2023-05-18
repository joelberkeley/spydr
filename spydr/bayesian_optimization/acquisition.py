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

import functools
from typing import Callable, TypeVar

import chex
from jax import vmap
import jax.numpy as jnp

from spydr.bayesian_optimization.model import Env
from spydr.distribution import ClosedFormDistribution, variance, Gaussian
from spydr.model import ProbabilisticModel
from spydr.types.reader import Reader


Acquisition = Callable[[jnp.ndarray], jnp.ndarray]

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def assert_acquisition_shapes(acquisition: Acquisition) -> Acquisition:
    @functools.wraps(acquisition)
    def _acquisition(x: jnp.array) -> jnp.ndarray:
        res = acquisition(x)
        chex.assert_rank(res, 0)
        return res

    return _acquisition


def expected_improvement(
        predict: ProbabilisticModel[ClosedFormDistribution], best: jnp.ndarray
) -> Acquisition:
    @assert_acquisition_shapes
    def acquisition(x: jnp.ndarray) -> jnp.ndarray:
        marginal = predict(x)
        return jnp.squeeze(
            (best - marginal.mean) * marginal.cdf(best) + variance(marginal) * marginal.pdf(best)
        )

    return acquisition


def expected_improvement_by_model() -> Reader[Env[ProbabilisticModel[ClosedFormDistribution]], Acquisition]:
    def binary(env: Env[ProbabilisticModel[ClosedFormDistribution]]) -> Acquisition:
        qp, obs = env.data.as_tuple()
        chex.assert_shape(qp, [None, 2])  # temp check
        chex.assert_shape(obs, [len(qp), 1])
        return expected_improvement(env.model, env.model(qp).mean.min(0))

    return Reader(binary)


def probability_of_feasibility(
        limit: jnp.ndarray
) -> Reader[Env[ProbabilisticModel[ClosedFormDistribution]], Acquisition]:
    return Reader(lambda env: assert_acquisition_shapes(
        lambda x: jnp.squeeze(env.model(x).cdf(limit))
    ))


def negative_lower_confidence_bound(
        beta: jnp.ndarray
) -> Reader[Env[ProbabilisticModel[Gaussian]], Acquisition]:
    chex.assert_rank(beta, 0)

    if beta < 0:
        raise ValueError

    def empiric(env: Env[ProbabilisticModel[Gaussian]]) -> Acquisition:
        @assert_acquisition_shapes
        def acquisition(x: jnp.ndarray) -> jnp.ndarray:
            marginal = env.model(x)
            return - jnp.squeeze(marginal.mean - beta * variance(marginal))

        return acquisition

    return Reader(empiric)


def expected_constrained_improvement(
        limit: jnp.ndarray
) -> Reader[Env[ProbabilisticModel[Gaussian]], Callable[[Acquisition], Acquisition]]:
    def empiric(env: Env[ProbabilisticModel[Gaussian]]) -> Callable[[Acquisition], Acquisition]:
        query_points, observations = env.data.as_tuple()

        def inner(constraint_fn: Acquisition) -> Acquisition:
            pof = vmap(constraint_fn)(query_points[:, None, ...])
            is_feasible = pof >= limit

            if not jnp.any(is_feasible):
                return constraint_fn

            feasible_query_points = query_points[is_feasible]
            eta = jnp.min(env.model(feasible_query_points).mean, axis=0)
            ei = expected_improvement(env.model, eta)

            return assert_acquisition_shapes(lambda at: ei(at) * constraint_fn(at))
        return inner
    return Reader(empiric)
