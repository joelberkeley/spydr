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
from typing import Callable, Tuple, TypeVar

from jax import numpy as np

from spydr.bayesian_optimization.util import Data
from spydr.distribution import ClosedFormDistribution, variance, Gaussian
from spydr.model import ProbabilisticModel, DistributionType_co

T_co = TypeVar("T_co", covariant=True)

Empiric = Callable[[Tuple[np.ndarray, np.ndarray], ProbabilisticModel[DistributionType_co]], T_co]

Acquisition = Callable[[np.ndarray], np.ndarray]


def expected_improvement(
        predict: ProbabilisticModel[ClosedFormDistribution], best: np.ndarray
) -> Acquisition:
    def acquisition(x: np.ndarray) -> np.ndarray:
        marginal = predict(x)
        return (best - marginal.mean) * marginal.cdf(best) + variance(marginal) * marginal.pdf(best)

    return acquisition


def expected_improvement_by_model(
        data: Data, predict: ProbabilisticModel[ClosedFormDistribution]
) -> Acquisition:
    return expected_improvement(predict, predict(data[0]).mean.min(0))


def probability_of_feasibility(limit: np.ndarray) -> Empiric[ClosedFormDistribution, Acquisition]:
    return lambda _, predict: lambda x: predict(x).cdf(limit)


def negative_lower_confidence_bound(beta: np.ndarray) -> Empiric[Gaussian, Acquisition]:
    if beta < 0:
        raise ValueError

    def empiric(_: Data, predict: ProbabilisticModel[Gaussian]) -> Acquisition:
        def acquisition(x: np.ndarray) -> np.ndarray:
            marginal = predict(x)
            return marginal.mean - beta * variance(marginal)

        return acquisition

    return empiric


def expected_constrained_improvement(
        limit: np.ndarray
) -> Empiric[Gaussian, Callable[[Acquisition], Acquisition]]:
    ...
