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
from abc import ABC, abstractmethod

from jax import numpy as np
from jax import scipy

from spydr.util import assert_shape


class Distribution(ABC):
    @property
    @abstractmethod
    def mean(self) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def cov(self) -> np.ndarray:
        ...


def variance(dist: Distribution) -> np.ndarray:
    return dist.cov.diagonal(0, 0, 1)[:, None]


class ClosedFormDistribution(Distribution):
    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def cdf(self, x: np.ndarray) -> np.ndarray:
        ...


class Gaussian(ClosedFormDistribution):
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self._mean = mean
        self._cov = cov

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def cov(self) -> np.ndarray:
        return self._cov

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return scipy.stats.multivariate_normal.pdf(x, self.mean, self.cov)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        ...
