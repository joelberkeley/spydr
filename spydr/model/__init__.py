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
from abc import abstractmethod, ABC
from typing import TypeVar, Protocol, Callable, Tuple

from jax import numpy as jnp

from spydr.distribution import Distribution
from spydr.optimize import Optimizer


Data = Tuple[jnp.ndarray, jnp.ndarray]


DistributionType_co = TypeVar("DistributionType_co", bound=Distribution, covariant=True)


class ProbabilisticModel(Protocol[DistributionType_co]):
    @abstractmethod
    def marginalise(self, x: jnp.ndarray) -> DistributionType_co:
        ...


Self = TypeVar("Self")


class Trainable(ABC):
    @abstractmethod
    def fit(self: Self, optimizer: Callable[[jnp.ndarray], Optimizer], data: Data) -> Self:
        ...
