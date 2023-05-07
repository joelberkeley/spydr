# Copyright 2022 Joel Berkeley
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
from typing import final

import jax.numpy as jnp

from spydr.shape import StaticShape
from spydr.types.monoid import Semigroup


@final
@dataclass(frozen=True, eq=False)
class Dataset(Semigroup):
    features: jnp.ndarray
    targets: jnp.ndarray

    def as_tuple(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.features, self.targets

    def op(self, other: Dataset) -> Dataset:
        return Dataset(
            jnp.concatenate([self.features, other.features]),
            jnp.concatenate([self.targets, other.targets])
        )

    @classmethod
    def empty(cls, feature_shape: StaticShape, target_shape: StaticShape) -> Dataset:
        return Dataset(jnp.empty([0] + list(feature_shape)), jnp.empty([0] + list(target_shape)))
