from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from spydr.data import Dataset

ModelType_co = TypeVar("ModelType_co", covariant=True)


@dataclass(frozen=True, eq=False)
class Env(Generic[ModelType_co]):
    data: Dataset
    model: ModelType_co


ModelType = TypeVar("ModelType")
ModelType_ = TypeVar("ModelType_")


def map_model(f: Callable[[ModelType], ModelType_]) -> Callable[[Env[ModelType]], Env[ModelType_]]:
    return lambda env: Env(env.data, f(env.model))
