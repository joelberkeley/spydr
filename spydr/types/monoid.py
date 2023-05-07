from abc import ABC, abstractmethod
from typing import Self


class Semigroup(ABC):
    @abstractmethod
    def op(self, other: Self) -> Self:
        ...
