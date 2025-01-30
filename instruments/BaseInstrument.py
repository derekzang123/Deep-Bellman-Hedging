from abc import ABC
from abc import abstractmethod
from typing import Iterator

import torch
from torch import Tensor

T = TypeVar("T", bound="BaseInstrument")


class BaseInstrument(ABC):
    """
    Abstract base class for financial instruments.
    """

    @abstractmethod
    def simulate(self: T, n_paths: int, n_steps: int, duration: float, **kwargs) -> Iterator[Tensor]:
        """
        Abstract method to simulate the instrument's price movements.

        Args:
            n_paths (int): Number of paths to simulate.
            duration (float): Duration of the simulation in time units.
            **kwargs: Additional keyword arguments for simulation parameters.

        Yields:
            torch.Tensor:
                A tensor (e.g., shape *(n_paths,)* or *(n_paths, features)*)
                representing the state of the instrument at each step.
        """
        pass
