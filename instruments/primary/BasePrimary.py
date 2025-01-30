from abc import abstractmethod
from typing import Iterator

import torch
from torch import Tensor

from instruments.BaseInstrument import BaseInstrument

T = TypeVar("T", bound="BasePrimary")


class BasePrimary(BaseInstrument):
    """
    Abstract base class for all primary instruments.

    A primary instrument is a basic financial instrument that is traded on a market
    and therefore has a market-accessible price.
    Examples include stocks, bonds, commodities, and currencies.

    Attributes:
        dt (float): The uniform length of a time-step in a simulated time-series.
        dtype (torch.dtype): The dtype used for the simulated time-series.
        device (torch.device): The device where the simulated time-series are stored.
    """

    dt: float
    cost: float
    dtype: torch.dtype
    device: torch.device

    def __init__(self) -> None:
        """
        Initializes the BasePrimary class.
        Sets up an ordered dictionary for buffers.
        """
        super().__init__()

    @abstractmethod
    def simulate(self, n_paths: int, n_steps: int, duration: float, **kwargs) -> Iterator[Tensor]:
        """
        Abstract method to yield time-series data for the instrument in a two-step recurrence:
        t_k, t_k+1.

        Shape:
            - Output: :math:`(N) x 1
              :math:`N` stands for the number of simulated paths
        Args:
            n_paths (int): The number of simulation paths.
            duration (float): The duration of the simulation in time units.
        """
        pass
