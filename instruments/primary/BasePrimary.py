from abc import abstractmethod
from collections import OrderedDict
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Tuple
from typing import TypeVar

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

    Buffers:
        - spot (:class:`torch.Tensor`): The spot price of the instrument.

    Attributes:
        dtype (torch.dtype): The dtype used for the simulated time-series.
        device (torch.device): The device where the simulated time-series are stored.
    """

    dt: float
    cost: float
    _buffers: Dict[str, Tensor]
    dtype: torch.dtype
    device: torch.device

    def __init__(self) -> None:
        """
        Initializes the BasePrimary class.
        Sets up an ordered dictionary for buffers.
        """
        super().__init__()
        self._buffers = OrderedDict()

    @abstractmethod
    def simulate(self, n_paths: int, duration: float) -> None:
        """
        Abstract method to simulate time-series data for the instrument.

        SShape:
            - Output: :math:`(N) x :math:`M where
              :math:`N` stands for the number of simulated paths, and :math:`M` 
              stands for the number of time steps simulated for each path.

        Args:
            n_paths (int): The number of simulation paths.
            duration (float): The duration of the simulation in time units.
        """
        pass

    def add_buffer(self, name: str, tensor: Tensor) -> None:
        """
        Adds a buffer to the instrument and converts it to the instrument's dtype and device.

        Args:
            name (str): The name of the buffer.
            tensor (torch.Tensor): The tensor to be added as a buffer.
        """
        if isinstance(tensor, Tensor):
            self.to(self.device, self.dtype)
        self._buffers[name] = tensor

    def buffers(self) -> Iterator[Tensor]:
        """
        Returns an iterator over all buffers of the instrument.

        Yields:
            torch.Tensor: Each buffer tensor.
        """
        for _, buffer in self._buffers.items():
            if buffer is not None:
                yield buffer

    def names_buffers(self) -> Iterator[Tuple[str, Tensor]]:
        """
        Returns an iterator over the names and values of all buffers.

        Yields:
            Tuple[str, torch.Tensor]: A tuple containing the name of the buffer and the buffer itself.
        """
        for name, buffer in self._buffers.items():
            if name is not None and buffer is not None:
                yield (name, buffer)

    def get_buffer(self, name: str) -> Tensor:
        """
        Retrieves a buffer by name.

        Args:
            name (str): The name of the buffer to retrieve.

        Returns:
            torch.Tensor: The buffer tensor.

        Raises:
            ValueError: If the buffer with the specified name does not exist.
        """
        if name in self._buffers.keys():
            return self._buffers[name]
        raise ValueError(self._get_name() + " has no buffer named " + name)

    def __get_attr__(self, name: str) -> Tensor:
        """
        Magic method to access a buffer by its name.

        Args:
            name (str): The name of the buffer.

        Returns:
            torch.Tensor: The corresponding buffer tensor.
        """
        return self.get_buffer(name)

    @property
    def spot(self) -> Tensor:
        """
        Returns the spot price buffer of the instrument.

        Returns:
            torch.Tensor: The spot price tensor.

        Raises:
            AttributeError: If the spot buffer is not defined.
        """
        name = "spot"
        if "buffers" in self.__dict__:
            buffers = self.__dict__["buffers"]
            if name in buffers:
                return buffers[name]
        raise AttributeError(
            f"'{self._get_name()}' object has no attribute '{name}'. "
            "Asset may not be simulated.")
    
    @property 
    def var(self) -> Tensor:
        """
        Returns the volatility buffer of an instrument

        Returns:
            torch.Tensor: The asset volatility tensor

        Raises: 
            AttributeError: If the volatility is not defined
        """
        name = "var"
        if "buffers" in self.__dict__:
            buffers = self.__dict__["buffers"]
            if name in buffers:
                return buffers[name]
        raise AttributeError(
            f"'{self._get_name()}' object has no attribute '{name}'. "
            "Asset may not be simulated.")
    
    def to(self: T, *args: Any, **kwargs: Any) -> T:
        """
        Moves or casts all buffers of the instrument to a specified device and/or dtype.

        Args:
            *args (Any): Positional arguments specifying device or dtype.
            **kwargs (Any): Keyword arguments specifying device or dtype.

        Returns:
            self: The instrument with buffers moved or cast.
        """
        device, dtype, *_ = self._parse_to(*args, **kwargs)
        self.device = device
        self.dtype = dtype
        for name, buffer in self.names_buffers():
            self._buffers[name] = buffer.to(self.device, self.dtype)
        return self

    @staticmethod
    def _parse_to(*args: Any,
                  **kwargs: Any) -> Tuple[torch.device, torch.dtype]:
        """
        Parses arguments to extract device and dtype for casting or moving buffers.

        Args:
            *args (Any): Positional arguments.
            **kwargs (Any): Keyword arguments.

        Returns:
            Tuple[torch.device, torch.dtype]: The device and dtype.

        Raises:
            TypeError: If the dtype is not floating point.
        """
        if len(args) > 0 and isinstance(args[0], BaseInstrument):
            instrument = args[0]
            return getattr(instrument, "device"), getattr(instrument, "dtype")
        elif "instrument" in kwargs:
            instrument = kwargs["instrument"]
            return getattr(instrument, "device"), getattr(instrument, "dtype")
        else:
            return torch._C._nn._parse_to(*args, **kwargs)