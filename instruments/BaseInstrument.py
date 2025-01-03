from abc import ABC
from abc import abstractmethod

from typing import Any
from typing import List
from typing import TypeVar

import torch
from torch import Tensor

from typing import Any, List, TypeVar
from abc import ABC, abstractmethod
import torch
from torch import Tensor

T = TypeVar("T", bound="BaseInstrument")


class BaseInstrument(ABC):
    """
    Abstract base class for financial instruments.

    This serves as a blueprint for implementing various financial instruments
    that include methods for accessing spot prices, simulation, and transferring
    the instrument between devices or data types.
    """

    @property
    @abstractmethod
    def spot(self: T) -> Tensor:
        """
        Abstract property to retrieve the spot price of the instrument.

        Shape:
            - Output: :math:`(N) x :math:`M where
              :math:`N` stands for the number of simulated paths, and :math:`M` 
              stands for the number of time steps simulated for each path.

        Returns:
            torch.Tensor: The spot price of the instrument.
        """
        pass

    @property
    @abstractmethod
    def var(self: T) -> Tensor:
        """
        Abstract property to retrieve the variance of the instrument. 

        Shape:
            - Output: :math:`(N) x :math:`M where
              :math:`N` stands for the number of simulated paths, and :math:`M` 
              stands for the number of time steps simulated for each path.

        Returns:
            torch.Tensor: The variance of the instrument. 
        """
        pass

    @abstractmethod
    def simulate(self: T, n_paths: int, duration: float, **kwargs) -> None:
        """
        Abstract method to simulate the instrument's price movements.

        Args:
            n_paths (int): Number of paths to simulate.
            duration (float): Duration of the simulation in time units.
            **kwargs: Additional keyword arguments for simulation parameters.

        Returns:
            None
        """
        pass

    @abstractmethod
    def to(self: T, *args: Any, **kwargs: Any) -> T:
        """
        Abstract method to move or cast the instrument's buffers.

        Args:
            *args: Positional arguments for specifying target device or dtype.
            **kwargs: Keyword arguments for specifying target device or dtype.

        Returns:
            self: The instrument after moving or casting.
        """
        pass

    def _get_name(self) -> str:
        return self.__class__.__name__

    def cpu(self: T) -> T:
        """
        Moves the instrument's buffers to CPU memory.

        Note:
            This method modifies the instrument in-place.

        Returns:
            self: The instrument on the CPU.
        """
        return self.to(torch.device("cpu"))

    def cuda(self: T) -> T:
        """
        Moves the instrument's buffers to GPU memory.

        Note:
            This method modifies the instrument in-place.

        Returns:
            self: The instrument on the GPU.
        """
        return self.to(torch.device("cuda"))

    def dtype(self: T, _dtype_: str) -> T:
        """
        Casts the instrument's buffers to a specified floating-point data type.

        Args:
            _dtype_ (str): The name of the desired dtype (e.g., "float32", "float64").

        Returns:
            self: The instrument cast to the specified dtype.

        Raises:
            ValueError: If the specified dtype is not a valid floating-point type.
        """
        # Define only valid floating-point dtypes in PyTorch
        dtypes_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if _dtype_ not in dtypes_map:
            raise ValueError(
                f"Unsupported dtype: {_dtype_}. Supported floating-point dtypes are: {list(dtypes_map.keys())}."
            )
        return self.to(dtype=dtypes_map[_dtype_])

    def _dinfo(self: T) -> List[str]:
        """
        Retrieves information about the instrument's current dtype and device.

        Returns:
            List[str]: A list of strings describing the instrument's dtype and device.
        """
        dinfo = []
        dtype = getattr(self, "dtype", None)
        if dtype is not None:
            if dtype != torch.get_default_dtype():
                dinfo.append("dtype=" + str(dtype))
        device = getattr(self, "device", None)
        if device is not None:
            if device.type != torch._C._get_default_device() or (
                device.type == "cuda" and torch.cuda.current_device() != device.index
            ):
                dinfo.append("device='" + str(device) + "'")
        return dinfo
