from abc import ABC
from abc import abstractmethod

from typing import Optional
from typing import TypeVar

import torch
from torch import Tensor

from ..BaseInstrument import BaseInstrument

T = TypeVar("T", bound="BaseOption")

class BaseOption(ABC):
    """
    A mixin class for general, financial instruments providing methods to compute moneyness and time to maturity (TTM).

    Attributes:
        underlier (BasePrimary): The underlying financial instrument.
        strike (float): The strike price of the option.
        maturity (float): The maturity (time to expiration) of the option.
    """

    underlier: BaseInstrument
    strike: float
    maturity: float

    @abstractmethod
    def payoff_fn(self) -> Tensor:
        """
        Abstract method to compute the payoff of the option or derivative.

        This method must be implemented by subclasses and should define the specific
        payoff logic, including handling path-dependent scenarios where the payoff
        depends on the full price path or intermediate values of the underlier(s).

        For path-independent derivatives (e.g., European options), the payoff typically
        depends only on the final step of the underlier's spot price.
        For path-dependent derivatives (e.g., Asian options, barrier options), this
        method should process the entire path or relevant intermediate steps to compute
        the payoff.

        Returns:
            Tensor: A tensor of shape `(n_paths,)` representing the computed payoff
            for each simulated path.

        Note:
            - If the derivative has multiple underliers, ensure that the payoff
            logic correctly aggregates or selects the relevant data from
            `self.underlier.spot`.
            - This method should assume that `self.underlier.spot` is a tensor of shape
            `(n_paths, n_steps)` for each underlier, with consistent paths and steps.
        """
        pass

    def payoff(self) -> Tensor:
        """
        Computes the payoff of the derivative using the `payoff_fn` method.

        This method provides a standardized interface to compute the payoff
        by calling the subclass-specific implementation of `payoff_fn`.

        Returns:
            Tensor: A tensor of shape `(n_paths,)` representing the payoff
            values for each path.

        Note:
            - For path-dependent derivatives, ensure that the `payoff_fn` processes
            the entire path or relevant intermediate steps as needed.
            - For derivatives with multiple underliers, ensure that `payoff_fn`
            correctly handles or aggregates the data from all underliers.
        """
        payoff = self.payoff_fn()
        return payoff

    def moneyness(self, step: Optional[int] = None, log: bool = False) -> Tensor:
        """
        Computes the moneyness or log-moneyness of the instrument.

        Args:
            step (Optional[int]): The specific time step for which to compute moneyness.
                If `None`, computes moneyness across all steps.
            log (bool): If True, computes log-moneyness. Defaults to False.

        Returns:
            Tensor: A tensor containing the moneyness or log-moneyness.

        Raises:
            ValueError: If the specified `step` is out of range.
        """
        n_steps = self.underlier.spot.size(1)
        if step is not None:
            if step < 0 or step >= n_steps:
                raise ValueError(f"step {step} is out of range for n_steps {n_steps}.")
        spot = (
            self.underlier.spot[..., step] if step is not None else self.underlier.spot
        )
        moneyness = spot / self.strike
        return torch.log(moneyness) if log else moneyness

    def ttm(self, step: Optional[int] = None) -> Tensor:
        """
        Computes the time to maturity (TTM) for the instrument.

        Args:
            step (Optional[int]): The specific time step for which to compute TTM.
                If `None`, computes TTM for all steps.

        Returns:
            Tensor: A tensor of shape `(n_paths, n_steps)` or `(n_paths, 1)` containing the TTM values.

        Raises:
            ValueError: If the specified `step` is out of range.
        """
        n_paths, n_steps = self.underlier.spot.size()
        dt = self.underlier.dt

        if step is None:
            times = torch.arange(n_steps, device=self.underlier.spot.device) * dt
            ttm = times[-1] - times
            return ttm.unsqueeze(0).expand(n_paths, -1)
        else:
            if step < 0 or step >= n_steps:
                raise ValueError(f"step {step} is out of range for n_steps {n_steps}.")
            ttm = (n_steps - step - 1) * dt
            return torch.full((n_paths, 1), ttm, device=self.underlier.spot.device)
