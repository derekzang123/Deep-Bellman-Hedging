from abc import abstractmethod
from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import TypeVar

import torch
from torch import Tensor

from ..BaseInstrument import BaseInstrument
from ..primary.BasePrimary import BasePrimary

T = TypeVar("T", bound="BaseDerivative")
Clause = Callable[[T, Tensor], Tensor]


class BaseDerivative(BaseInstrument):
    """
    Abstract base class for derivatives, encapsulating properties such as cost, maturity,
    underliers, and clauses. A derivative is a financial instrument whose payoff is contingent on
    a primary instrument (or a set of primary instruments). A derivative is not traded OTC, so price is not directly accessible. 

    Attributes:
        underlier (BasePrimary): The base class for primary instruments.
        cost (float): Cost of the derivative.
        maturity (float): Maturity period of the derivative.
        pricer (Optional[Callable[[Any], Tensor]]): Function to compute the price of the derivative.
        _clauses (Dict[str, Clause]): Dictionary of clauses applied to the payoff function.
        _underliers (Dict[str, BasePrimary]): Dictionary of underlier instruments.
    """

    underlier = BasePrimary
    cost: float
    maturity: float
    pricer: Optional[Callable[[Any], Tensor]]
    _clauses: Dict[str, Clause]
    _underliers: Dict[str, BasePrimary]

    def __init__(self) -> None:
        """
        Initializes the BaseDerivative class with default values.
        """
        super().__init__()
        self.pricer = None
        self.cost = 0.0
        self._clauses = OrderedDict()
        self._underliers = OrderedDict()

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """
        Returns the data type of the underlier(s).

        Raises:
            AttributeError: If there are multiple underliers.
        """
        if len(list(self._underliers())) == 1:
            return self._underliers.items()[0][1].dtype
        else:
            raise AttributeError(
                "dtype() is not well-defined for a derivative with multiple underliers"
            )

    @property
    def device(self) -> Optional[torch.device]:
        """
        Returns the device of the underlier(s).

        Raises:
            AttributeError: If there are multiple underliers.
        """
        if len(list(self._underliers())) == 1:
            return self._underliers.items()[0][1].device
        else:
            raise AttributeError(
                "device() is not well-defined for a derivative with multiple underliers"
            )

    def simulate(self, n_paths: int) -> None:
        """Simulate time series associated with the underlier.

        Args:
            n_paths (int): The number of paths to simulate.
        """
        for _, underlier in self._underliers.items():
            underlier.simulate(n_paths=n_paths, duration=self.maturity)

    def ul(self, index) -> BasePrimary:
        """
        Retrieves an underlier by its index.

        Args:
            index (int): Index of the underlier to retrieve.

        Returns:
            BasePrimary: The underlier at the specified index.
        """
        return self._underliers.items()[index][1]

    def to(self: T, *args: Any, **kwargs: Any) -> T:
        """
        Transfers the derivative and its underliers to a specified device.

        Args:
            *args: Positional arguments for the device transfer.
            **kwargs: Keyword arguments for the device transfer.

        Returns:
            T: The updated instance of the derivative.
        """
        for _, underlier in self._underliers.items():
            underlier.to(*args, **kwargs)
        return self

    @abstractmethod
    def payoff_fn(self) -> Tensor:
        """
        Defines the payoff function of the derivative.

        Shape:
            - Output: :math:(N) where
              :math:N stands for the number of simulated paths.

        Returns:
            torch.Tensor
        """
        pass

    def payoff(self) -> Tensor:
        """
        Computes the payoff of the derivative, applying all registered clauses.

        Shape:
            - Output: :math:`(N)` where
              :math:`N` stands for the number of simulated paths.

        Returns:
            torch.Tensor: Adjusted payoff values.
        """
        payoff = self.payoff_fn
        for _, clause in self._clauses.items():
            payoff = clause(self, payoff)
        return payoff

    def list(self: T, pricer: Callable[[T], Tensor], cost: float) -> None:
        """
        Lists the derivative with a specified pricer and cost.

        Args:
            pricer (Callable[[T], Tensor]): Function to compute the price of the derivative.
            cost (float): Cost associated with the derivative.
        """
        self.pricer = pricer
        self.cost = cost

    def unlist(self: T) -> None:
        """
        Removes the pricer and resets the cost to zero.
        """
        self.pricer = None
        self.cost = 0.0

    def add_clause(self, name: str, clause: Clause) -> None:
        """
        Adds a clause to modify the payoff function.

        Args:
            name (str): Name of the clause.
            clause (Clause): Callable clause function.
        """
        self._clauses[name] = clause

    def add_underlier(self, name: str, underlier: BasePrimary) -> None:
        """
        Adds an underlier to the derivative.

        Args:
            name (str): Name of the underlier.
            underlier (BasePrimary): Underlier instrument to add.
        """
        self._underliers[name] = underlier

    def clauses(self) -> Iterator[Clause]:
        """
        Iterates over the registered clauses.

        Yields:
            Clause: Callable clause functions.
        """
        for _, clause in self._clauses.items():
            if clause is not None:
                yield clause

    def names_clauses(self) -> Iterator[Tuple[str, Clause]]:
        """
        Iterates over the names and registered clauses.

        Yields:
            Tuple[str, Clause]: Name and callable clause functions.
        """
        for name, clause in self._clauses.items():
            if name is not None and clause is not None:
                yield (name, clause)

    def underliers(self) -> Iterator[BasePrimary]:
        """
        Iterates over the registered underliers.

        Yields:
            BasePrimary: Underlier instruments.
        """
        for _, underlier in self._underliers.items():
            if underlier is not None:
                yield underlier

    def names_underliers(self) -> Iterator[Tuple[str, BasePrimary]]:
        """
        Iterates over the names and registered underliers.

        Yields:
            Tuple[str, BasePrimary]: Name and underlier instruments.
        """
        for name, underlier in self._underliers.items():
            if name is not None and underlier is not None:
                yield (name, underlier)

    def get_underlier(self, name: str) -> BasePrimary:
        """
        Retrieves an underlier by its name.

        Args:
            name (str): Name of the underlier.

        Returns:
            BasePrimary: The underlier with the specified name.

        Raises:
            ValueError: If the name does not correspond to any registered underlier.
        """
        if name in self._underliers:
            return self._underliers[name]
        raise ValueError(self._get_name() + " has no buffer named " + name)

    def __getattr__(self, name: str) -> BasePrimary:
        """
        Retrieves an underlier using attribute access.

        Args:
            name (str): Name of the underlier.

        Returns:
            BasePrimary: The underlier with the specified name.
        """
        return self.get_underlier(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets an attribute, registering it as an underlier if applicable.

        Args:
            name (str): Attribute name.
            value (Any): Attribute value.
        """
        if isinstance(value, BasePrimary):
            self.register_underlier(name, value)
        super().__setattr__(name, value)

    @property
    def spot(self) -> Tensor:
        """
        Computes the spot price of the derivative.

        Returns:
            torch.Tensor: Spot price.

        Raises:
            ValueError: If the derivative is not listed.
        """
        if self.pricer is None:
            raise ValueError("self is not listed.")
        return self.pricer(self)
