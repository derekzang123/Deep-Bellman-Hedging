import torch
from torch import Tensor
from typing import Tuple
import numpy as np
from instruments.primary.BasePrimary import BasePrimary


class HestonPathsGenerator(BasePrimary):
    def __init__(self, S0: float, v0: float, rho: float, kappa: float,
                 theta: float, xi: float, mu: float, n_steps: int,
                 T: float, dtype=torch.float32, device=torch.device('cpu')) -> None:
        """
        Args:
            S0: Current price of underlying asset
            v0: Initial volatility
            rho: Correlation between Brownian motions
            kappa: Mean reversion rate
            theta: Long-term mean level
            xi: volatility of volatility
            mu: Drift
            n_steps: Time steps in simulation
            T: Total time for simulation
            dtype: Data type for tensors
            device: Device for tensors
        """
        super().__init__()

        # Model parameters
        self.S0 = S0
        self.v0 = v0
        self.rho = rho
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.mu = mu

        # Interest state buffer
        self.interest = None

        # Simulation parameters
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.sqrt_dt = torch.sqrt(torch.tensor(self.dt, dtype=dtype, device=device))

        # Tensor configuration
        self.dtype = dtype
        self.device = device

        # State buffers
        self.S = None
        self.v = None

        # Global parameters
        self.current_step = 0

    def vasicek_step(self):
        """
        Advances the Heston model by one time step.
        Returns:
        dict: A dictionary containing:
            - "interest_rate" (Tensor): The simulated interest ratesat the current step.

        """
        pass

    def vasicek_simulate(self):
        """
        Simulate paths for spot prices and variances for the Heston model.
        Args:
            n_paths: Number of paths to simulate
            duration: Total duration of the simulation

        Yields:
            (Tensor, Tensor): The current spot prices and variances at each time step
        """
        pass

    def reset(self, n_paths: int) -> None:
        """Resets the simulator to initial state"""
        self.current_step = 0
        self.S = torch.full((n_paths,), self.S0,
                            dtype=self.dtype, device=self.device)
        self.v = torch.full((n_paths,), self.v0,
                            dtype=self.dtype, device=self.device)

    def generate_brownians(self, n_paths: int) -> tuple[Tensor, Tensor]:
        Z = torch.randn(2, n_paths, dtype=self.dtype, device=self.device)
        W1 = Z[0]
        W2 = self.rho * Z[0] + torch.sqrt(1 - self.rho ** 2) * Z[1]
        return W1 * self.sqrt_dt, W2 * self.sqrt_dt

    def step(self) -> dict:
        """
        Advances the Heston model by one time step using the Full Truncation Euler Scheme to ensure
        non-negative variance.
        Returns:
            dict: A dictionary containing:
                - "spot" (Tensor): The simulated spot prices at the current step.
                - "volatility" (Tensor): The simulated variance values at the current step.
                - "current_step" (int): The current step in the Heston model.
        """

        n_paths = self.S.shape[0]
        dW1, dW2 = self.generate_brownians(n_paths)

        v_tmp = (self.v + self.kappa
                 * (self.theta - self.v) * self.dt
                 + self.xi * torch.sqrt(self.v) * dW2
        )

        self.v = torch.max(
            v_tmp, torch.tensor(0.0, dtype=self.dtype, device=self.device)
        )

        self.S *= torch.exp(
            (self.mu - 0.5 * self.v) * self.dt + torch.sqrt(self.v) * dW1
        )
        self.current_step += 1

        return {"spot": self.S.clone(), "var": self.v.clone(), "nstep": self.current_step, "interest_rate": self.interest_rate}

    # Add n_steps
    def simulate(self, n_paths: int, duration: float) -> tuple[Tensor, Tensor]:
        """
        Simulate paths for spot prices and variances for the Heston model.
        Args:
            n_paths: Number of paths to simulate
            duration: Total duration of the simulation

        Yields:
            (Tensor, Tensor): The current spot prices and variances at each time step
        """
        S_paths = torch.empty(
            (self.n_steps + 1, n_paths), dtype=self.dtype, device=self.device
        )
        v_paths = torch.empty(
            (self.n_steps + 1, n_paths), dtype=self.dtype, device=self.device
        )

        S_paths[0] = self.S
        v_paths[0] = self.v

        for t in range(1, self.n_steps + 1):
            S_paths[t], v_paths[t] = self.step()

        return S_paths, v_paths

    def heston_generator(self, n_paths: int):
        yield self.S.clone(), self.v.clone()

        while self.current_step < self.n_steps:
            yield self.step

    def characteristic_function(self, u: torch.Tensor, K: float, T: float, r: float):
        """
        Characteristic function of Heston model
        Args:
            u(torch.Tensor): Integration variable
            K(float): Strike price of the option
            T(float): Maturity of the option
            r(float): Risk-free rate

        Returns:

        """
        xi = self.kappa - self.rho * self.xi * 1j * u
        d = torch.sqrt(
            (self.rho * self.xi * 1j * u - xi) ** 2 - self.xi ** 2 * (-u * 1j - u ** 2))
        g = (xi - self.rho * self.xi * 1j * u - d) / (
                xi - self.rho * self.xi * 1j * u + d)

        C = r * 1j * u * self.T + (self.kappa * self.theta) / self.xi ** 2 * (
                (xi - self.rho * self.xi * 1j * u - d) * self.T
                - 2 * torch.log((1 - g * torch.exp(-d * self.T)) / (1 - g))
        )

        D = (xi - self.rho * self.xi * 1j * u - d) / self.xi ** 2 * (
                (1 - torch.exp(-d * self.T)) / (1 - g * torch.exp(-d * self.T))
        )

        return torch.exp(
            C + D * self.v0 + 1j * u *
            torch.log(torch.tensor(self.S0,
                                   dtype=self.dtype, device=self.device))
        )