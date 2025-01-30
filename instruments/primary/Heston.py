import numpy as np
import matplotlib.pyplot as plt

from instruments.primary.BasePrimary import BasePrimary

import torch


class Heston(BasePrimary):
    def __init__(self, S0: float, v0: float, rho: float, kappa: float, theta: float, xi: float, mu: float, n_steps: int,
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
        self.S0 = S0
        self.v0 = v0
        self.rho = rho
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.mu = mu
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.dtype = dtype
        self.device = device

    def simulate(self, n_paths: int, duration: float) -> None:
        """
        Simulate paths for spot prices and variances for the Heston model.
        Args:
            n_paths: Number of paths to simulate
            duration: Total duration of the simulation

        Yields:
            (Tensor, Tensor): The current spot prices and variances at each time step
        """
        n_steps = int(duration / self.dt)

        # TODO - initialize s and v to s0 and v0 and have simulation go form there ,,, add generator loop to world.py
        S = torch.zeros((n_paths,), dtype=self.dtype, device=self.device)
        v = torch.zeros((n_paths,), dtype=self.dtype, device=self.device)

        yield S, v

        for t in range(1, n_steps + 1):
            # Independent Brownian motions
            Z1 = torch.normal(mean=torch.zeros(n_paths, dtype=self.dtype, device=self.device),
                              std=torch.ones(n_paths, dtype=self.dtype, device=self.device))
            Z2 = torch.normal(mean=torch.zeros(n_paths, dtype=self.dtype, device=self.device),
                              std=torch.ones(n_paths, dtype=self.dtype, device=self.device))

            # Generate correlated Brownian motions
            W1 = Z1
            W2 = self.rho * Z1 + torch.sqrt(
                torch.tensor(1.0 - self.rho ** 2, dtype=self.dtype, device=self.device)) * Z2

            # Variance process (CIR model)
            v = torch.abs(
                v + self.kappa * (self.theta - v) * self.dt +
                self.xi * torch.sqrt(v) * torch.sqrt(torch.tensor(self.dt, dtype=self.dtype, device=self.device)) * W2)

            # Stock price process
            S = S * torch.exp(
                (self.mu - 0.5 * v) * self.dt +
                torch.sqrt(v) * torch.sqrt(torch.tensor(self.dt, dtype=self.dtype, device=self.device)) * W1)

            yield S, v

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
        d = torch.sqrt((self.rho * self.xi * 1j * u - xi) ** 2 - self.xi ** 2 * (-u * 1j - u ** 2))
        g = (xi - self.rho * self.xi * 1j * u - d) / (xi - self.rho * self.xi * 1j * u + d)

        C = (
                r * 1j * u * self.T
                + (self.kappa * self.theta) / self.xi ** 2
                * ((xi - self.rho * self.xi * 1j * u - d) * self.T - 2 * torch.log(
            (1 - g * torch.exp(-d * self.T)) / (1 - g))))
        D = (xi - self.rho * self.xi * 1j * u - d) / self.xi ** 2 * (
                (1 - torch.exp(-d * self.T)) / (1 - g * torch.exp(-d * self.T)))

        return torch.exp(C + D * self.v0 + 1j * u * torch.log(self.S0))
