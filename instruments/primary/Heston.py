import numpy as np
import matplotlib.pyplot as plt

from BasePrimary import BasePrimary

import torch

class Heston(BasePrimary):
    def __init__(self, S0: float, v0: float, rho: float, kappa: float, theta: float, xi: float, mu: float, n_steps: int, T:float, dtype=torch.float32, device=torch.device('cpu')) -> None:
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
        n_steps = int(duration / self.dt)

        S = torch.zeros((n_paths, n_steps + 1), dtype=self.dtype, device=self.device)
        v = torch.zeros((n_paths, n_steps + 1), dtype=self.dtype, device=self.device)

        S[:, 0] = self.S0

        for t in range(1, n_steps + 1):
            Z1 = torch.normal(mean=torch.zeros(n_paths, dtype=self.dtype, device=self.device),
                              std=torch.ones(n_paths, dtype=self.dtype, device=self.device))
            Z2 = torch.normal(mean=torch.zeros(n_paths, dtype=self.dtype, device=self.device),
                              std=torch.ones(n_paths, dtype=self.dtype, device=self.device))
            W1 = Z1
            W2 = self.rho * Z1 + torch.sqrt(torch.tensor(1.0 - self.rho ** 2, dtype=self.dtype, device=self.device)) * Z2

            # Variance process (CIR model)
            v[:, t] = torch.abs(v[:, t - 1] + self.kappa * (self.theta - v[:, t - 1]) * self.dt + self.xi * torch.sqrt(v[:, t - 1]) * torch.sqrt(torch.tensor(self.dt, dtype=self.dtype, device=self.device)) * W2)

            # Stock price process
            S[:, t] = S[:, t - 1] * torch.exp((self.mu - 0.5 * v[:, t - 1]) * self.dt + torch.sqrt(v[:, t - 1]) * torch.sqrt(torch.tensor(self.dt, dtype=self.dtype, device=self.device)) * W1)

        self.add_buffer("stock_prices", S)
        self.add_buffer("variances", v)
        self.add_buffer("spot", S[:, -1])

    @property
    def spot(self) -> torch.Tensor:
        """
        Returns the spot price (stock prices) buffer of the instrument.

        Returns:
            torch.Tensor: The stock price tensor.
        """
        return self.get_buffer("spot")



