import torch

from .BasePrimary import BasePrimary
class Vasicek(BasePrimary):
    def __init__(self, theta: float, mu: float, sigma: float, r0: float, T: float, n_steps: int, dtype=torch.float32, device=torch.device('cpu')) -> None:
        super().__init__()
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.r0 = r0
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.dtype = dtype
        self.device = device

    def vasicek(self, n_paths: int) -> None:
        N = self.n_steps
        r = torch.zeros((n_paths, N + 1), dtype=self.dtype, device=self.device)
        r[:, 0] = self.r0

        for t in range(1, N + 1):
            Z = torch.normal(mean=torch.zeros(n_paths, dtype=self.dtype, device=self.device),
                              std=torch.ones(n_paths, dtype=self.dtype, device=self.device))
            dr = self.theta * (self.mu - r[:, t - 1]) * self.dt + self.sigma * torch.sqrt(r[:, t - 1] * self.dt) * Z
            r[:, t] = r[:, t - 1] + dr

        self.add_buffer("interest_rates", r)
        self.add_buffer("spot", r[:,-1])

    def spot(self) -> torch.Tensor:
        return self.get_buffer("spot")

