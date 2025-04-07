import torch
from torch import Tensor
from typing import OrderedDict
from .derivative.BaseOption import BaseOption
from torch.distributions.distribution import Distribution

class Portfolio():
    wvec : Tensor
    ovec : OrderedDict[str, BaseOption]
    otraj: Deque[Tensor]

    def init(self, dist: Distribution):
        self.wvec = dist.sample((len(self.ovec,)))

    def simulate(self, n_steps: int, n_paths: int):
        generators = [
            option.simulate(n_paths=n_paths, n_steps=n_steps, duration=self.maturity)
            for option in self.ovec
        ]
        trajectories = [[] for _ in self.ovec]
        for t in range(n_steps):
            for i, gen in enumerate(generators):
                try:
                    value = next(gen)
                except StopIteration:
                    value = torch.zeros(n_paths)
                trajectories[i].append(value)
        trajectories = [w * torch.stack(traj, dim=-1) for w, traj in zip(self.wvec, trajectories)]
        self.otraj = torch.stack(trajectories, dim=1)