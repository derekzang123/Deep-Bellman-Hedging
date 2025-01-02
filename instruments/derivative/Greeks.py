import torch

from torch import Tensor

from ..derivative.BaseOption import BaseOption
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def __init__(self, option: BaseOption):
        self.option = option

    def get_delta(self) -> Tensor:
        spot = self.option._underlier.spot.detach().clone()
        spot.requires_grad_()

        self.option._underlier.spot = spot
        price = self.option.payoff().mean()

        delta = torch.autograd.grad(price, spot, create_graph=True)
        return delta

    def gamma(self) -> Tensor:
        spot = self.option._underlier.spot.detach().clone()
        spot.requires_grad_()

        self.option._underlier.spot = spot
        price = self.option.payoff().mean()

        delta, = torch.autograd.grad(price, spot, create_graph=True)
        gamma, = torch.autograd.grad(delta, spot)




