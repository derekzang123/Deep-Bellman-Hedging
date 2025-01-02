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

    def get_delta(self) -> Tensor: # dependent variable - option price, independent - value of underlying asset
        spot = self.option._underlier.spot.detach().clone()
        spot.requires_grad_()

        self.option._underlier.spot = spot
        price = self.option.payoff().mean()

        delta = torch.autograd.grad(price, spot, create_graph=True)
        return delta.item()

    def get_gamma(self) -> Tensor: # dependent - delta, indep = value of underlying asset
        spot = self.option._underlier.spot.detach().clone()
        spot.requires_grad_()

        self.option._underlier.spot = spot
        price = self.option.payoff().mean()

        delta, = torch.autograd.grad(price, spot, create_graph=True)
        gamma, = torch.autograd.grad(delta, spot)

        return gamma.item()

    def get_vega(self) -> Tensor: # dependent - option price, independent, volatility
        price = self.option.payoff().mean()

        volatility = None

        vega, = torch.autograd.grad(price, volatility)

        return vega.item()

    def get_theta(self) -> Tensor: # dependent - option price, indep - ttm
        time_to_maturity = self.option.ttm.detach().clone()
        time_to_maturity.requires_grad_()

        price = self.option.payoff().mean()

        theta, = torch.autograd.grad(price, time_to_maturity)

        return theta.item()
    def get_rho(self) -> Tensor: # dependent - option price, indep, interest rate
        price = self.option.payoff().mean()

        interest_rate = None

        vega, = torch.autograd.grad(price, interest_rate)

        return vega.item()



