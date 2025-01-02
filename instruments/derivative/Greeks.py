import torch

from ..derivative.BaseDerivative import BaseDerivative
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def __init__(self, derivative: BaseDerivative):
        self.derivative = derivative

    # only works for European options which only exercise at expiry... must add functionality to support early exercise afforded by American options
    def delta(self, epsilon: float = 1e-5) -> float: #  how much an option's price can be expected to move for every $1 change in the price of the underlying security or index
        spot_price = self.derivative.spot

        up_price = spot_price + 1
        down_price = spot_price - 1

        self.derivative.simulate(n_paths=10000)
        self.derivative.spot = up_price
        price_up = self.derivative.payoff()

        self.derivative.simulate(n_paths = 10000)
        self.derivative.spot = down_price
        price_down = self.derivative.payoff()

        delta = (price_up - price_down) / 2
        return delta.mean().item()