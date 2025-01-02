import torch

from torch import Tensor

from ..derivative.BaseOption import BaseOption

class Greeks:
    @staticmethod
    def get_delta(derivative: BaseOption, params: dict) -> Tensor:
        """
        Computes the delta of the option.

        Args:
            derivative (BaseOption): The option derivative.
            params (dict): A dictionary containing 'spot_price'

        Returns:
            Tensor: The delta of the option.
        """
        spot = params["spot_price"]
        spot_tensor = spot.detach().clone()
        spot_tensor.requires_grad_()

        derivative._underlier.spot = spot
        price = derivative.payoff().mean()

        delta, = torch.autograd.grad(price, spot_tensor, create_graph=True)
        return delta

    @staticmethod
    def get_gamma(derivative: BaseOption, params: dict) -> Tensor:
        """
            Computes the gamma of the option.

            Args:
                derivative (BaseOption): The option derivative.
                params (dict): A dictionary containing 'spot_price'

            Returns:
                Tensor: The gamma of the option.
        """
        spot = params["spot_price"]
        spot_tensor = spot.detach().clone()
        spot_tensor.requires_grad_()

        derivative._underlier.spot = spot
        price = derivative.payoff().mean()

        delta, = torch.autograd.grad(price, spot_tensor, create_graph=True)
        gamma, = torch.autograd.grad(delta, spot_tensor)

        return gamma

    @staticmethod
    def get_vega(derivative: BaseOption, params: dict) -> Tensor: # dependent - option price, independent, volatility
        """
            Computes the vega of the option.

            Args:
                derivative (BaseOption): The option derivative.
                params (dict): A dictionary containing 'volatility'

            Returns:
                Tensor: The vega of the option.
        """
        volatility = params["volatility"]

        derivative._underlier.volatility = volatility
        price = derivative.payoff().mean()

        vega, = torch.autograd.grad(price, volatility)

        return vega
    @staticmethod
    def get_theta(derivative: BaseOption, params: dict) -> Tensor: # dependent - option price, indep - ttm
        """
            Computes the theta of the option.

            Args:
                derivative (BaseOption): The option derivative.
                params (dict): A dictionary containing 'time_to_maturity'.

            Returns:
                Tensor: The theta of the option.
        """
        time_to_maturity = params["time_to_maturity"]
        time_to_maturity_tensor = time_to_maturity.detach().clone()
        time_to_maturity_tensor.requires_grad_()

        derivative._underlier.time_to_maturity = time_to_maturity_tensor
        price = derivative.payoff().mean()

        theta, = torch.autograd.grad(price, time_to_maturity_tensor)

        return theta
    @staticmethod
    def get_rho(derivative: BaseOption, params: dict) -> Tensor: # dependent - option price, indep, interest rate
        """
            Computes the rho of the option.

            Args:
                derivative (BaseOption): The option derivative.
                params (dict): A dictionary containing 'interest_rate'.

            Returns:
                Tensor: The rho of the option.
        """
        interest_rate = params["interest_rate"]

        derivative._underlier.interest_rate = interest_rate
        price = derivative.payoff().mean()

        rho, = torch.autograd.grad(price, interest_rate)

        return rho



