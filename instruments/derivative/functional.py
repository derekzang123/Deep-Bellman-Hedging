import numpy as np
from scipy.integrate import quad
from typing import Callable, Any
import torch
from torch import Tensor

from ..primary import Heston
from .BaseOption import BaseOption


def heston_option_price(heston: Heston, option: BaseOption, r: float, option_type=str):
    if option_type == "call":
        return heston_call_price(heston, option, r)
    elif option_type == "put":
        return heston_put_price(heston, option, r)
    else:
        raise ValueError("Invalid option type, must be 'call' or 'put'")


def heston_call_price(heston: Heston, option: BaseOption, r: float):
    """
   Calculate the price of a European call option using the Heston model.

   Args:
      option (BaseOption): Instance of BaseOption, provies the strike/maturity
      heston (Heston): Intance of the Heston model, provides the characteristic function
      r: Risk-free interest rate

   Returns:
      Tensor: the price of the European call option using the Heston model.

   """
    K = option.strike
    T = option.maturity

    integrand = lambda u: torch.real(torch.exp(-1j * u * torch.log(torch.tensor(K))) /
                                     (1j * u) * heston.characteristic_function(u - 1j, K, T, r))
    integral, _ = quad(integrand, 0, torch.inf)
    return torch.exp(torch.tensor(-r * T)) * 0.5 * heston.S0 - torch.exp(torch.tensor(-r * T)) / torch.pi * integral


def heston_put_price(heston: Heston, option: BaseOption, r: float):
    """
   Calculate the price of a European put option using the Heston model.

   Args:
      option (BaseOption): Instance of BaseOption, provies the strike/maturity
      heston (Heston): Intance of the Heston model, provides the characteristic function
      r: Risk-free interest rate

   Returns:
      Tensor: the price of the European put option using the Heston model.

   """

    K = option.strike
    T = option.maturity

    integrand = lambda u: torch.real(torch.exp(-1j * u * torch.log(K)) /
                                     (1j * u) * heston.characteristic_function(u - 1j, K, T, r))
    integral, _ = quad(integrand, 0, torch.inf)
    return torch.exp(torch.tensor(-r * T)) / torch.pi * integral - heston.S0 + K * torch.exp(torch.tensor(-r * T))
