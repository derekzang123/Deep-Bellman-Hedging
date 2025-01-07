from typing import Optional

import numpy as np
from numpy.polynomial import polyfit
from scipy.integrate import quad
import torch
from torch import Tensor

from ..primary import Heston
from .BaseOption import BaseOption


########################## AMERICAN ##########################

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
       option (BaseOption): Instance of BaseOption, provides the strike/maturity
       heston (Heston): Intance of the Heston model, provides the characteristic function
       r: Risk-free interest rate

    Returns:
       Tensor: the price of the European call option using the Heston model.

    """
    K = option.strike
    T = option.maturity

    integrand = lambda u: torch.real(
        torch.exp(-1j * u * torch.log(torch.tensor(K)))
        / (1j * u)
        * heston.characteristic_function(u - 1j, K, T, r)
    )
    integral, _ = quad(integrand, 0, torch.inf)
    return (
        torch.exp(torch.tensor(-r * T)) * 0.5 * heston.S0
        - torch.exp(torch.tensor(-r * T)) / torch.pi * integral
    )


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

    integrand = lambda u: torch.real(
        torch.exp(-1j * u * torch.log(K))
        / (1j * u)
        * heston.characteristic_function(u - 1j, K, T, r)
    )
    integral, _ = quad(integrand, 0, torch.inf)
    return (
        torch.exp(torch.tensor(-r * T)) / torch.pi * integral
        - heston.S0
        + K * torch.exp(torch.tensor(-r * T))
    )

########################## AMERICAN ##########################

def amLS(option: BaseOption, drift: float, deg: Optional[int] = 4) -> Tensor:
    """
    Approximate the price of an American-style option using the Longstaff-Schwartz algorithm.

    This method uses backward induction and polynomial regression to estimate the
    continuation value at each time step, allowing the determination of whether
    early exercise or continuation provides a higher payoff.

    Args:
        option (BaseOption): The American option to be priced, with attributes such
            as spot prices, intrinsic values, and time step size.
        drift (float): The discount rate used to compute the discount factor for
            time value adjustments.
        deg (Optional[int], default=4): The degree of the polynomial used for
            regression in continuation value estimation.

    Returns:
        Tensor: The approximated price of the American-style option.
    """
    v = torch.zeros_like(BaseOption.intrsc)
    v[:-1] = BaseOption.intrsc[:-1]
    disc = np.exp(-drift * option._underlier.dt)

    for t in range(option.spot.size(dim=1) - 1, -1, -1):
        polyn = polyfit(
            option.spot[:, t].numpy(force=True),
            (disc * v[:, t + 1]).numpy(force=True),
            deg,
        )
        cv = torch.tensor(
            polyn(option.spot[:, t].numpy(force=True)), device=option.device
        )
        v[:, t] = torch.where(
            option.intrsc[:, t] > cv,  # Early exercise
            option.intrsc[:, t],  # Intrinsic value
            disc * v[:, t + 1],  # Continuation value
        )
    price = disc * torch.mean(v[:, 0]).item()
    return price


########################## EXOTIC PD ##########################
# def mconstr(option: BaseOption) -> Tensor:
#     """
#     Constructs a spot mesh for the given option based on its underlying assets.

#     Args:
#     option : BaseOption
#         An instance of the BaseOption class containing underlying assets.

#     Returns:
#     Tensor
#         A 3D tensor of shape (num_underliers, num_paths, num_steps) representing the price mesh.
#     """
#     myield = (underlier.spot for underlier in option._underliers)
#     mstack = torch.stack(list(myield), dim=0)
#     return mstack

# def density(spot: float, pspot: float, vol: float, drift: float) -> float:
#     drift = np.log(pspot) + drift
#     pdf = np.exp(-(np.log(spot) - drift) ** 2 / (2 * vol) ** 2) / (spot * vol * np.sqrt(2 * np.pi))
#     return pdf

# def weight(states, pstates, nsamples, nassets, drift, vol) -> float:
#     pass

# def weights():
#     pass

# def intrsc():
#     pass

# def QMat():
#     pass

# def mestmt():
#     pass

# def pestmt():
#     pass
