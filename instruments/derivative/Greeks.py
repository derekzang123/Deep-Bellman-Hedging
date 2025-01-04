# TODO: WE NEED A MONTE CARLO PRICING SCHEME THAT COMPUTES THE PRICE OF AN OPTION WRT ANY NUMBER OF STOCHASTIC VARIABLES
# AS SUCH, WE NEED A FLEXIBLE PRICER DEFINITION https://en.wikipedia.org/wiki/Monte_Carlo_methods_for_option_pricing

from typing import Callable
from typing import Any

import torch
from torch import Tensor


def set_grad(tensor: Tensor, key: str, **kwargs: Any) -> None:
    """
    Sets the requires_grad attribute of a tensor and updates the kwargs dictionary.

    Args:
        tensor (Tensor): The tensor to be modified.
        key (str): The key in kwargs to associate with the tensor.
        kwargs (Any): The dictionary containing additional arguments.
    """
    tensor = tensor.detach().clone()
    tensor.requires_grad_()
    kwargs[key] = tensor


def delta(
    pricer: Callable[..., Tensor], create_graph: bool = False, **kwargs: Any
) -> Tensor:
    """
    Computes the delta of the option.

    Args:
        pricer (Callable): A pricer function that takes the required arguments and returns the option price.
        create_graph (bool): Whether to create the computational graph for higher-order derivatives.
        kwargs (Any): Additional arguments to pass to the pricer function. Must include 'spot'.

    Returns:
        Tensor: The delta of the option.
    """
    if "spot" not in kwargs:
        raise ValueError(
            "The 'spot' parameter must be provided in kwargs to compute delta."
        )

    set_grad(kwargs["spot"], "spot", kwargs)
    price = pricer(**kwargs)
    (delta,) = torch.autograd.grad(price, kwargs["spot"], create_graph=create_graph)
    return delta


def gamma(
    pricer: Callable[..., Tensor], create_graph: bool = False, **kwargs: Any
) -> Tensor:
    """
    Computes the delta of the option.

    Args:
        pricer (Callable): A pricer function that takes the required arguments and returns the option price.
        create_graph (bool): Whether to create the computational graph for higher-order derivatives.
        kwargs (Any): Additional arguments to pass to the pricer function. Must include 'spot'.

    Returns:
        Tensor: The delta of the option.
    """
    if "spot" not in kwargs:
        raise ValueError(
            "The 'spot' parameter must be provided in kwargs to compute delta."
        )

    set_grad(kwargs["spot"], "spot", kwargs)
    hessian = torch.autograd.functional.hessian(
        lambda spot_param: pricer(spot=spot_param, **kwargs),
        kwargs["spot"],
        create_graph=create_graph,
    )
    gamma = hessian.item() if hessian.numel() == 1 else hessian
    return gamma


def vega(
    pricer: Callable[..., Tensor], create_graph: bool = False, **kwargs: Any
) -> Tensor:
    """
    Computes the delta of the option.

    Args:
        pricer (Callable): A pricer function that takes the required arguments and returns the option price.
        create_graph (bool): Whether to create the computational graph for higher-order derivatives.
        kwargs (Any): Additional arguments to pass to the pricer function. Must include 'spot' and 'ttm'.

    Returns:
        Tensor: The delta of the option.
    """
    for key in ["spot", "var"]:
        if key not in kwargs:
            raise ValueError(
                f"The '{key}' parameter must be provided in kwargs to compute vega."
            )
        set_grad(kwargs[key], kwargs, key)

    price = pricer(**kwargs)
    (vega,) = torch.autograd.grad(price, kwargs["var"], create_graph=create_graph)
    return vega


def theta(
    pricer: Callable[..., Tensor], create_graph: bool = False, **kwargs: Any
) -> Tensor:
    """
    Computes the delta of the option.

    Args:
        pricer (Callable): A pricer function that takes the required arguments and returns the option price.
        create_graph (bool): Whether to create the computational graph for higher-order derivatives.
        kwargs (Any): Additional arguments to pass to the pricer function. Must include 'spot' and 'ttm'.

    Returns:
        Tensor: The delta of the option.
    """
    for key in ["spot", "ttm"]:
        if key not in kwargs:
            raise ValueError(
                f"The '{key}' parameter must be provided in kwargs to compute vega."
            )
        set_grad(kwargs[key], kwargs, key)

    price = pricer(**kwargs)
    (theta,) = torch.autograd.grad(price, kwargs["ttm"])
    return theta


# TODO: CHECK VASICEK
def rho(
    pricer: Callable[..., Tensor], create_graph: bool = False, **kwargs: Any
) -> Tensor:
    """
    Computes the delta of the option.

    Args:
        pricer (Callable): A pricer function that takes the required arguments and returns the option price.
        create_graph (bool): Whether to create the computational graph for higher-order derivatives.
        kwargs (Any): Additional arguments to pass to the pricer function. Must include 'spot' and 'interest'.

    Returns:
        Tensor: The delta of the option.
    """
    for key in ["spot", "interest"]:
        if key not in kwargs:
            raise ValueError(
                f"The '{key}' parameter must be provided in kwargs to compute vega."
            )
        set_grad(kwargs[key], kwargs, key)

    price = pricer(**kwargs)
    (rho,) = torch.autograd.grad(price, kwargs["interest"])
    return rho