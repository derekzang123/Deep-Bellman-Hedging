import torch
from torch import Tensor

from .BaseDerivative import BaseDerivative


class BaseOption(BaseDerivative):
    strike: float
    option: str