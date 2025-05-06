import statistics
from collections.abc import Iterable
from itertools import chain
from typing import List

import torch
from torch import Tensor


def variance(iterable: List | Tensor) -> float | Tensor:
    if isinstance(iterable, List):
        return statistics.variance(iterable)
    elif isinstance(iterable, Tensor):
        dim: int = 1 if iterable.ndim > 1 else 0
        return torch.var(iterable.to(torch.float), dim=dim)
    else:
        raise TypeError(f"unsupported type {type(iterable)}")


def contains(iterable: List | Tensor, value: int) -> bool | Tensor:
    if isinstance(iterable, List):
        return value in set(iterable)
    elif isinstance(iterable, Tensor):
        dim: int = 1 if iterable.ndim > 1 else 0
        return torch.any(iterable == value, dim=dim)
    else:
        raise TypeError


def concat(*args) -> List | Tensor:
    if isinstance(args[0], Tensor):
        dim: int = 1 if args[0].ndim > 1 else 0
        return torch.cat(args, dim=dim)
    elif isinstance(args[0], Iterable):
        return list(chain(*args))
    else:
        raise TypeError
