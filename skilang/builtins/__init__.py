from collections.abc import Iterable
from itertools import chain
from typing import List

import torch
from torch import cat, Tensor


def variance(tensors: List[Tensor]) -> Tensor:
    dim: int = 0 if tensors[0].dim() == 1 else 1
    return cat(tensors, dim=1).var(dim=dim)


def contains(iterables: List | Tensor, value: int) -> bool | Tensor:
    print(iterables)
    if isinstance(iterables, List):
        return value in set(iterables)
    elif isinstance(iterables, Tensor):
        dim: int = 1 if iterables.ndim > 1 else 0
        return torch.any(iterables == value, dim=dim)
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
