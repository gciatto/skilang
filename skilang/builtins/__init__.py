import statistics
import sys
from collections.abc import Iterable
from itertools import chain
from typing import List, Any

import pandas as pd
import torch
from torch import Tensor

tensor = torch.tensor
DataFrame = pd.DataFrame


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
        dim: int = 0 if iterable.ndim == 1 else 1
        return torch.any(iterable == value, dim=dim)
    else:
        raise TypeError


def concat(*args) -> List | Tensor:
    if isinstance(args[0], Tensor):
        if args[0].dim() >= 2:
            # Batch-wise: stack along new dim (e.g., dim=1)
            return torch.stack(args, dim=1)
        # Case 2: All arguments are 1D tensors (no batch)
        elif args[0].dim() == 1:
            return torch.cat(args, dim=0)
        else:
            raise TypeError(f"unsupported type {type(args[0])} of value {args[0]}")
    elif isinstance(args[0], Iterable):
        return list(chain(*args))
    else:
        raise TypeError


def skilang_builtins_dict() -> dict[str, Any]:
    current_module = sys.modules[__name__]
    return {name: func for name, func in vars(current_module).items() if not name.startswith("_")}
