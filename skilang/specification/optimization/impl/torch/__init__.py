from typing import Iterator, Callable

import torch.nn.modules.loss as loss_module
import torch.optim
from torch.nn import Parameter
from torch.nn.modules.loss import _Loss as TorchLoss
from torch.optim.optimizer import Optimizer as TorchOptimizer

from skilang.specification.optimization import Loss, Optimizer


# def get_loss_class(loss_name: str) -> nn.Module:
#     if loss_name not in loss_module.__all__:
#         raise ValueError(f"{loss_name} is not a valid loss function.")
#
#     loss_class: nn.Module = getattr(nn, loss_name)
#     return loss_class


loss_map: dict[Loss, TorchLoss] = {
    Loss.CROSS_ENTROPY: loss_module.CrossEntropyLoss(reduction="none"),
    Loss.MSE: loss_module.MSELoss(reduction="none"),
    Loss.BINARY_CROSS_ENTROPY: loss_module.BCELoss(reduction="none"),
    Loss.NLL: loss_module.NLLLoss(reduction="none"),
    Loss.L1: loss_module.L1Loss(reduction="none"),
    Loss.KL_DIVERGENCE: loss_module.KLDivLoss(reduction="none"),
}


def get_torch_loss(loss_name: Loss) -> TorchLoss:
    """
    Get the loss function class from the loss name.
    :param loss_name: The name of the loss function.
    :return: The loss function class.
    """
    try:
        return loss_map[loss_name]
    except KeyError:
        raise ValueError(
            f"Unknown loss function: {loss_name}. Available are: {', '.join(Loss.__members__.keys())}"
        )


optimizer_map: dict[Optimizer, Callable[..., TorchOptimizer]] = {
    Optimizer.ADAM: torch.optim.Adam,
    Optimizer.SGD: torch.optim.SGD,
    Optimizer.ASGD: torch.optim.ASGD,
    Optimizer.ADAGRAD: torch.optim.Adagrad,
    Optimizer.ADADELTA: torch.optim.Adadelta,
    Optimizer.ADAMW: torch.optim.AdamW,
    Optimizer.ADAMAX: torch.optim.Adamax,
    Optimizer.ADAFACTOR: torch.optim.Adafactor,
}


def get_torch_optimizer(optimizer_name: Optimizer, params: Iterator[Parameter], lr: float) -> TorchOptimizer:
    """
    Get the optimizer class from the optimizer name.
    :param optimizer_name: The name of the optimizer.
    :param params: An iterator of parameters to optimize.
    :param lr: The learning rate for the optimizer.
    :return: The optimizer class.
    """
    try:
        optimizer_class = optimizer_map[optimizer_name]
        return optimizer_class(params, lr=lr)
    except KeyError:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. Available are: {', '.join(Optimizer.__members__.keys())}"
        )
