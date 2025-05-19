from typing import Iterator

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


def get_torch_loss(loss_name: Loss) -> TorchLoss:
    """
    Get the loss function class from the loss name.
    :param loss_name: The name of the loss function.
    :return: The loss function class.
    """
    if loss_name == Loss.CROSS_ENTROPY:
        loss = loss_module.CrossEntropyLoss(reduction="none")
    elif loss_name == Loss.MSE:
        loss = loss_module.MSELoss(reduction="none")
    elif loss_name == Loss.BINARY_CROSS_ENTROPY:
        loss = loss_module.BCELoss(reduction="none")
    elif loss_name == Loss.NLL:
        loss = loss_module.NLLLoss(reduction="none")
    elif loss_name == Loss.L1:
        loss = loss_module.L1Loss(reduction="none")
    elif loss_name == Loss.KL_DIVERGENCE:
        loss = loss_module.KLDivLoss(reduction="none")
    else:
        raise ValueError(
            f"Unknown loss function: {loss_name}. Available are: {', '.join(Loss.__members__.keys())}"
        )
    return loss


def get_torch_optimizer(optimizer_name: Optimizer, params: Iterator[Parameter], lr: float) -> TorchOptimizer:
    """
    Get the optimizer class from the optimizer name.
    :param optimizer_name: The name of the optimizer.
    :return: The optimizer class.
    """
    if optimizer_name == Optimizer.ADAM:
        optimizer = torch.optim.Adam(params, lr=lr)
    elif optimizer_name == Optimizer.SGD:
        optimizer = torch.optim.SGD(params, lr=lr)
    elif optimizer_name == Optimizer.ASGD:
        optimizer = torch.optim.ASGD(params, lr=lr)
    elif optimizer_name == Optimizer.ADAGRAD:
        optimizer = torch.optim.Adagrad(params, lr=lr)
    elif optimizer_name == Optimizer.ADADELTA:
        optimizer = torch.optim.Adadelta(params, lr=lr)
    elif optimizer_name == Optimizer.ADAMW:
        optimizer = torch.optim.AdamW(params, lr=lr)
    elif optimizer_name == Optimizer.ADAMAX:
        optimizer = torch.optim.Adamax(params, lr=lr)
    elif optimizer_name == Optimizer.ADAFACTOR:
        optimizer = torch.optim.Adafactor(params, lr=lr)
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. Available are: {', '.join(Optimizer.__members__.keys())}"
        )
    return optimizer
