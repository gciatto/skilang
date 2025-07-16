from typing import Callable, Tuple

import torch
from torch import Tensor
from torchic.nn import NeuralNetwork
from torchic.nn.trainers import AbstractTrainer


class SkiTrainer(AbstractTrainer):
    def __init__(
        self, model: NeuralNetwork, regularization_fn: Callable[[Tensor, Tensor, Tensor], Tensor]
    ) -> None:
        super().__init__(model)
        self.regularization_fn: Callable[[Tensor, Tensor, Tensor], Tensor] = regularization_fn

    def train_step(
        self, input_batch: Tensor, target: Tensor, loss_fn: Callable
    ) -> Tuple[Tensor, torch.types.Number]:
        # Compute prediction error
        pred: Tensor = self.model(input_batch)
        loss: Tensor = loss_fn(pred, target)
        regularization_tensor: Tensor = self.regularization_fn(input_batch, pred, target)
        modified_loss = loss + regularization_tensor
        # Backpropagation
        # if loss is not reduced to a scalar
        if modified_loss.dim() != 0:
            modified_loss = modified_loss

        modified_loss.backward()
        return pred, modified_loss.item()

    def eval_step(
        self, input_batch: Tensor, target: Tensor, loss_fn: Callable
    ) -> Tuple[Tensor, torch.types.Number]:
        pred: Tensor = self.model(input_batch)
        loss: Tensor = loss_fn(pred, target)
        modified_loss = loss + self.regularization_fn(input_batch, pred, target)
        if modified_loss.dim() != 0:
            modified_loss = modified_loss
        return pred, modified_loss.item()
