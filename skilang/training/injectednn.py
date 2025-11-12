from typing import Callable

import lightning as L
import mlflow
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer as TorchOptimizer
from torchic.nn import NeuralNetwork
from torchmetrics import Accuracy, F1Score, Metric
from typing import List


class InjectedNN(L.LightningModule):
    def __init__(
        self,
        nn: NeuralNetwork,
        loss_fn: Callable,
        optimizer: TorchOptimizer,
        regularization_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
        metrics: List[Metric],
    ) -> None:
        super().__init__()
        self.model = nn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.regularization_fn = regularization_fn
        self.metrics = metrics

    def forward(self, inputs: Tensor):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs: Tensor
        target: Tensor
        inputs, target = batch
        pred: Tensor = self.forward(inputs)
        loss: Tensor = self.loss_fn(pred, target)

        regularization_tensor: Tensor = self.regularization_fn(inputs, pred, target)
        modified_loss: Tensor = torch.zeros_like(regularization_tensor).to(loss.device)
        if loss.dim() == regularization_tensor.dim():
            modified_loss += loss + regularization_tensor
        else:
            modified_loss += loss.mean() + regularization_tensor
        # Backpropagation
        # if loss is not reduced to a scalar
        if modified_loss.dim() != 0:
            modified_loss = modified_loss.mean()
        if loss.dim() != 0:
            loss = loss.mean()

        for metric_fn in self.metrics:
            metric = metric_fn.update(pred, target)

        log_dict: Dict = {"train_injected_loss": modified_loss, "train_raw_loss": loss}
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return modified_loss

    def on_train_epoch_end(self) -> None:
        log_dict = {}
        for metric_fn in self.metrics:
            metric = metric_fn.compute()
            log_dict[metric_fn.__class__.__name__] = metric
            metric_fn.reset()

        mlflow.log_metrics(log_dict, step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def configure_optimizers(self):
        return self.optimizer

    def _shared_eval(self, batch, batch_idx, prefix):
        inputs: Tensor
        target: Tensor
        inputs, target = batch
        pred: Tensor = self(inputs)
        loss: Tensor = self.loss_fn(pred, target)

        regularization_tensor = self.regularization_fn(inputs, pred, target)
        modified_loss: Tensor = torch.zeros_like(regularization_tensor).to(loss.device)
        if loss.dim() == regularization_tensor.dim():
            modified_loss += loss + regularization_tensor
        else:
            modified_loss += loss.mean() + regularization_tensor
        if modified_loss.dim() != 0:
            modified_loss = modified_loss.mean()
        if loss.dim() != 0:
            loss = loss.mean()

        log_dict: Dict = {f"{prefix}_injected_loss": modified_loss, f"{prefix}_raw_loss": loss}
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return modified_loss
