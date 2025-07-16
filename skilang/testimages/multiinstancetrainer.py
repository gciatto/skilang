# from typing import Callable, Tuple
#
# import torch
# from torch import Tensor
# from torchic.nn import NeuralNetwork
# from torchic.nn.trainers import AbstractTrainer
#
# from torch.optim import Optimizer
#
#
# class MultiInstanceTrainer(AbstractTrainer):
#
#     def __init__(
#         self, model: NeuralNetwork, regularization_fn: Callable[[Tensor, Tensor, Tensor], Tensor]
#     ) -> None:
#         super().__init__(model)
#         self.regularization_fn: Callable[[Tensor, Tensor, Tensor], Tensor] = regularization_fn
#
#     def __train(
#         self,
#         dataloader,
#         loss_fn: Callable[[Tensor, Tensor], Tensor],
#         optimizer: Optimizer,
#     ) -> None:
#         size = len(dataloader.dataset)
#         num_batches = len(dataloader)
#         batch_loss: float = 0.0
#         self.model.train()
#         for batch, (X_instances, y_instances) in enumerate(dataloader):
#             assert len(X_instances) == len(y_instances), "Input and target tensors must have the same length."
#             num_instances = len(X_instances)
#             loss: float = 0.0
#             for X, y in zip(X_instances, y_instances):
#                 input_batch, target = X.to(self.model.device()), y.to(self.model.device())
#                 pred, loss = self.train_step(input_batch, target, loss_fn)
#
#
#             # regularization_tensor: Tensor = self.regularization_fn(input_batches, pred, target)
#             # modified_loss = loss + regularization_tensor
#             # Backpropagation
#             # if loss is not reduced to a scalar
#             if modified_loss.dim() != 0:
#                 modified_loss = modified_loss.sum()
#
#             modified_loss.backward()
#
#             batch_loss += loss
#             optimizer.step()
#             optimizer.zero_grad()
#
#             if batch % 100 == 0:
#                 loss_value, current = loss, (batch + 1) * num_instances
#                 print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
#
#         avg_loss: float = batch_loss / num_batches
#         self.model.train_losses.append(avg_loss)
#
#     def __test(self, dataloader, loss_fn: Callable[[Tensor, Tensor], Tensor]) -> None:
#         size: int = len(dataloader.dataset)
#         num_batches = len(dataloader)
#         self.model.eval()
#         test_loss: float = 0
#         correct_predictions: int = 0
#         with torch.no_grad():
#             for X, y in dataloader:
#                 input_batch, target = (
#                     X.to(self.model.device()),
#                     y.to(self.model.device()),
#                 )
#                 pred, loss = self.eval_step(input_batch, target, loss_fn)
#                 test_loss = test_loss + loss
#                 predictions_tensor: Tensor = pred.argmax(1) == target
#                 correct_predictions = correct_predictions + int(
#                     predictions_tensor.int().sum().item()
#                 )
#
#         avg_loss: float = test_loss / num_batches
#         self.model.test_losses.append(avg_loss)
#         accuracy: float = 100 * correct_predictions / size
#         print(
#             f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {avg_loss:>8f} \n"
#         )
#
#     def train_step(
#             self, input_batch: Tensor, target: Tensor, loss_fn: Callable
#     ) -> Tuple[Tensor, torch.types.Number]:
#         # Compute prediction error
#         pred: Tensor = self.model(input_batch)
#         loss: Tensor = loss_fn(pred, target)
#         return pred, loss.item()
