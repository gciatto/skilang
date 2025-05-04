from typing import Tuple, Callable, List

import torch
from torch import nn, optim, Tensor
from torchic.nn import NeuralNetwork
from torchic.nn.trainers import AbstractTrainer

from skilang import Formula, parse
from skilang.usecase.poker.dataset import train_loader, test_loader
from skilang.usecase.poker.model import model

rules: List[str] = [
    (
        "hand[rank1] == hand[rank2] or hand[rank1] == hand[rank3] or hand[rank1] == hand[rank4] or hand[rank1] == hand[rank5] or "
        "hand[rank2] == hand[rank3] or hand[rank2] == hand[rank4] or hand[rank2] == hand[rank5] or hand[rank3] == hand[rank4] or "
        "hand[rank3] == hand[rank5] or hand[rank4] == hand[rank5]"
    ),
    (
        "(hand[rank1] == hand[rank2] and (hand[rank3] == hand[rank4] or hand[rank3] == hand[rank5] or hand[rank4] == hand[rank5])) or "
        "(hand[rank1] == hand[rank3] and (hand[rank2] == hand[rank4] or hand[rank2] == hand[rank5] or hand[rank4] == hand[rank5])) or "
        "(hand[rank1] == hand[rank4] and (hand[rank2] == hand[rank3] or hand[rank2] == hand[rank5] or hand[rank3] == hand[rank5])) or "
        "(hand[rank1] == hand[rank5] and (hand[rank2] == hand[rank3] or hand[rank2] == hand[rank4] or hand[rank3] == hand[rank4]))"
    ),
]

rules = list(map(lambda rule: rule.replace("[", "[:, "), rules))


def regularization(input_batch: Tensor, pred: Tensor, target: Tensor) -> Tensor:
    assignments: dict = {"rank1": 1, "rank2": 3, "rank3": 5, "rank4": 7, "rank5": 9, "hand": input_batch}
    regularization_tensor: Tensor = torch.ones(input_batch.shape[0]).to(input_batch.device)

    for index, rule in enumerate(rules):
        formula: Formula = parse(rule)
        rule_target = index + 1
        respected_rule: Tensor = formula.evaluate(**assignments)
        unrespected_rule: Tensor = torch.bitwise_not(respected_rule)
        # logger.info("how many unrespected rules? %s", unrespected_rule.sum())
        apply_penalty: Tensor = (target == rule_target) & unrespected_rule
        # apply_advantage: Tensor = (target == rule_target) & respected_rule

        regularization_tensor[apply_penalty] = 1.5
        # regularization_tensor[apply_advantage] = 0.5
    return regularization_tensor


class SkiTrainer(AbstractTrainer):
    def __init__(self, model: NeuralNetwork) -> None:
        super().__init__(model)

    def train_step(
        self, input_batch: Tensor, target: Tensor, loss_fn: Callable
    ) -> Tuple[Tensor, torch.types.Number]:
        # Compute prediction error
        pred: Tensor = self.model(input_batch)
        loss: Tensor = loss_fn(pred, target)
        regularization_tensor = regularization(input_batch, pred, target)
        modified_loss = loss * regularization_tensor
        # Backpropagation
        # if loss is not reduced to a scalar
        if modified_loss.dim() != 0:
            modified_loss = modified_loss.mean()

        modified_loss.backward()
        return pred, modified_loss.item()

    def eval_step(
        self, input_batch: Tensor, target: Tensor, loss_fn: Callable
    ) -> Tuple[Tensor, torch.types.Number]:
        pred: Tensor = self.model(input_batch)
        loss: Tensor = loss_fn(pred, target)
        if loss.dim() != 0:
            loss = loss.mean()
        return pred, loss.item()


trainer = SkiTrainer(model)

loss = nn.CrossEntropyLoss(reduction="none")
optimizer = optim.Adam(model.parameters(), lr=0.001)


trainer.fit(train_loader, test_loader, loss, optimizer, epochs=2)

model.plot_loss()
model.save("model.pth")
