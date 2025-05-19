from itertools import islice
from typing import Tuple, Callable, Dict, Any, List

import torch
from torch import nn, optim, Tensor
from torchic.nn import NeuralNetwork
from torchic.nn.trainers import AbstractTrainer

from skilang import Formula
from skilang.__main__ import parse_specification
from skilang.specification.knowledge import get_rules_assignments, get_knowledge, Rule
from skilang.usecase.poker.dataset import train_loader, test_loader
from skilang.usecase.poker.model import model
from tests.resources import resource_path

specification: Dict = parse_specification(resource_path("poker-hand.yml"))


def regularization(input_batch: Tensor, target: Tensor) -> Tensor:
    base_assignments: Dict[str, Any] = {
        "suit1": 0,
        "rank1": 1,
        "suit2": 2,
        "rank2": 3,
        "suit3": 4,
        "rank3": 5,
        "suit4": 6,
        "rank4": 7,
        "suit5": 8,
        "rank5": 9,
        "hand": input_batch,
    }
    knowledge: Dict[str, Callable] = get_rules_assignments(specification, base_assignments)
    rules: List[Rule] = get_knowledge(specification)

    regularization_tensor: Tensor = torch.zeros(input_batch.shape[0]).to(input_batch.device)

    assignments = base_assignments.copy()
    for index, rule in enumerate(rules[:9]):
        formula: Formula = rule.clause
        rule_target = index + 1
        knowledge_so_far: Dict = dict(islice(knowledge.items(), index + 1))
        assignments.update(knowledge_so_far)
        respected_rule: Tensor = formula.evaluate(**assignments)
        unrespected_rule: Tensor = torch.logical_not(respected_rule)
        # logger.info("how many unrespected rules? %s", unrespected_rule.sum())
        apply_penalty: Tensor = (target == rule_target) & unrespected_rule
        # apply_advantage: Tensor = (target == rule_target) & respected_rule

        regularization_tensor[apply_penalty] = weight
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
        regularization_tensor = regularization(input_batch, target)
        modified_loss = loss + regularization_tensor
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


trainer.fit(train_loader, test_loader, loss, optimizer, epochs=10)

model.plot_loss()
model.save("model.pth")
