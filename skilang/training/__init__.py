from itertools import islice
from typing import List, Dict, Callable, Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchic.nn import NeuralNetwork

from skilang import Formula
from skilang.specification.constraints import Constraint, ConstraintType
from skilang.specification.data import Dataset
from skilang.specification.knowledge import Rule, get_rules_assignments
from skilang.specification.learnable import Learnable, Backend
from skilang.specification.learnable.impl import create_torch_model, create_torch_dataloader
from skilang.specification.optimization import Optimization
from skilang.specification.optimization.impl import get_torch_loss, get_torch_optimizer
from skilang.training.skitrainer import SkiTrainer


def start_training(
    datasets: List[Dataset],
    optimization: Optimization,
    learnables: List[Learnable],
    knowledge: List[Rule],
    constraints: List[Constraint],
) -> List[NeuralNetwork]:
    """
    Start the training process.
    :return: The trained model.
    """
    trained_models: List[NeuralNetwork] = []
    for learnable in learnables:
        dataset: Dataset = next(dataset for dataset in datasets if dataset.name == learnable.dataset_name)

        if learnable.backend == Backend.PYTORCH:
            model: NeuralNetwork = create_torch_model(learnable)
            trained_model = train_torch_model(
                learnable.name, model, dataset, optimization, knowledge, constraints
            )
            trained_models.append(trained_model)
        else:
            raise ValueError(f"Unsupported backend: {learnable.backend}")

    return trained_models


def train_torch_model(
    model_name: str,
    model: NeuralNetwork,
    dataset: Dataset,
    optimization: Optimization,
    knowledge: List[Rule],
    constraints: List[Constraint],
) -> NeuralNetwork:
    train_loader: DataLoader = create_torch_dataloader(dataset.training, optimization.batch_size)
    test_loader: DataLoader = create_torch_dataloader(dataset.test, optimization.batch_size)

    dataset_assignments: Dict[str, int] = {}
    for feature in dataset.features:
        dataset_assignments[feature.name] = feature.column

    for target_value in dataset.target.values:
        dataset_assignments[target_value] = dataset.target.get_mapped_value(target_value)


    target_name: str = dataset.target.name
    instance_name: str = dataset.instance_name

    ski_trainer = SkiTrainer(
        model, create_regularization_fn(model_name, instance_name, target_name,knowledge, constraints, dataset_assignments)
    )
    loss = get_torch_loss(optimization.loss)
    optimizer = get_torch_optimizer(optimization.optimizer, model.parameters(), optimization.learning_rate)
    epochs = optimization.epochs
    ski_trainer.fit(train_loader, test_loader, loss, optimizer, epochs=epochs)
    model.plot_loss()
    model.save("model.pth")
    return model


def create_regularization_fn(
    model_name: str,
    instance_name: str,
    target_name: str,
    knowledge: List[Rule],
    constraints: List[Constraint],
    dataset_assignments: Dict[str, Any],
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    def regularization_fn(input_batch: Tensor, pred: Tensor, target: Tensor) -> Tensor:
        batch_assignments: Dict[str, Any] = {
            target_name: target,
            instance_name: input_batch,
        }

        regularization_tensor: Tensor = torch.zeros(input_batch.shape[0]).to(input_batch.device)

        # print("ACTUAL ", pred.size(), "ID", id(pred))

        def create_model_output_fn(pred_snapshot):
            def model_output_fn(batch_snapshot):
                # print(batch_snapshot.size())
                # print("SNAP", pred_snapshot.size(), "ID", id(pred_snapshot))
                return pred_snapshot.argmax(dim=1)

            return model_output_fn
        # Add the model output to the assignments
        batch_assignments.update({model_name: create_model_output_fn(pred)})

        # assignments = base_assignments.copy()
        # assignments.update(batch_assignments)
        # assignments.update(rules_assignments)

        rules_assignments: Dict[str, Callable] = get_rules_assignments(knowledge, dataset_assignments | batch_assignments)
        assignments: Dict[str, Callable] = dataset_assignments | batch_assignments | rules_assignments
        # print(input_batch.size())

        for index, constraint in enumerate(constraints):
            # knowledge_so_far: Dict = dict(islice(rules_assignments.items(), index + 1))
            # assignments.update(knowledge_so_far)

            respected_constraint = constraint.clause.evaluate(**assignments)

            respected_condition: Tensor = torch.tensor([])

            if constraint.type == ConstraintType.NEVER:
                respected_constraint = torch.logical_not(respected_constraint)

            if constraint.condition is not None:
                respected_condition = constraint.condition.evaluate(**assignments)

            unrespected_constraint: Tensor = torch.logical_not(respected_constraint)

            if respected_condition.nelement() > 0:
                if constraint.type == ConstraintType.IMPLICATION:
                    apply_penalty: Tensor = torch.logical_and(respected_condition, unrespected_constraint)
                # elif constraint.type == ConstraintType.DOUBLE_IMPLICATION:
                #     apply_penalty: Tensor = torch.logical_xor(respected_condition, unrespected_constraint)
                else:
                    raise ValueError(f"Unsupported constraint type: {constraint.type}")
            else:
                apply_penalty: Tensor = unrespected_constraint

            multiplier: float = 10
            regularization_tensor[apply_penalty] += multiplier * constraint.weight

            # logger.info("how many unrespected rules? %s", unrespected_constraint.sum())
            # apply_penalty: Tensor = (target == rule_target) & unrespected_constraint
            # apply_advantage: Tensor = (target == rule_target) & respected_constraint

            # regularization_tensor[apply_advantage] = 0.5
        return regularization_tensor

    return regularization_fn


