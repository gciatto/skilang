from typing import List, Dict, Callable, Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchic.nn import NeuralNetwork

from skilang.specification.constraints import Constraint, ConstraintType
from skilang.specification.data import Dataset
from skilang.specification.knowledge import Rule, get_rules_assignments
from skilang.specification.learnable import Learnable, Backend
from skilang.specification.learnable.enum import EncodingType
from skilang.specification.learnable.impl import create_torch_model, create_torch_dataloader
from skilang.specification.optimization import Optimization
from skilang.specification.optimization.impl import get_torch_loss, get_torch_optimizer
from skilang.training.encodings import encode_dataset
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
            encodings: Dict[EncodingType, List[str]] = learnable.encodings
            trained_model = train_torch_model(
                learnable.name, model, dataset, encodings, optimization, knowledge, constraints
            )
            trained_models.append(trained_model)
        else:
            raise ValueError(f"Unsupported backend: {learnable.backend}")

    return trained_models


def train_torch_model(
    model_name: str,
    model: NeuralNetwork,
    dataset: Dataset,
    encodings: Dict[EncodingType, List[str]],
    optimization: Optimization,
    knowledge: List[Rule],
    constraints: List[Constraint],
) -> NeuralNetwork:
    mappings: Dict[str, Dict[str, float]] = {}
    if len(encodings) != 0:
        dataset, mappings = encode_dataset(dataset, encodings)

    train_loader: DataLoader = create_torch_dataloader(dataset.training, optimization.batch_size)
    test_loader: DataLoader = create_torch_dataloader(dataset.test, optimization.batch_size)

    dataset_assignments: Dict[str, float] = {}

    if len(mappings) != 0:
        for feature_name, values in mappings.items():
            for value, encoded_value in values.items():
                dataset_assignments[value] = encoded_value

    for feature in dataset.features:
        dataset_assignments[feature.name] = feature.column

    for target in dataset.targets:
        for t_name, t_value in target.values.items():
            dataset_assignments[t_name] = t_value

    target_names: List[str] = [target.name for target in dataset.targets]
    # TODO: support multiple targets
    target_name: str = target_names[0]
    instance_name: str = dataset.instance_name

    ski_trainer = SkiTrainer(
        model,
        create_regularization_fn(
            model_name, instance_name, target_name, knowledge, constraints, dataset_assignments
        ),
    )
    loss = get_torch_loss(optimization.loss)
    optimizer = get_torch_optimizer(optimization.optimizer, model.parameters(), optimization.learning_rate)
    epochs = optimization.epochs
    ski_trainer.fit(train_loader, test_loader, loss, optimizer, epochs=epochs)
    model.plot_loss()
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

        def create_model_output_fn(pred_snapshot):
            return lambda batch_snapshot: pred_snapshot.argmax(dim=1)

        # Add the model output to the assignments
        batch_assignments.update({model_name: create_model_output_fn(pred)})
        stripped_dataset_assignments = {k.strip(): v for k, v in dataset_assignments.items()}

        rules_assignments: Dict[str, Callable] = get_rules_assignments(
            knowledge, stripped_dataset_assignments | batch_assignments
        )
        assignments: Dict[str, Callable] = (
            stripped_dataset_assignments | batch_assignments | rules_assignments
        )

        for index, constraint in enumerate(constraints):
            respected_constraint = constraint.clause.evaluate(**assignments)

            respected_condition: Tensor = torch.tensor([])

            if constraint.type == ConstraintType.NEVER:
                respected_constraint = torch.logical_not(respected_constraint)

            unrespected_constraint: Tensor = torch.logical_not(respected_constraint)

            if constraint.condition is not None:
                respected_condition = constraint.condition.evaluate(**assignments)

            apply_penalty: Tensor
            if respected_condition.nelement() > 0:
                if constraint.type == ConstraintType.IMPLICATION:
                    # A -> B that is NOT(A) OR B
                    # we want the penalty so: NOT(NOT(A) OR B) that is A AND NOT(B)
                    apply_penalty = torch.logical_and(respected_condition, unrespected_constraint)

                elif constraint.type == ConstraintType.DOUBLE_IMPLICATION:
                    # A <-> B that is NOT(A XOR B)
                    # we want the penalty so: NOT(NOT(A XOR B)) that is A XOR B
                    apply_penalty = torch.logical_xor(respected_condition, respected_constraint)
                else:
                    raise ValueError(f"Unsupported constraint type: {constraint.type}")
            else:
                apply_penalty = unrespected_constraint

            multiplier: float = 1
            regularization_tensor[apply_penalty] += multiplier * constraint.weight
        return regularization_tensor

    return regularization_fn
