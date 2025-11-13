import os
from typing import List, Dict, Callable, Any

import mlflow
import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from mlflow.models import infer_signature
from torch import Tensor
from torch.utils.data import DataLoader
from torchic.nn import NeuralNetwork
from torchic.utils import get_current_device
from torchinfo import summary
from torchmetrics import Metric, Accuracy, Recall

from skilang.specification.constraints import Constraint, ConstraintType
from skilang.specification.data import Dataset
from skilang.specification.knowledge import Rule, get_rules_assignments
from skilang.specification.learnable import Learnable, Backend
from skilang.specification.learnable.enum import EncodingType
from skilang.specification.learnable.impl import create_torch_model, create_torch_dataloader
from skilang.specification.optimization import Optimization
from skilang.specification.optimization.impl import get_torch_loss, get_torch_optimizer
from skilang.training.encodings import encode_dataset
from skilang.training.injectednn import InjectedNN


def start_training(
    datasets: List[Dataset],
    optimization: Optimization,
    learnables: List[Learnable],
    knowledge: List[Rule],
    constraints: List[Constraint],
    seed: int = 0,
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
                learnable.name, model, dataset, encodings, optimization, knowledge, constraints, seed
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
    seed: int = 0,
) -> NeuralNetwork:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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

    loss = get_torch_loss(optimization.loss)
    optimizer = get_torch_optimizer(optimization.optimizer, model.parameters(), optimization.learning_rate)
    epochs = optimization.epochs
    regularization_fn = create_regularization_fn(
        model_name, instance_name, target_name, knowledge, constraints, dataset_assignments
    )

    # find distinct value in dataset.targets
    target_values = set(dataset.targets[0].values.values())
    num_classes: int = len(target_values)

    metrics: List[Metric] = [
        Recall(task="multiclass", num_classes=num_classes).to(get_current_device()),
        Accuracy(task="multiclass", num_classes=num_classes).to(get_current_device()),
        # F1Score(task="multiclass", num_classes=num_classes).to(get_current_device()),
    ]

    ski_model = InjectedNN(model, loss, optimizer, regularization_fn, metrics)

    # sets seeds for numpy, torch and python.random.
    seed_everything(seed, workers=True)
    trainer = Trainer(deterministic=True, max_epochs=epochs)

    mlflow.pytorch.autolog(log_models=False)
    mlflow.set_experiment(model_name)
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(5)

    with mlflow.start_run() as run:
        trainer.fit(ski_model, train_loader, test_loader)
        tensor_example: Tensor = next(iter(test_loader))[0][0, :]
        column_tensor_example: Tensor = tensor_example.reshape(1, -1)
        input_example: np.ndarray = column_tensor_example.numpy()
        output_example = model.inference(column_tensor_example).tensor.cpu().numpy()
        signature = infer_signature(input_example, output_example)
        model_info = mlflow.pytorch.log_model(
            model, name=f"{run.info.run_name}_model_{seed}", signature=signature, input_example=input_example
        )
        model_uri = model_info.model_uri
        # print(model_info)
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")
        os.remove("model_summary.txt")

        # Load and use the model
        loaded_model = mlflow.pyfunc.load_model(model_uri)

        # Make predictions
        predictions = loaded_model.predict(input_example)
        print("Predictions:", predictions)

    return ski_model.model


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
        regularization_scalar: Tensor = torch.tensor(0.0).to(input_batch.device)

        def straight_through(x, tau=0.5):
            return (x > tau).float() + (x - x.detach())

        def create_model_output_fn(pred_snapshot):
            # return lambda batch_snapshot: pred_snapshot.argmax(dim=1)
            return lambda batch_snapshot: straight_through(pred_snapshot.softmax(dim=1)[:, 1])

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

            if respected_constraint.dim() == 0:
                regularization_scalar += constraint.weight * respected_constraint

            elif respected_constraint.dim() == 1:
                d = respected_constraint.shape[0]
                regularization_scalar += constraint.weight * (d - respected_constraint.sum()) / d

            else:
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

        if torch.zeros_like(regularization_scalar):
            # print("regularization_tensor.requires_grad =", regularization_tensor.requires_grad)
            # print("regularization_tensor.grad_fn =", regularization_tensor.grad_fn)
            # print("regularization_tensor is leaf =", regularization_tensor.is_leaf)
            return regularization_tensor
        else:
            # print("regularization_scalar.requires_grad =", regularization_scalar.requires_grad)
            # print("regularization_scalar.grad_fn =", regularization_scalar.grad_fn)
            # print("regularization_scalar is leaf =", regularization_scalar.is_leaf)
            return regularization_scalar

    return regularization_fn
