from dataclasses import dataclass
from enum import Enum
from typing import Dict, TypeVar

T = TypeVar("T", bound=Enum)


class Loss(Enum):
    CROSS_ENTROPY = "cross_entropy"
    MSE = "mse"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    NLL = "nll"
    L1 = "l1"
    KL_DIVERGENCE = "kl_divergence"


class Optimizer(Enum):
    ADAM = "adam"
    SGD = "sgd"
    ASGD = "asgd"
    ADAGRAD = "adagrad"
    ADAMW = "adamw"
    ADAMAX = "adamax"
    ADADELTA = "adadelta"
    ADAFACTOR = "adafactor"


@dataclass
class Optimization:
    epochs: int
    optimizer: Optimizer
    learning_rate: float
    loss: Loss
    batch_size: int


def __get_enum_value(enum_class, name: str) -> T:
    try:
        return enum_class(name)
    except ValueError:
        available = ", ".join(e.value for e in enum_class)
        raise ValueError(f"Unknown {enum_class.__name__} type: {name}, available are: {available}")


def __get_loss(loss_name: str) -> Loss:
    return __get_enum_value(Loss, loss_name)


def __get_optimizer(optimizer_name: str) -> Optimizer:
    return __get_enum_value(Optimizer, optimizer_name)


def get_optimization(specification: dict) -> Optimization:
    optimization: Dict = specification.get("optimization", {})
    assert "loss" in optimization.keys(), "Loss function is required in optimization"
    assert "optimizer" in optimization.keys(), "Optimizer is required in optimization"
    loss: Loss = __get_loss(optimization["loss"])
    optimizer: Optimizer = __get_optimizer(optimization["optimizer"])

    return Optimization(
        epochs=optimization.get("epochs", 10),
        optimizer=optimizer,
        learning_rate=optimization.get("learning_rate", 0.001),
        loss=loss,
        batch_size=optimization.get("batch_size", 128),
    )
