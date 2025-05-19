from dataclasses import dataclass, field
from typing import List

from skilang.specification.learnable.base import Learnable, Backend
from skilang.specification.learnable.enum import LayerType, Activation, Regularization


@dataclass
class Layer:
    type: LayerType
    input_size: int
    output_size: int
    activation: Activation = None
    regularization: Regularization = None
    dropout: float = 0.0
    kernel_size: int = None


@dataclass
class FeedForward(Learnable):
    layers: List[Layer] = field(default_factory=list)


def create_feed_forward(name: str, dataset_name: str, backend: Backend, layers: List[Layer]) -> FeedForward:
    return FeedForward(
        name=name,
        dataset_name=dataset_name,
        backend=backend,
        layers=layers,
    )
