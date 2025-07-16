from dataclasses import dataclass, field
from typing import List, Optional, Dict

from skilang.specification.learnable.base import Learnable, Backend
from skilang.specification.learnable.enum import LayerType, Activation, Regularization, EncodingType


@dataclass
class Layer:
    type: LayerType
    input_size: int
    output_size: int
    activation: Optional[Activation] = None
    regularization: Optional[Regularization] = None
    dropout: Optional[float] = None
    kernel_size: Optional[int] = None


@dataclass
class FeedForward(Learnable):
    layers: List[Layer] = field(default_factory=list)


def create_feed_forward(
    name: str,
    dataset_name: str,
    backend: Backend,
    encodings: Dict[EncodingType, List[str]],
    layers: List[Layer],
) -> FeedForward:
    return FeedForward(
        name=name,
        dataset_name=dataset_name,
        encodings=encodings,
        backend=backend,
        layers=layers,
    )
