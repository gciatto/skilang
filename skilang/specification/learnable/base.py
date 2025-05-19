from abc import ABC
from dataclasses import dataclass

from skilang.specification.learnable.enum import Backend


@dataclass
class Learnable(ABC):
    name: str
    dataset_name: str
    backend: Backend = Backend.PYTORCH
