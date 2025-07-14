from abc import ABC
from dataclasses import dataclass
from typing import Dict

from skilang.specification.learnable.enum import Backend, EncodingType


@dataclass
class Learnable(ABC):
    name: str
    dataset_name: str
    encodings: Dict[str, EncodingType]
    backend: Backend = Backend.PYTORCH
