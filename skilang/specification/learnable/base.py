from abc import ABC
from dataclasses import dataclass
from typing import Dict, List

from skilang.specification.learnable.enum import Backend, EncodingType


@dataclass
class Learnable(ABC):
    name: str
    dataset_name: str
    encodings: Dict[EncodingType, List[str]]
    backend: Backend = Backend.PYTORCH
