from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from skilang import Formula, parse
from skilang.utils import preprocess_batch_tensors


class ConstraintType(Enum):
    IMPLICATION = "if"
    DOUBLE_IMPLICATION = "iff"
    ALWAYS = "always"
    NEVER = "never"


@dataclass
class Constraint:
    type: ConstraintType
    clause: Formula
    condition: Optional[Formula] = None
    weight: float = 0.2


def get_constraints(specification: Dict) -> List[Constraint]:
    constraints: List[Constraint] = []
    constraints_spec = specification.get("constraints", [])
    for constraint in constraints_spec:
        type: Optional[ConstraintType] = next((ct for ct in ConstraintType if constraint.get(ct.value)), None)

        if type is None:
            raise ValueError("Incorrect constraint type")

        if type == ConstraintType.IMPLICATION or type == ConstraintType.DOUBLE_IMPLICATION:
            condition = constraint.get(type.value)
            clause = constraint.get("then")
        else:
            condition = None
            clause = constraint.get(type.value)

        weight = constraint.get("weight", 0.2)
        constraints.append(
            Constraint(
                type=type,
                clause=parse(preprocess_batch_tensors(clause)),
                condition=parse(preprocess_batch_tensors(condition)) if condition else None,
                weight=weight,
            )
        )

    return constraints
