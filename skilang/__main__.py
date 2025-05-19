import argparse
from pathlib import Path
from typing import Dict, List

import yaml
from torchic.nn import NeuralNetwork

from skilang.specification.constraints import Constraint, get_constraints
from skilang.specification.data import Dataset, get_datasets
from skilang.specification.knowledge import get_knowledge, Rule
from skilang.specification.learnable import get_learnables, Learnable
from skilang.specification.optimization import Optimization, get_optimization
from skilang.training import start_training


def parse_specification(file: Path) -> Dict:
    return yaml.load(open(file), Loader=yaml.FullLoader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-file", type=Path, required=True, help="Path to specification file")
    args = parser.parse_args()

    if not args.spec_file.is_file():
        raise FileNotFoundError(f"File not found: {args.spec_file}")

    print(f"Reading from: {args.spec_file}")
    specification: Dict = parse_specification(args.spec_file)
    datasets: List[Dataset] = get_datasets(specification)
    optimization: Optimization = get_optimization(specification)
    learnables: List[Learnable] = get_learnables(specification)
    knowledge: List[Rule] = get_knowledge(specification)
    constraints: List[Constraint] = get_constraints(specification)
    trained_models: List[NeuralNetwork] = start_training(
        datasets, optimization, learnables, knowledge, constraints
    )
    for model in trained_models:
        # print(f"Trained model: {model}")
        # print(f"Model parameters: {model.parameters()}")
        model.save(f"PokerHand.pth")


if __name__ == "__main__":
    main()
