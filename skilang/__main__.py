from pathlib import Path
from typing import Dict, List

import fire
import yaml
from torchic.nn import NeuralNetwork

from skilang import logger
from skilang.specification.constraints import Constraint, get_constraints
from skilang.specification.data import Dataset, get_datasets
from skilang.specification.knowledge import get_knowledge, Rule
from skilang.specification.learnable import get_learnables, Learnable
from skilang.specification.optimization import Optimization, get_optimization
from skilang.training import start_training


def parse_specification(file: Path) -> Dict:
    return yaml.load(open(file), Loader=yaml.FullLoader)


def main(spec_file: Path, population: int = 30, seed: int = 0):
    spec_file = Path(spec_file).resolve()
    if not spec_file.is_file():
        raise FileNotFoundError(f"File not found: {spec_file}")

    spec_file_path: Path = spec_file.resolve()
    spec_file_dir: Path = spec_file_path.parent
    print(f"Reading from: {spec_file_path}")

    specification: Dict = parse_specification(spec_file_path)
    datasets: List[Dataset] = get_datasets(specification, spec_file_dir)
    optimization: Optimization = get_optimization(specification)
    learnables: List[Learnable] = get_learnables(specification)
    knowledge: List[Rule] = get_knowledge(specification)
    constraints: List[Constraint] = get_constraints(specification)

    for i in range(population):
        current_seed = seed + i
        print(f"Training iteration {i + 1}/{population} with seed {current_seed}")
        trained_models: List[NeuralNetwork] = start_training(
            datasets, optimization, learnables, knowledge, constraints, seed=current_seed
        )
        logger.info(f"Trained models: {trained_models}")


if __name__ == "__main__":
    fire.Fire(main)
