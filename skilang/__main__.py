import argparse
import os
from pathlib import Path
from typing import Dict, List

import yaml
from torchic.nn import NeuralNetwork

from skilang.specification.constraints import Constraint, get_constraints
from skilang.specification.data import Dataset, get_datasets
from skilang.specification.knowledge import get_knowledge, Rule
from skilang.specification.learnable import get_learnables, Learnable
from skilang.specification.optimization import Optimization, get_optimization
from skilang.training import instantiate_models, start_training
from skilang.training.encodings import encode_dataset
from sklearn.metrics import confusion_matrix


def parse_specification(file: Path) -> Dict:
    return yaml.load(open(file), Loader=yaml.FullLoader)


parser = argparse.ArgumentParser()
parser.add_argument("--spec", type=Path, required=True, help="Path to skilang specification file")
parser.add_argument("--load", type=Path, required=False, help="Path to model weights file")


def train(
        spec_file_path: Path,
        specification: Dict,
        datasets: List[Dataset],
        optimization: Optimization,
        learnables: List[Learnable],
        knowledge: List[Rule],
        constraints: List[Constraint]):
    spec_file_dir: Path = spec_file_path.parent
    trained_models: List[NeuralNetwork] = start_training(
        datasets, optimization, learnables, knowledge, constraints
    )
    for learnable, model in zip(learnables, trained_models):
        relative_path = specification["learnable"][learnable.name]["destination"]
        dest_path = spec_file_dir / relative_path / f"{learnable.name}.pth"
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        model.save(dest_path)


def inference(model_file_path: Path, datasets: List[Dataset], learnables: List[Learnable]):
    def get_dataset_by_name(name: str) -> Dataset:
        for dataset in datasets:
            if dataset.name == name:
                return dataset
        raise KeyError(f"Dataset with name {name} not found.")    

    models: List[NeuralNetwork] = instantiate_models(*learnables)
    datasets_per_model = [get_dataset_by_name(learnable.dataset_name) for learnable in learnables]
    for learnable, model, dataset in zip(learnables, models, datasets_per_model):
        targets = [t.name for t in dataset.targets]
        encodings = learnable.encodings
        if len(encodings) != 0:
            dataset, _ = encode_dataset(dataset, encodings)
        model.load(model_file_path)
        X = dataset.test.drop(targets, axis=1)
        y = dataset.test[targets]
        results = model.inference(X.values)
        y_pred = results.tensor.argmax(dim=1).cpu().numpy().round()
        y_true = y.values.argmax(axis=1)
        values = sorted(dataset.targets[0].values.items(), key=lambda x: x[1])
        labels = [value[0] for value in values]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print(f"Confusion Matrix for {learnable.name}:")
        print(cm)
        return cm


def main():
    args = parser.parse_args()

    if not args.spec.is_file():
        raise FileNotFoundError(f"File not found: {args.spec}")
    
    if args.load and not args.load.is_file():
        raise FileNotFoundError(f"File not found: {args.load}")
    
    mode = "train"
    spec_file_path: Path = args.spec.resolve()
    print(f"Reading from: {spec_file_path}")

    if args.load:
        mode = "inference"
        model_file_path: Path = args.load.resolve()
        print(f"Loading model weights from: {model_file_path}")

    specification: Dict = parse_specification(spec_file_path)
    datasets: List[Dataset] = get_datasets(specification, spec_file_path.parent)
    optimization: Optimization = get_optimization(specification)
    learnables: List[Learnable] = get_learnables(specification)
    knowledge: List[Rule] = get_knowledge(specification)
    constraints: List[Constraint] = get_constraints(specification)
    if mode == "train":
        train(spec_file_path, specification, datasets, optimization, learnables, knowledge, constraints)
    elif mode == "inference":
        inference(model_file_path, datasets, learnables)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'train' or 'inference'.")


if __name__ == "__main__":
    main()
