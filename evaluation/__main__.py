import fire
import os
from pathlib import Path
from evaluation.results import PATH as RESULTS_PATH
from typing import Dict, List
import torch
import yaml
from torch.utils.data import DataLoader
from evaluation import load_model, statistical_parity
from skilang.specification.data import Dataset, get_datasets
from skilang.specification.learnable import get_learnables, Learnable, EncodingType
from skilang.specification.learnable.impl import create_torch_dataloader
from skilang.specification.optimization import Optimization, get_optimization
from skilang.training import encode_dataset


CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent


def parse_specification(file: Path) -> Dict:
    return yaml.load(open(file), Loader=yaml.FullLoader)


def main(spec_file: str, protected_idx: int = 8):
    spec_file_path: Path = PROJECT_ROOT / spec_file
    if not spec_file_path.is_file():
        raise FileNotFoundError(f"File not found: {spec_file_path}")

    spec_file_dir: Path = spec_file_path.parent
    print(f"Reading from: {spec_file_path}")
    specification: Dict = parse_specification(spec_file_path)
    datasets: List[Dataset] = get_datasets(specification, spec_file_dir)
    optimization: Optimization = get_optimization(specification)
    learnables: List[Learnable] = get_learnables(specification)

    for learnable in learnables:
        dataset: Dataset = next(dataset for dataset in datasets if dataset.name == learnables[0].dataset_name)
        encodings: Dict[str, EncodingType] = learnable.encodings
        trained_model = load_model(learnable.name)

        if len(encodings) != 0:
            dataset, _ = encode_dataset(dataset, encodings)

        test_loader: DataLoader = create_torch_dataloader(dataset.test, optimization.batch_size)

        # collect the predictions of the model on the test set
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs = inputs.to(trained_model.device())
                outputs = trained_model(inputs)
                predictions.append(outputs)
        # Reshape predictions to a single tensor
        predictions = torch.cat(predictions, dim=0)

        # Save the predictions to a file
        # THe file must be a csv with 3 columns:
        # - value of the protected attribute
        # - predicted class
        # - true class
        dest_path = RESULTS_PATH / f"{learnable.name}_predictions.csv"
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, 'w') as f:
            f.write("protected_attribute,predicted_class,true_class\n")
            for i, (pred, target) in enumerate(zip(predictions, dataset.test["income"])):
                protected_value = dataset.test.iloc[i, protected_idx].item()
                predicted_class = pred.argmax()
                f.write(f"{protected_value},{predicted_class},{target}\n")

        # Print statistical parity
        statistical_parity_diff = statistical_parity(predictions, dataset.test.iloc[:, protected_idx])
        print(f"Statistical parity difference for {learnable.name}: {statistical_parity_diff:.4f}")


if __name__ == "__main__":
    fire.Fire(main)