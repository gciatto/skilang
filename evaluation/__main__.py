import fire
from pathlib import Path

import numpy as np

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


def main(spec_file: str, protected_idx: int = 8, population: int = 5):
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, f1_score

    spec_file_path: Path = PROJECT_ROOT / spec_file
    if not spec_file_path.is_file():
        raise FileNotFoundError(f"File not found: {spec_file_path}")

    spec_file_dir: Path = spec_file_path.parent
    print(f"Reading from: {spec_file_path}")
    specification: Dict = parse_specification(spec_file_path)
    datasets: List[Dataset] = get_datasets(specification, spec_file_dir)
    optimization: Optimization = get_optimization(specification)
    learnables: List[Learnable] = get_learnables(specification)

    subfolders = ["educated", "uneducated"]
    colors = {"educated": "blue", "uneducated": "orange"}

    metrics = {
        "Statistical Parity": {"educated": [], "uneducated": []},
        "Accuracy": {"educated": [], "uneducated": []},
        "F1 Score": {"educated": [], "uneducated": []}
    }

    for learnable in learnables:
        for subfolder in subfolders:
            dataset: Dataset = next(dataset for dataset in datasets if dataset.name == learnable.dataset_name)
            encodings: Dict[str, EncodingType] = learnable.encodings

            if len(encodings) != 0:
                dataset, _ = encode_dataset(dataset, encodings)

            test_loader: DataLoader = create_torch_dataloader(dataset.test, optimization.batch_size)

            sp_values = []
            accuracy_values = []
            f1_values = []

            for i in range(population):
                model_name = f"{subfolder}/{learnable.name}_seed_{i}"
                trained_model = load_model(model_name)

                predictions = []
                protected_values = []
                true_classes = []
                with torch.no_grad():
                    for batch in test_loader:
                        inputs, targets = batch
                        inputs = inputs.to(trained_model.device())
                        outputs = trained_model(inputs)
                        predictions.append(outputs)
                        true_classes.append(targets)
                        protected_values.append(inputs[:, protected_idx])

                predictions = torch.cat(predictions, dim=0)
                protected_values = torch.cat(protected_values, dim=0)
                true_classes = torch.cat(true_classes, dim=0)
                predicted_classes = predictions.argmax(dim=1)

                sp_values.append(statistical_parity(predicted_classes, protected_values))
                accuracy_values.append(accuracy_score(true_classes.cpu(), predicted_classes.cpu()))
                f1_values.append(f1_score(true_classes.cpu(), predicted_classes.cpu(), average="weighted"))

            metrics["Statistical Parity"][subfolder].extend(sp_values)
            metrics["Accuracy"][subfolder].extend(accuracy_values)
            metrics["F1 Score"][subfolder].extend(f1_values)

    plt.figure(figsize=(12, 8))
    for i, (metric_name, group_values) in enumerate(metrics.items()):
        plt.subplot(1, 3, i + 1)

        educated_data = group_values["educated"]
        uneducated_data = group_values["uneducated"]

        boxplot = plt.boxplot(
            [educated_data, uneducated_data],
            patch_artist=True,
            boxprops=dict(color="black"),
            medianprops=dict(color='black')
        )

        colors_list = [colors["educated"], colors["uneducated"]]
        for patch, color in zip(boxplot['boxes'], colors_list):
            patch.set_facecolor(color)

        plt.xticks([1, 2], ['Educated', 'Uneducated'])
        plt.title(metric_name)

    plt.suptitle("Metrics Comparison Between Educated and Uneducated")
    plot_path = RESULTS_PATH / "metrics_comparison_boxplot"
    plt.savefig(str(plot_path) + ".png")
    plt.savefig(str(plot_path) + ".pdf")
    plt.close()


if __name__ == "__main__":
    fire.Fire(main)
