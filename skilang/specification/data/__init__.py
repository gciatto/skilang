from dataclasses import dataclass, field
from typing import List, Dict, Any

import pandas as pd

from tests.resources import resource_path


@dataclass
class Feature:
    name: str
    column: int
    values: List[str]
    mapping: Dict[str, int] = field(default_factory=dict)

    def get_mapped_value(self, input_value: Any) -> Any:
        if isinstance(input_value, str):
            return self.mapping.get(input_value, input_value)
        return input_value


class Target(Feature):
    pass


@dataclass
class Dataset:
    """
    Dataset class to hold the dataset information.

    :param name: name of dataset
    :param features: the list of features
    :param target: target feature
    :param training: DataFrame of training dataset
    :param test: DataFrame of test dataset
    """

    name: str
    instance_name: str
    features: List[Feature]
    target: Target
    training: pd.DataFrame
    test: pd.DataFrame
    columns: List[str] = field(init=False)

    def __post_init__(self):
        self.columns = self.training.columns.tolist()


def get_datasets(specification: Dict) -> List[Dataset]:
    datasets_spec: Dict = specification["data"]
    datasets: List[Dataset] = []
    for dataset_name, current_dataset in datasets_spec.items():
        features: List[Feature] = [
            Feature(name=feature["name"], column=feature["column"], values=feature["values"], mapping=feature["mapping"])
            for feature in current_dataset["features"]
        ]
        target: Target = Target(
            name=current_dataset["target"]["name"],
            column=current_dataset["target"]["column"],
            values=current_dataset["target"]["values"],
            mapping=current_dataset["target"]["mapping"],
        )

        training = pd.read_csv(resource_path(current_dataset["training"]["file"]))
        test = pd.read_csv(resource_path(current_dataset["test"]["file"]))

        columns: List[str] = [feature.name for feature in features]
        columns.append(target.name)

        training.columns = columns
        test.columns = columns

        datasets.append(
            Dataset(name=dataset_name, instance_name=current_dataset["instance_name"], features=features, target=target, training=training, test=test)
        )

    return datasets
