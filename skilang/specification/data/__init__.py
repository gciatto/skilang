import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urlparse

import pandas as pd
import requests
from sklearn.model_selection import train_test_split


class FeatureType(Enum):
    CATEGORICAL = "categorical"
    INTEGER = "int"
    FLOAT = "float"


@dataclass
class Feature:
    name: str
    column: int
    type: FeatureType
    values: Optional[Dict[str, int]] = field(default_factory=dict)


class Target(Feature):
    pass


@dataclass
class Dataset:
    """
    Dataset class to hold the dataset information.

    :param name: name of dataset
    :param features: the list of features
    :param targets: the list of target features
    :param training: DataFrame of training dataset
    :param test: DataFrame of test dataset
    """

    name: str
    instance_name: str
    features: List[Feature]
    targets: List[Target]
    training: pd.DataFrame
    test: pd.DataFrame
    columns: List[str] = field(init=False)

    def __post_init__(self):
        self.columns = self.training.columns.tolist()


def download_file(url: str, dest_folder: Path) -> Path:
    dest_folder.mkdir(parents=True, exist_ok=True)

    filename = Path(urlparse(url).path).name
    file_path = dest_folder / filename

    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with file_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded to {file_path}")
    return file_path


def unzip_file(zip_path: Path, extract_to: Path) -> None:
    print(f"Unzipping {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print("Unzip complete.")


def get_folder_name_from_url(url: str) -> str:
    filename = Path(urlparse(url).path).name
    folder_name = filename.split(".")[0]  # filename without extension
    return folder_name


def split_dataset(df, target_col, test_size, stratify=True, random_state=42):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=strat, random_state=random_state
    )
    return pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1)


def download_and_extract(uri: str, dest_folder: Path) -> None:
    if dest_folder.exists() and any(dest_folder.iterdir()):
        print(f"Folder {dest_folder} is not empty. Skipping download and extraction.")
    else:
        file_path = download_file(uri, dest_folder)

        if file_path.suffix == ".zip":
            unzip_file(file_path, dest_folder)

        file_path.unlink()


def preprocess_dataset(preprocessing_specs: Dict, training: pd.DataFrame, test: pd.DataFrame):
    for action in preprocessing_specs:
        if "remove" in action:
            training.drop(columns=action["remove"], inplace=True)
            test.drop(columns=action["remove"], inplace=True)
        elif "replace" in action:
            column = action["replace"]["column"]
            replacer = action["replace"]["with"]
            training[column] = training[column].replace(action["replace"]["value"], replacer)
            test[column] = test[column].replace(action["replace"]["value"], replacer)


def create_features(raw_features: List) -> List[Feature]:
    features: List[Feature] = []
    for index, feature in enumerate(raw_features):
        feature_values = feature.get("values", None)
        if isinstance(feature_values, List):
            feature_values = {value: i for i, value in enumerate(feature_values)}
        features.append(
            Feature(
                name=feature["name"],
                column=index,
                type=FeatureType(feature.get("type", "categorical")),
                values=feature_values,
            )
        )
    return features


def create_targets(raw_targets: List, starting_column: int) -> List[Target]:
    targets: List[Target] = []
    for target_feature in raw_targets:
        target_values = target_feature.get("values", None)
        if isinstance(target_values, List):
            target_values = {value: i for i, value in enumerate(target_values)}
        targets.append(
            Target(
                name=target_feature["name"],
                column=starting_column,
                type=FeatureType(target_feature.get("type", "categorical")),
                values=target_values,
            )
        )
        starting_column += 1
    return targets


def get_datasets(specification: Dict, spec_file_path: Path) -> List[Dataset]:
    datasets_spec: Dict = specification["data"]
    datasets: List[Dataset] = []
    for dataset_name, current_dataset in datasets_spec.items():
        raw_features: List = current_dataset["features"]
        raw_targets: List = current_dataset["targets"]

        test_percentage: Optional[float] = None
        if "training" in current_dataset:
            training_dataset: Dict = current_dataset["training"]
            test_dataset: Dict = current_dataset["test"]
        elif "dataset" in current_dataset:
            training_dataset = current_dataset["dataset"]
            test_dataset = current_dataset["dataset"]
            test_percentage = test_dataset["split"]["test"]
        else:
            raise ValueError(f"Dataset {dataset_name} does not have training/test or dataset keys.")

        separator: str = training_dataset["type"].split("'")[1]

        if "uri" in training_dataset:
            dest_folder: Path = spec_file_path / training_dataset["unpack"]
            download_and_extract(training_dataset["uri"], dest_folder)
            training = pd.read_csv(dest_folder / training_dataset["file_name"], sep=separator)
            test = pd.read_csv(dest_folder / test_dataset["file_name"], sep=separator)
        else:
            training = pd.read_csv(spec_file_path / training_dataset["file"], sep=separator)
            test = pd.read_csv(spec_file_path / test_dataset["file"], sep=separator)

        columns: List[str] = [feature["name"] for feature in raw_features]
        columns += [target["name"] for target in raw_targets]

        training.columns = columns
        test.columns = columns

        if test_percentage is not None:
            training, test = split_dataset(training, raw_targets[0]["name"], test_percentage)

        if "preprocess" in current_dataset:
            preprocess_dataset(current_dataset["preprocess"], training, test)
            preprocessed_raw_features = [
                raw_feature for raw_feature in raw_features if raw_feature["name"] in training.columns
            ]
            preprocessed_raw_targets = [
                raw_target for raw_target in raw_targets if raw_target["name"] in training.columns
            ]
            raw_features = preprocessed_raw_features
            raw_targets = preprocessed_raw_targets

        features: List[Feature] = create_features(raw_features)
        targets: List[Target] = create_targets(raw_targets, starting_column=len(raw_features))

        # Convert categorical columns to strings
        for column in features + targets:
            if column.type == FeatureType.CATEGORICAL:
                training[column.name] = training[column.name].astype("str")
                test[column.name] = test[column.name].astype("str")

        datasets.append(
            Dataset(
                name=dataset_name,
                instance_name=current_dataset["instance_name"],
                features=features,
                targets=targets,
                training=training,
                test=test,
            )
        )

    return datasets
