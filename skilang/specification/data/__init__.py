import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
from urllib.parse import urlparse

import pandas as pd
import requests


@dataclass
class Feature:
    name: str
    column: int
    values: Dict[str, int] = field(default_factory=dict)


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


def download_and_extract(uri: str, dest_folder: Path) -> None:
    if dest_folder.exists() and any(dest_folder.iterdir()):
        print(f"Folder {dest_folder} is not empty. Skipping download and extraction.")
    else:
        file_path = download_file(uri, dest_folder)

        if file_path.suffix == ".zip":
            unzip_file(file_path, dest_folder)

        file_path.unlink()


def get_datasets(specification: Dict, spec_file_path: Path) -> List[Dataset]:
    datasets_spec: Dict = specification["data"]
    datasets: List[Dataset] = []
    for dataset_name, current_dataset in datasets_spec.items():
        features: List[Feature] = [
            Feature(
                name=feature["name"],
                column=index,
                values=feature["values"],
            )
            for index, feature in enumerate(current_dataset["features"])
        ]
        target: Target = Target(
            name=current_dataset["target"]["name"],
            column=len(current_dataset["features"]),
            values=current_dataset["target"]["values"],
        )

        separator: str = current_dataset["training"]["type"].split("'")[1]

        if "uri" in current_dataset["training"]:
            dest_folder: Path = spec_file_path / current_dataset["training"]["unpack"]
            download_and_extract(current_dataset["training"]["uri"], dest_folder)
            training = pd.read_csv(dest_folder / current_dataset["training"]["file_name"], sep=separator)
            test = pd.read_csv(dest_folder / current_dataset["test"]["file_name"], sep=separator)
        else:
            training = pd.read_csv(spec_file_path / current_dataset["training"]["file"], sep=separator)
            test = pd.read_csv(spec_file_path / current_dataset["test"]["file"], sep=separator)

        columns: List[str] = [feature.name for feature in features]
        columns.append(target.name)

        training.columns = columns
        test.columns = columns

        datasets.append(
            Dataset(
                name=dataset_name,
                instance_name=current_dataset["instance_name"],
                features=features,
                target=target,
                training=training,
                test=test,
            )
        )

    return datasets
