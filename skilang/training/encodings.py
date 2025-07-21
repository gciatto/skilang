from copy import deepcopy
from typing import Dict, Tuple, List

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

from skilang.specification.data import Dataset
from skilang.specification.learnable import EncodingType


def encode_dataset(
    dataset: Dataset,
    encodings: Dict[EncodingType, List[str]],
) -> Tuple[Dataset, Dict[str, Dict[str, float]]]:
    """
    Apply encodings to the dataset.
    :param dataset: The dataset to encode.
    :param encodings: A dictionary mapping features to encoding types.
    :return: The encoded dataset.
    """
    mappings: Dict[str, Dict[str, float]] = {}
    encoded_dataset: Dataset = deepcopy(dataset)
    for encoding, columns in encodings.items():
        if encoding == EncodingType.ONE_HOT:
            raise ValueError(f"Unsupported encoding type: {encoding}")
        elif encoding == EncodingType.ORDINAL:
            encoded_dataset, mappings = ordinal_encoding(dataset, columns)
        else:
            raise ValueError(f"Unsupported encoding type: {encoding}")

    return encoded_dataset, mappings


def ordinal_encoding(
    dataset: "Dataset",
    columns: List[str],
) -> Tuple["Dataset", Dict[str, Dict[str, float]]]:
    encoder = OrdinalEncoder()
    scaler = StandardScaler()
    label_encoder = LabelEncoder()

    features: List[str] = [column.split(".")[-1] for column in columns if column.startswith("features.")]
    targets: List[str] = [column.split(".")[-1] for column in columns if column.startswith("targets.")]

    encoded_datasets = [deepcopy(dataset.training), deepcopy(dataset.test)]
    final_mappings: Dict[str, Dict[str, float]] = {}
    for index, df in enumerate([dataset.training, dataset.test]):
        X_raw = df.iloc[:, :-1]
        y_raw = df.iloc[:, -1]

        # categorical_cols = X_raw.select_dtypes(include=["object", "category"]).columns
        X_encoded = X_raw.copy()

        # if it's the training set
        if index == 0:
            X_encoded[features] = encoder.fit_transform(X_raw[features])
            X_scaled_array = scaler.fit_transform(X_encoded)
        else:
            X_encoded[features] = encoder.transform(X_raw[features])
            X_scaled_array = scaler.transform(X_encoded)

        X_scaled_df = pd.DataFrame(X_scaled_array, columns=X_encoded.columns, index=X_encoded.index)

        features_mappings: Dict[str, Dict[str, float]] = {}
        for i, col in enumerate(features):
            categories = encoder.categories_[i]
            mapping: Dict[str, float] = {}

            for j, cat in enumerate(categories):
                # Create a temp row to isolate the effect of a single value
                temp_row = X_encoded.iloc[0:1].copy()
                temp_row.loc[:, :] = 0  # zero out all features
                temp_row[col] = j  # set the category value

                # Scale the row
                scaled_value = scaler.transform(temp_row)[0][X_encoded.columns.get_loc(col)]
                mapping[cat] = scaled_value

            features_mappings[col] = mapping

        final_mappings = features_mappings
        if y_raw.name == targets[0]:
            # Encode target
            y_encoded_array = label_encoder.fit_transform(y_raw)
            y_encoded_series = pd.Series(y_encoded_array, name=y_raw.name, index=y_raw.index)

            target_mapping = {y_raw.name: dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}

            encoded_datasets[index] = pd.concat([X_scaled_df, y_encoded_series], axis=1)

            final_mappings.update(target_mapping)

    encoded_dataset: Dataset = deepcopy(dataset)
    encoded_dataset.training = encoded_datasets[0]
    encoded_dataset.test = encoded_datasets[1]
    return encoded_dataset, final_mappings
    # TODO: 2 problemi: valori classe 2 e 4 sostituiscono colonne,
    #  e i dieci valori da 1 a 10 sono diversi per ogni feature (quindi tiene l'ultimo)
