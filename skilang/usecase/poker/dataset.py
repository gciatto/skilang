from pathlib import Path

import pandas as pd
import torch
from openml import datasets, config
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from skilang import logger

config.set_root_cache_directory(Path(__file__).resolve().parent)
dataset = datasets.get_dataset(dataset_id=1569, download_features_meta_data=True).get_data()[0]
assert isinstance(dataset, pd.DataFrame)

columns = []
for i in range(5):
    columns.append(f"suit{i}")
    columns.append(f"rank{i}")

columns.append("class")
logger.info(f"Dataset columns: {dataset.columns}")
logger.info(f"Columns: {columns}")
dataset.columns = columns

last_col = dataset.columns[-1]
dataset[last_col] = dataset[last_col].astype(int) - 1  # Convert to zero-based index

# print(dataset.describe())

# numerous_classes = dataset[last_col].sort_values().unique()[:2]
#
# print(numerous_classes)
# print(dataset[last_col].value_counts())
# filtered_dataset = dataset[dataset[last_col].isin(numerous_classes)]
#
# fraction = 0.1
# sampled = filtered_dataset.groupby(last_col, group_keys=False).apply(lambda df: df.sample(frac=fraction, random_state=42))
# remaining = dataset[~dataset[last_col].isin(numerous_classes)]
# dataset = pd.concat([sampled, remaining], ignore_index=True)
# print(dataset[last_col].value_counts())

# Data preprocessing
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


logger.info(f"X shape: {X.shape}")
logger.info(f"X: {X}")
logger.info(f"y shape: {y.shape}")
logger.info(f"y: {y}")


# One-hot encoding of class variable
# encoder = OneHotEncoder(sparse_output=False)
# logger.info(f"OneHotEncoder: {encoder}")
# y = encoder.fit_transform(y.reshape(-1, 1))
# logger.info(f"encoded y shape: {y.shape}")
# logger.info(f"encoded y: {y}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# training = pd.read_csv(dataset_path("poker-hand/poker-hand-testing.csv"))
# testing = pd.read_csv(dataset_path("poker-hand/poker-hand-training.csv"))
# logger.info(f"Training data shape: {training}")
#
# print(training.describe())
# print(testing.describe())
#
# X_train = training.iloc[:, :-1].values
# y_train = training.iloc[:, -1].values
#
# X_test = testing.iloc[:, :-1].values
# y_test = testing.iloc[:, -1].values

# Conversione in tensori PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
logger.info(f"X_train shape: {X_train.shape}")
logger.info(f"X_train: {X_train}")
logger.info(f"y_train shape: {y_train.shape}")
logger.info(f"y_train: {y_train}")

# Creazione del DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
