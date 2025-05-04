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

print(dataset.describe())

# Data preprocessing
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


logger.info(f"X shape: {X.shape}")
logger.info(f"X: {X}")
logger.info(f"y shape: {Y.shape}")
logger.info(f"y: {Y}")


# One-hot encoding of class variable
# encoder = OneHotEncoder(sparse_output=False)
# logger.info(f"OneHotEncoder: {encoder}")
# y = encoder.fit_transform(y.reshape(-1, 1))
# logger.info(f"encoded y shape: {y.shape}")
# logger.info(f"encoded y: {y}")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


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
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)
logger.info(f"X_train shape: {X_train.shape}")
logger.info(f"X_train: {X_train}")
logger.info(f"y_train shape: {Y_train.shape}")
logger.info(f"y_train: {Y_train}")

# Creazione del DataLoader
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
