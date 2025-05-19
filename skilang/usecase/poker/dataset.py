import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from skilang import logger
from tests.resources.datasets import dataset_path


training = pd.read_csv(dataset_path("poker-hand/poker-hand-testing.csv"))
test = pd.read_csv(dataset_path("poker-hand/poker-hand-training.csv"))
logger.info(f"Training data shape: {training}")


columns = []
for i in range(5):
    columns.append(f"suit{i}")
    columns.append(f"rank{i}")

columns.append("class")
training.columns = columns
test.columns = columns

print(training.describe())
print(test.describe())
#
X_train = training.iloc[:, :-1].values
y_train = training.iloc[:, -1].values

X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1].values

print(training.columns)
print(test.columns)
print("TRAINING")
print(training["class"].value_counts().sum())
print("TEST")
print(test["class"].value_counts().sum())


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
