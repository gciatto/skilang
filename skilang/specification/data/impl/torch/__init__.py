import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from skilang.specification.data import Dataset


class DataModule(L.LightningDataModule):
    def __init__(self, dataset: Dataset, batch_size: int = 32):
        super().__init__()
        self.dataset: Dataset = dataset
        self.batch_size = batch_size

        self.train_dataset: TensorDataset = TensorDataset()
        self.val_dataset: TensorDataset = TensorDataset()

    def prepare_data(self) -> None:
        # Download or prepare data here
        # Do nothing because dataset is downloaded in previous steps
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        self.train_dataset = self._create_tensor_dataset(self.dataset.training)
        self.val_dataset = self._create_tensor_dataset(self.dataset.test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)
    #
    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass

    def _create_tensor_dataset(self, dataset_df: pd.DataFrame) -> TensorDataset:
        """
        Create a TensorDataset from a pandas DataFrame.
        :param dataset_df: The dataset to be converted.
        :return: A DataLoader object.
        """

        X = dataset_df.iloc[:, :-1].values
        y = dataset_df.iloc[:, -1].values

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return TensorDataset(X, y)
