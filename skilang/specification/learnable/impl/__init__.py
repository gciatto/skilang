from typing import List

import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchic.nn import NeuralNetwork
from torchic.nn.builder import NeuralNetworkBuilder
from torchic.utils import get_current_device

from skilang.specification.learnable import Activation
from skilang.specification.learnable.base import Learnable
from skilang.specification.learnable.feed_forward import FeedForward, LayerType, Layer


def create_torch_layers(layer: Layer) -> List[nn.Module]:
    layers: List[nn.Module] = []
    if layer.type == LayerType.Linear:
        layers.append(nn.Linear(layer.input_size, layer.output_size))
    elif layer.type == LayerType.Conv1d:
        layers.append(nn.Conv1d(layer.input_size, layer.output_size, layer.kernel_size))
    elif layer.type == LayerType.Conv2d:
        layers.append(nn.Conv2d(layer.input_size, layer.output_size, layer.kernel_size))
    else:
        raise ValueError(f"Unsupported layer type: {layer.type}")

    if layer.dropout is not None:
        layers.append(nn.Dropout(layer.dropout))
    if layer.activation is not None:
        if layer.activation == Activation.RELU:
            layers.append(nn.ReLU())
        elif layer.activation == Activation.LEAKY_RELU:
            layers.append(nn.LeakyReLU())
        elif layer.activation == Activation.SIGMOID:
            layers.append(nn.Sigmoid())
        elif layer.activation == Activation.TANH:
            layers.append(nn.Tanh())
        elif layer.activation == Activation.SOFTMAX:
            layers.append(nn.Softmax(dim=1))
        else:
            raise ValueError(f"Unsupported activation function: {layer.activation}")
    return layers


def create_torch_model(learnable: Learnable) -> NeuralNetwork:
    if isinstance(learnable, FeedForward):
        builder = NeuralNetworkBuilder(get_current_device())
        for layer in learnable.layers:
            torch_layers: List[nn.Module] = create_torch_layers(layer)
            for torch_layer in torch_layers:
                builder.add_layer(torch_layer)
        return builder.build()
    else:
        raise ValueError(f"Unsupported learnable type: {learnable}")


def create_torch_dataloader(dataset: pd.DataFrame, batch_size: int) -> DataLoader:
    """
    Create a PyTorch DataLoader from a pandas DataFrame.
    :param dataset: The dataset to be converted.
    :param batch_size: The batch size for the DataLoader.
    :return: A DataLoader object.
    """

    # for NON Adult dataset
    # X = dataset.iloc[:, :-1].values
    # y = dataset.iloc[:, -1].values
    #
    # X = torch.tensor(X, dtype=torch.float32)
    # y = torch.tensor(y, dtype=torch.float32)

    X_raw = dataset.iloc[:, :-1]
    y_raw = dataset.iloc[:, -1]

    categorical_cols = X_raw.select_dtypes(include=["object", "category"]).columns
    encoder = OrdinalEncoder()
    X_encoded = X_raw.copy()
    X_encoded[categorical_cols] = encoder.fit_transform(X_raw[categorical_cols])

    # # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Encode target
    y_encoded = LabelEncoder().fit_transform(y_raw)

    # Convert to tensors
    X = torch.tensor(X_encoded, dtype=torch.float32)
    y = torch.tensor(y_encoded, dtype=torch.float32)

    tensor_dataset = TensorDataset(X, y)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
