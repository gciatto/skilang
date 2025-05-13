from torch import nn
from torchic.nn import NeuralNetwork
from torchic.nn.builder import NeuralNetworkBuilder
from torchic.utils import get_current_device, logger

INPUT_SIZE = 10
NUM_CLASSES = 10

builder = NeuralNetworkBuilder(device=get_current_device())
model: NeuralNetwork = (
    builder.add_linear(INPUT_SIZE, 64)
    .add_layer(nn.ReLU())
    .add_linear(64, NUM_CLASSES)
    .add_layer(nn.Softmax(dim=1))
    .build()
)

logger.info("NUM_CLASSES: %s", NUM_CLASSES)
