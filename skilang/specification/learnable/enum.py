from enum import Enum


class LearnableType(Enum):
    FEED_FORWARD = "neural_network"
    TRANSFORMER = "transformer"


class Backend(Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    JAX = "jax"


class LayerType(Enum):
    Linear = "linear"
    Bilinear = "bilinear"
    Conv1d = "conv1d"
    Conv2d = "conv2d"


class Activation(Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"


class Regularization(Enum):
    L1 = "l1"
    L2 = "l2"
