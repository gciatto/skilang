import torch
from torchic.nn.builder import NeuralNetworkBuilder
from torchic.utils import get_current_device
from models import PATH as MODEL_PATH
from torchic.nn import NeuralNetwork


def load_model(name: str) -> NeuralNetwork:
    model_path = MODEL_PATH / f"{name}.pth"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = (NeuralNetworkBuilder(get_current_device())
             .add_linear(14,64)
             .add_layer(torch.nn.ReLU())
             .add_linear(64, 64)
             .add_layer(torch.nn.ReLU())
             .add_linear(64, 2)
             .add_layer(torch.nn.Softmax())
             ).build()
    model.load(model_path)

    return model


def statistical_parity(predictions: torch.Tensor, protected_attribute: torch.Tensor) -> float:
    """
    Calculate the statistical parity difference.
    :param predictions: Model predictions (tensor of shape [N, num_classes])
    :param protected_attribute: Protected attribute values (tensor of shape [N])
    :return: Statistical parity difference
    """
    # Convert predictions to binary (1 for positive class, 0 for negative class)
    unique_values = protected_attribute.unique()
    value_0 = max(unique_values)
    predicted_classes = predictions.argmax(dim=1)

    # Calculate the proportion of positive predictions for each group
    group_0_positive_rate = predicted_classes[protected_attribute == value_0].float().mean().item()
    group_1_positive_rate = predicted_classes[protected_attribute != value_0].float().mean().item()

    # Calculate statistical parity difference
    return abs(group_0_positive_rate - group_1_positive_rate)
