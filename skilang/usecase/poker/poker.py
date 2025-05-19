from torch import tensor, Tensor
from torchic.nn import InferenceResult

from skilang.usecase.poker.dataset import X_test, y_test
from skilang.usecase.poker.model import model
from skilang.usecase.poker.utils import parse_hand, get_class_name

model.load("model.pth")


for i in range(100):
    x, target = X_test[i, :], y_test[i]
    tensor_x: Tensor = tensor(x).float()
    hand: str = parse_hand(tensor_x)
    input = tensor_x.reshape(1, -1)

    result: InferenceResult = model.inference(input)
    predicted, actual = get_class_name(result.predicted), get_class_name(target)
    print(f"Hand: {hand}")
    print(f'Predicted: "{predicted}", Actual: "{actual}"\n')
