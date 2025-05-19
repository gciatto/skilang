from typing import List, Dict

from skilang.specification.learnable.base import Learnable
from skilang.specification.learnable.enum import LayerType, Activation, Regularization, Backend, LearnableType
from skilang.specification.learnable.feed_forward import Layer, create_feed_forward


def __parse_layers(layer_specs: List[Dict]) -> List[Layer]:
    layers = []
    for item in layer_specs:
        layer_type_str, params = item.popitem()
        layer = Layer(
            type=LayerType(layer_type_str),
            input_size=params["input"],
            output_size=params["output"],
            activation=Activation(params.get("activation")) if params.get("activation") else None,
            regularization=Regularization(params.get("regularization"))
            if params.get("regularization")
            else None,
            dropout=params.get("dropout", 0.0),
        )
        layers.append(layer)
    return layers


def get_learnables(spec: Dict) -> List[Learnable]:
    learnables: List[Learnable] = []
    for name, props in spec.get("learnable", {}).items():
        if "dataset" not in props:
            raise ValueError(f"Missing dataset for learnable '{name}'")

        learnable_type = LearnableType(props["type"])
        backend = Backend(props.get("backend", Backend.PYTORCH.value))

        if learnable_type == LearnableType.FEED_FORWARD:
            layers = __parse_layers(props["layers"])
            learnables.append(create_feed_forward(name, props["dataset"], backend, layers))

        # Add support for more learnable types here

    return learnables
