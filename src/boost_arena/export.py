"""Writing a policy network as a Boost Arena submission file (ONNX).

A network is described as a list of layers, each ("linear", weight, bias) or
("layer_norm", scale, shift), with the activation applied after every hidden layer
(after the layer norm, when there is one) and no activation after the last linear layer.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .interface import NUM_ACTIONS, OBS_SIZE

LAYER_NORM_EPSILON = 1e-5  # PyTorch's default, which GigaLearnCPP uses

ACTIVATIONS = {
    "leaky_relu": ("LeakyRelu", {"alpha": 0.01}),
    "relu": ("Relu", {}),
    "tanh": ("Tanh", {}),
    "sigmoid": ("Sigmoid", {}),
}


def read_gigalearn_layers(path):
    """The parameters of one .lt file as a list of ("linear" | "layer_norm", weight, bias)."""
    import warnings

    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        module = torch.jit.load(path, map_location="cpu")
    parameters = {name: value.detach().numpy().astype(np.float32) for name, value in module.named_parameters()}

    layers = []
    for index in sorted({int(name.split(".")[0]) for name in parameters}):
        weight, bias = parameters[f"{index}.weight"], parameters[f"{index}.bias"]
        layers.append(("linear" if weight.ndim == 2 else "layer_norm", weight, bias))
    return layers


def build_onnx(layers, activation):
    op_type, attributes = ACTIVATIONS[activation]
    nodes, initializers = [], []
    current = "obs"

    def constant(name, array):
        initializers.append(numpy_helper.from_array(array, name))
        return name

    for i, (kind, weight, bias) in enumerate(layers):
        is_output_layer = i == len(layers) - 1
        if kind == "linear":
            out = "logits" if is_output_layer else f"linear_{i}"
            nodes.append(helper.make_node(
                # PyTorch stores the weights as (outputs, inputs), so they are transposed
                "Gemm", [current, constant(f"weight_{i}", weight), constant(f"bias_{i}", bias)], [out], transB=1
            ))
            current = out
            # GigaLearnCPP applies the activation after the layer norm, when there is one
            followed_by_norm = not is_output_layer and layers[i + 1][0] == "layer_norm"
            if not is_output_layer and not followed_by_norm:
                nodes.append(helper.make_node(op_type, [current], [f"activation_{i}"], **attributes))
                current = f"activation_{i}"
        else:
            nodes.append(helper.make_node(
                "LayerNormalization", [current, constant(f"scale_{i}", weight), constant(f"shift_{i}", bias)],
                [f"norm_{i}"], axis=-1, epsilon=LAYER_NORM_EPSILON,
            ))
            nodes.append(helper.make_node(op_type, [f"norm_{i}"], [f"activation_{i}"], **attributes))
            current = f"activation_{i}"

    if layers[-1][0] != "linear":
        raise SystemExit("The policy does not end in an output layer")

    graph = helper.make_graph(
        nodes, "boost_arena_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["N", OBS_SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", NUM_ACTIONS])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="boost-arena")
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def reference_forward(layers, activation, obs):
    """The same network in plain numpy, to check the ONNX file against."""
    def activate(x):
        if activation == "leaky_relu":
            return np.where(x > 0, x, 0.01 * x)
        if activation == "relu":
            return np.maximum(x, 0)
        if activation == "tanh":
            return np.tanh(x)
        return 1 / (1 + np.exp(-x))

    x = obs.astype(np.float64)
    for i, (kind, weight, bias) in enumerate(layers):
        if kind == "linear":
            x = x @ weight.T.astype(np.float64) + bias
            followed_by_norm = i + 1 < len(layers) and layers[i + 1][0] == "layer_norm"
            if i < len(layers) - 1 and not followed_by_norm:
                x = activate(x)
        else:
            mean = x.mean(axis=-1, keepdims=True)
            variance = x.var(axis=-1, keepdims=True)
            x = activate((x - mean) / np.sqrt(variance + LAYER_NORM_EPSILON) * weight + bias)
    return x



def check_and_serialize(layers, activation) -> bytes:
    """Builds the ONNX model, checks it against a plain numpy version of the network, and returns the bytes."""
    import onnxruntime as ort

    first, last = layers[0][1], layers[-1][1]
    if first.shape[1] != OBS_SIZE or last.shape[0] != NUM_ACTIONS:
        raise ValueError(f"This network takes {first.shape[1]} inputs and gives {last.shape[0]} outputs. Boost Arena needs {OBS_SIZE} and {NUM_ACTIONS}.")

    data = build_onnx(layers, activation).SerializeToString()
    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    obs = np.random.default_rng(0).uniform(-1, 1, size=(256, OBS_SIZE)).astype(np.float32)
    difference = np.abs(session.run(None, {"obs": obs})[0] - reference_forward(layers, activation, obs)).max()
    if difference > 1e-3:
        raise ValueError(f"The exported model disagrees with the network by {difference:.4g}")
    return data


def layers_from_sequential(sequential):
    """The layer list of a torch nn.Sequential made of Linear, LayerNorm and activation modules."""
    import torch.nn as nn

    layers = []
    for module in sequential:
        if isinstance(module, nn.Linear):
            layers.append(("linear", module.weight.detach().cpu().numpy().astype(np.float32), module.bias.detach().cpu().numpy().astype(np.float32)))
        elif isinstance(module, nn.LayerNorm):
            layers.append(("layer_norm", module.weight.detach().cpu().numpy().astype(np.float32), module.bias.detach().cpu().numpy().astype(np.float32)))
    return layers
