#!/usr/bin/env python3
"""Converts a GigaLearnCPP checkpoint into a Boost Arena submission (one ONNX file).

A checkpoint folder holds SHARED_HEAD.lt and POLICY.lt. Both are read, joined into one
network and written as ONNX. The critic and the optimiser files are not needed.

    python tools/convert_gigalearn_checkpoint.py checkpoints/1300280064 my_bot.onnx

Only convert checkpoints you trained yourself or fully trust: the .lt format can run
code when it is opened. The ONNX file this writes cannot, which is why submissions use it.

Needs PyTorch:  pip install torch
"""

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OBS_SIZE = 53
NUM_ACTIONS = 90
LAYER_NORM_EPSILON = 1e-5  # PyTorch's default, which GigaLearnCPP uses

ACTIVATIONS = {
    "leaky_relu": ("LeakyRelu", {"alpha": 0.01}),
    "relu": ("Relu", {}),
    "tanh": ("Tanh", {}),
    "sigmoid": ("Sigmoid", {}),
}


def read_layers(path):
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


def main():
    parser = argparse.ArgumentParser(description="Convert a GigaLearnCPP checkpoint into a Boost Arena ONNX file")
    parser.add_argument("checkpoint", help="Checkpoint folder holding SHARED_HEAD.lt and POLICY.lt")
    parser.add_argument("output", help="ONNX file to write")
    parser.add_argument("--activation", choices=sorted(ACTIVATIONS), default="leaky_relu",
                        help="The activation the network was trained with (default: leaky_relu)")
    args = parser.parse_args()

    layers = []
    for name in ("SHARED_HEAD.lt", "POLICY.lt"):
        path = os.path.join(args.checkpoint, name)
        if os.path.exists(path):
            layers += read_layers(path)
        elif name == "POLICY.lt":
            raise SystemExit(f"No POLICY.lt in {args.checkpoint}")

    first, last = layers[0][1], layers[-1][1]
    if first.shape[1] != OBS_SIZE or last.shape[0] != NUM_ACTIONS:
        raise SystemExit(
            f"This network takes {first.shape[1]} inputs and gives {last.shape[0]} outputs. "
            f"Boost Arena needs {OBS_SIZE} and {NUM_ACTIONS}."
        )

    model = build_onnx(layers, args.activation)
    data = model.SerializeToString()

    import onnxruntime as ort

    session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    obs = np.random.default_rng(0).uniform(-1, 1, size=(256, OBS_SIZE)).astype(np.float32)
    difference = np.abs(session.run(None, {"obs": obs})[0] - reference_forward(layers, args.activation, obs)).max()
    if difference > 1e-3:
        raise SystemExit(f"The converted model disagrees with the checkpoint by {difference:.4g}. Not written.")

    with open(args.output, "wb") as f:
        f.write(data)

    parameters = sum(weight.size + bias.size for _, weight, bias in layers)
    print(f"Wrote {args.output}: {parameters:,} parameters, {len(data) / 1e6:.1f} MB, largest difference {difference:.2g}")


if __name__ == "__main__":
    sys.exit(main())
