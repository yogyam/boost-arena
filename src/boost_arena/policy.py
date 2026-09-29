"""Loads a submitted model and lets it choose actions.

A submission is a single ONNX file that maps observations to one score ("logit") per action:

    input   "obs"     float32, shape (N, 53)
    output  "logits"  float32, shape (N, 90)

Submitted files come from strangers, so the file is checked before it is run. ONNX holds a
list of mathematical operations, not program code, and only plain feed-forward operations
are accepted. Choosing the action from the logits is done here, not inside the model.
"""

import numpy as np
import onnx
import onnxruntime as ort

from . import interface

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_PARAMETERS = 16_000_000
MAX_NODES = 2_000
MAX_OPSET = 21
MAX_LOGIT = 1e6  # Trained networks give logits in the tens. Anything near this is a broken model

# Everything a feed-forward network needs. Anything else is refused: no loops or branches,
# no operations that read files, no custom operations.
ALLOWED_OPERATIONS = frozenset({
    "Gemm", "MatMul", "Add", "Sub", "Mul", "Div", "Neg", "Sqrt", "Pow", "Reciprocal", "Abs", "Exp", "Log",
    "Relu", "LeakyRelu", "PRelu", "Elu", "Selu", "Celu", "Gelu", "Mish", "Tanh", "Sigmoid", "HardSigmoid",
    "HardSwish", "Softplus", "Softsign", "Softmax", "LogSoftmax", "Erf", "Clip", "Min", "Max",
    "LayerNormalization", "BatchNormalization", "ReduceMean", "ReduceSum",
    "Identity", "Constant", "Concat", "Flatten", "Reshape", "Squeeze", "Unsqueeze", "Slice", "Gather",
    "Transpose", "Cast", "Dropout",
})

# Logits of actions the car may not take are pushed down by this much
_DISABLED_LOGIT = np.float32(-1e10)
_MIN_PROBABILITY = np.float32(1e-11)


class InvalidModel(Exception):
    """The file is not an acceptable submission. The message says why."""


def _shape_of(value_info):
    dims = value_info.type.tensor_type.shape.dim
    return [d.dim_value if d.HasField("dim_value") else None for d in dims]


def check_model(data: bytes) -> dict:
    """Checks a model file. Returns a short description, or raises InvalidModel."""
    if len(data) > MAX_FILE_BYTES:
        raise InvalidModel(f"The file is {len(data) / 1e6:.1f} MB, the limit is {MAX_FILE_BYTES / 1e6:.0f} MB")

    try:
        model = onnx.load_model_from_string(data)
    except Exception as e:
        raise InvalidModel(f"Not a readable ONNX file: {e}") from None

    for opset in model.opset_import:
        if opset.domain not in ("", "ai.onnx"):
            raise InvalidModel(f"Uses the operation set '{opset.domain}', only the standard set is accepted")
        if opset.version > MAX_OPSET:
            raise InvalidModel(f"Uses operation set version {opset.version}, the newest accepted is {MAX_OPSET}")

    if len(model.functions) > 0:
        raise InvalidModel("Contains model-defined functions, which are not accepted")

    graph = model.graph
    if len(graph.node) > MAX_NODES:
        raise InvalidModel(f"Has {len(graph.node)} operations, the limit is {MAX_NODES}")

    for node in graph.node:
        if node.domain not in ("", "ai.onnx"):
            raise InvalidModel(f"Operation '{node.op_type}' is from '{node.domain}', only standard operations are accepted")
        if node.op_type not in ALLOWED_OPERATIONS:
            raise InvalidModel(f"Operation '{node.op_type}' is not accepted. Only feed-forward operations are allowed")
        for attribute in node.attribute:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                raise InvalidModel(f"Operation '{node.op_type}' contains a nested graph, which is not accepted")
            if attribute.type in (onnx.AttributeProto.TENSOR, onnx.AttributeProto.TENSORS):
                for tensor in [attribute.t] + list(attribute.tensors):
                    if tensor.data_location == onnx.TensorProto.EXTERNAL or len(tensor.external_data) > 0:
                        raise InvalidModel("Refers to data in another file. The model must be one self-contained file")

    parameters = 0
    for tensor in graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL or len(tensor.external_data) > 0:
            raise InvalidModel("Refers to data in another file. The model must be one self-contained file")
        parameters += int(np.prod(tensor.dims)) if len(tensor.dims) else 1
    if parameters > MAX_PARAMETERS:
        raise InvalidModel(f"Has {parameters:,} parameters, the limit is {MAX_PARAMETERS:,}")

    # How a runtime handles these differs between computers, so they are refused outright
    for tensor in graph.initializer:
        if tensor.data_type in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT16):
            try:
                values = onnx.numpy_helper.to_array(tensor)
            except Exception as e:
                raise InvalidModel(f"The values of '{tensor.name}' could not be read: {e}") from None
            if not np.isfinite(values).all():
                raise InvalidModel(f"'{tensor.name}' contains a value that is not a finite number")

    initializer_names = {tensor.name for tensor in graph.initializer}
    inputs = [i for i in graph.input if i.name not in initializer_names]
    if len(inputs) != 1 or len(graph.output) != 1:
        raise InvalidModel(f"Must have exactly one input and one output, found {len(inputs)} and {len(graph.output)}")

    for what, info, size in (("input", inputs[0], interface.OBS_SIZE), ("output", graph.output[0], interface.NUM_ACTIONS)):
        if info.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise InvalidModel(f"The {what} must be 32-bit floats")
        shape = _shape_of(info)
        if len(shape) != 2 or shape[1] != size:
            raise InvalidModel(f"The {what} must have shape (N, {size}), found {shape}")
        if shape[0] is not None:
            raise InvalidModel(f"The first dimension of the {what} must be left open, so several cars can be decided at once")

    return {
        "parameters": parameters,
        "operations": len(graph.node),
        "file_bytes": len(data),
        "input_name": inputs[0].name,
        "output_name": graph.output[0].name,
    }


class Policy:
    """A checked model, ready to choose actions."""

    def __init__(self, data: bytes, threads: int = 1):
        self.info = check_model(data)

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        try:
            self._session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
        except Exception as e:
            raise InvalidModel(f"The model could not be loaded: {e}") from None

    @classmethod
    def from_file(cls, path, **kwargs):
        with open(path, "rb") as f:
            return cls(f.read(), **kwargs)

    def logits(self, obs: np.ndarray) -> np.ndarray:
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        (logits,) = self._session.run([self.info["output_name"]], {self.info["input_name"]: obs})
        if logits.shape != (obs.shape[0], interface.NUM_ACTIONS):
            raise InvalidModel(f"The model returned shape {logits.shape} for {obs.shape[0]} observations")
        if not np.isfinite(logits).all() or np.abs(logits).max() > MAX_LOGIT:
            raise InvalidModel("The model returned a value that is not a finite number of a sensible size")
        return logits

    def act(self, obs: np.ndarray, masks: np.ndarray, rngs=None) -> np.ndarray:
        """One action index per row of `obs`.

        With `rngs` (one random generator per row) each action is drawn at random in proportion
        to the model's probabilities, which is how bots act while training. Without it the most
        likely action is taken.
        """
        logits = self.logits(obs).astype(np.float32)
        logits = logits + _DISABLED_LOGIT * (~masks.astype(bool))

        if rngs is None:
            return logits.argmax(axis=1)

        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        probabilities = np.clip(probabilities, _MIN_PROBABILITY, 1).astype(np.float64)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        actions = np.empty(len(probabilities), dtype=np.int64)
        for i, row in enumerate(probabilities):
            actions[i] = rngs[i].choice(interface.NUM_ACTIONS, p=row)
        return actions


def uniform_model() -> bytes:
    """A model that gives every action the same logit, so the bot picks among its allowed actions at random.

    Useful as a baseline, and for checking that a scoring setup works.
    """
    from onnx import helper, numpy_helper

    graph = helper.make_graph(
        [helper.make_node("Gemm", ["obs", "weight", "bias"], ["logits"], transB=1)],
        "uniform_policy",
        [helper.make_tensor_value_info("obs", onnx.TensorProto.FLOAT, ["N", interface.OBS_SIZE])],
        [helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, ["N", interface.NUM_ACTIONS])],
        [
            numpy_helper.from_array(np.zeros((interface.NUM_ACTIONS, interface.OBS_SIZE), dtype=np.float32), "weight"),
            numpy_helper.from_array(np.zeros(interface.NUM_ACTIONS, dtype=np.float32), "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="boost-arena")
    model.ir_version = 8
    return model.SerializeToString()
