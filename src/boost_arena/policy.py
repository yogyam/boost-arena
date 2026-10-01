"""Loads a submitted model and lets it choose actions.

A submission is a single ONNX file that maps observations to one score ("logit") per action:

    input   "obs"     float32, shape (N, 53)
    output  "logits"  float32, shape (N, 90)

Submitted files come from strangers, so the file is checked before it is run. ONNX holds a
list of mathematical operations, not program code, and only plain feed-forward operations
are accepted. Choosing the action from the logits is done here, not inside the model.
"""

import time

import numpy as np
import onnx
import onnxruntime as ort

from . import interface

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_PARAMETERS = 16_000_000
MAX_NODES = 2_000
MAX_OPSET = 21
MAX_LOGIT = 1e6  # Trained networks give logits in the tens. Anything near this is a broken model

# Bounds on the work one decision may take, so a model cannot stall or exhaust the scoring service.
# Shapes are checked for a batch of this many cars; intermediate values are capped per car.
CHECK_BATCH = 32
MAX_ELEMENTS_PER_CAR = 1_000_000  # Any intermediate value, e.g. a 1000x1000 matrix per car
MAX_TOTAL_ELEMENTS_PER_CAR = 8_000_000  # All intermediate values together
MAX_SECONDS_PER_BATCH = 0.05  # Average over several timed decisions for CHECK_BATCH cars
TIMING_ROUNDS = 20

# Everything a feed-forward network needs. Anything else is refused: no loops or branches,
# no operations that read files, no custom operations.
ALLOWED_OPERATIONS = frozenset(
    {
        "Gemm",
        "MatMul",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Neg",
        "Sqrt",
        "Pow",
        "Reciprocal",
        "Abs",
        "Exp",
        "Log",
        "Relu",
        "LeakyRelu",
        "PRelu",
        "Elu",
        "Selu",
        "Celu",
        "Gelu",
        "Mish",
        "Tanh",
        "Sigmoid",
        "HardSigmoid",
        "HardSwish",
        "Softplus",
        "Softsign",
        "Softmax",
        "LogSoftmax",
        "Erf",
        "Clip",
        "Min",
        "Max",
        "LayerNormalization",
        "BatchNormalization",
        "ReduceMean",
        "ReduceSum",
        "Identity",
        "Constant",
        "Concat",
        "Flatten",
        "Reshape",
        "Squeeze",
        "Unsqueeze",
        "Slice",
        "Gather",
        "Transpose",
        "Cast",
    }
)

# Logits of actions the car may not take are pushed down by this much
DISABLED_LOGIT = -1e10
MIN_PROBABILITY = 1e-11
_DISABLED_LOGIT = np.float32(DISABLED_LOGIT)
_MIN_PROBABILITY = np.float32(MIN_PROBABILITY)


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

    if len(graph.sparse_initializer) > 0 or len(model.training_info) > 0:
        raise InvalidModel("Contains sparse tensors or training information, which are not accepted")

    tensors = list(graph.initializer)  # Every tensor the file carries: weights, and constants inside operations
    for node in graph.node:
        if node.domain not in ("", "ai.onnx"):
            raise InvalidModel(f"Operation '{node.op_type}' is from '{node.domain}', only standard operations are accepted")
        if node.op_type not in ALLOWED_OPERATIONS:
            raise InvalidModel(f"Operation '{node.op_type}' is not accepted. Only feed-forward operations are allowed")
        for attribute in node.attribute:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                raise InvalidModel(f"Operation '{node.op_type}' contains a nested graph, which is not accepted")
            if attribute.type in (onnx.AttributeProto.SPARSE_TENSOR, onnx.AttributeProto.SPARSE_TENSORS):
                raise InvalidModel("Contains sparse tensors, which are not accepted")
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensors.append(attribute.t)
            elif attribute.type == onnx.AttributeProto.TENSORS:
                tensors.extend(attribute.tensors)

    parameters = 0
    for tensor in tensors:
        if tensor.data_location == onnx.TensorProto.EXTERNAL or len(tensor.external_data) > 0:
            raise InvalidModel("Refers to data in another file. The model must be one self-contained file")
        parameters += int(np.prod(tensor.dims)) if len(tensor.dims) else 1
    if parameters > MAX_PARAMETERS:
        raise InvalidModel(f"Has {parameters:,} parameters, the limit is {MAX_PARAMETERS:,}")

    # How a runtime handles these differs between computers, so they are refused outright
    for tensor in tensors:
        if tensor.data_type in (
            onnx.TensorProto.FLOAT,
            onnx.TensorProto.DOUBLE,
            onnx.TensorProto.FLOAT16,
            onnx.TensorProto.BFLOAT16,
        ):
            try:
                values = onnx.numpy_helper.to_array(tensor)
            except Exception as e:
                raise InvalidModel(f"The values of '{tensor.name}' could not be read: {e}") from None
            if not np.isfinite(values.astype(np.float64)).all():
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

    _check_intermediate_sizes(model)

    return {
        "parameters": parameters,
        "operations": len(graph.node),
        "file_bytes": len(data),
        "input_name": inputs[0].name,
        "output_name": graph.output[0].name,
    }


def _check_intermediate_sizes(model) -> None:
    """Refuses a model whose intermediate values would be huge: a few parameters can still
    describe a network that needs gigabytes of memory for one decision."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    except Exception as e:
        raise InvalidModel(f"The shapes of the model's values could not be worked out: {e}") from None
    total = 0
    for info in list(inferred.graph.value_info) + list(inferred.graph.output):
        dims = info.type.tensor_type.shape.dim
        if not dims:
            continue
        elements = 1
        for i, dim in enumerate(dims):
            if dim.HasField("dim_value"):
                elements *= max(dim.dim_value, 1)
            elif i == 0:
                elements *= CHECK_BATCH
            # Any other unknown dimension is counted as 1: the timed decision below catches the rest
        elements //= CHECK_BATCH
        if elements > MAX_ELEMENTS_PER_CAR:
            raise InvalidModel(f"'{info.name}' would hold {elements:,} values per car, the limit is {MAX_ELEMENTS_PER_CAR:,}")
        total += elements
    if total > MAX_TOTAL_ELEMENTS_PER_CAR:
        raise InvalidModel(
            f"The model's intermediate values would hold {total:,} values per car, the limit is {MAX_TOTAL_ELEMENTS_PER_CAR:,}"
        )


class Policy:
    """A checked model, ready to choose actions."""

    def __init__(self, data: bytes, threads: int = 1, timed: bool = True):
        self.info = check_model(data)

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        try:
            self._session = ort.InferenceSession(data, sess_options=options, providers=["CPUExecutionProvider"])
        except Exception as e:
            raise InvalidModel(f"The model could not be loaded: {e}") from None
        if timed:
            self._check_speed()

    def _check_speed(self) -> None:
        """One decision for a batch of cars must be quick, or scoring would take hours."""
        obs = np.zeros((CHECK_BATCH, interface.OBS_SIZE), dtype=np.float32)
        try:
            self.logits(obs)  # Warm up, and catch a model that fails on its first run
            started = time.perf_counter()
            for _ in range(TIMING_ROUNDS):
                self.logits(obs)
            seconds = (time.perf_counter() - started) / TIMING_ROUNDS
        except InvalidModel:
            raise
        except Exception as e:
            raise InvalidModel(f"The model failed on a test decision: {type(e).__name__}") from None
        if seconds > MAX_SECONDS_PER_BATCH:
            raise InvalidModel(
                f"A decision for {CHECK_BATCH} cars takes {seconds * 1000:.0f} ms, the limit is {MAX_SECONDS_PER_BATCH * 1000:.0f} ms"
            )

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
