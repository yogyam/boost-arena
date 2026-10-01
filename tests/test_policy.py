"""Checks that acceptable model files are accepted and everything else is refused."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from boost_arena.policy import InvalidModel, Policy, check_model


def make_model(
    nodes=None,
    input_shape=("N", 53),
    output_shape=("N", 90),
    initializers=None,
    opset=17,
    input_type=TensorProto.FLOAT,
    domain="",
):
    rng = np.random.default_rng(0)
    if initializers is None:
        initializers = [
            numpy_helper.from_array(rng.normal(size=(90, 53)).astype(np.float32), "weight"),
            numpy_helper.from_array(np.zeros(90, dtype=np.float32), "bias"),
        ]
    if nodes is None:
        nodes = [helper.make_node("Gemm", ["obs", "weight", "bias"], ["logits"], transB=1)]
    graph = helper.make_graph(
        nodes,
        "test",
        [helper.make_tensor_value_info("obs", input_type, list(input_shape))],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, list(output_shape))],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid(domain, opset)])
    model.ir_version = 8
    return model.SerializeToString()


def test_accepts_a_plain_network():
    info = check_model(make_model())
    assert info["parameters"] == 90 * 53 + 90


def test_chooses_only_allowed_actions():
    policy = Policy(make_model())
    obs = np.random.default_rng(1).normal(size=(64, 53)).astype(np.float32)
    masks = np.zeros((64, 90), dtype=bool)
    masks[:, 10:20] = True

    likely = policy.act(obs, masks)
    drawn = policy.act(obs, masks, [np.random.default_rng(i) for i in range(64)])
    for actions in (likely, drawn):
        assert ((actions >= 10) & (actions < 20)).all()


def test_drawn_actions_repeat_with_the_same_seed():
    policy = Policy(make_model())
    obs = np.random.default_rng(1).normal(size=(16, 53)).astype(np.float32)
    masks = np.ones((16, 90), dtype=bool)
    first = policy.act(obs, masks, [np.random.default_rng(i) for i in range(16)])
    second = policy.act(obs, masks, [np.random.default_rng(i) for i in range(16)])
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        (dict(input_shape=("N", 52)), "shape"),
        (dict(output_shape=("N", 89)), "shape"),
        (dict(input_shape=(1, 53)), "first dimension"),
        (dict(input_type=TensorProto.DOUBLE), "32-bit"),
        (dict(opset=99), "operation set version"),
    ],
)
def test_refuses_the_wrong_shape_or_type(kwargs, reason):
    with pytest.raises(InvalidModel, match=reason):
        check_model(make_model(**kwargs))


def test_refuses_operations_that_are_not_feed_forward():
    nodes = [
        helper.make_node("Gemm", ["obs", "weight", "bias"], ["hidden"], transB=1),
        helper.make_node("RandomNormalLike", ["hidden"], ["logits"]),
    ]
    with pytest.raises(InvalidModel, match="RandomNormalLike"):
        check_model(make_model(nodes=nodes))


def test_refuses_custom_operations():
    nodes = [helper.make_node("Gemm", ["obs", "weight", "bias"], ["logits"], transB=1, domain="com.example")]
    with pytest.raises(InvalidModel, match="standard"):
        check_model(make_model(nodes=nodes))


def test_refuses_data_kept_in_another_file():
    weight = numpy_helper.from_array(np.zeros((90, 53), dtype=np.float32), "weight")
    weight.ClearField("raw_data")
    weight.data_location = TensorProto.EXTERNAL
    entry = weight.external_data.add()
    entry.key, entry.value = "location", "../../somewhere/else.bin"
    bias = numpy_helper.from_array(np.zeros(90, dtype=np.float32), "bias")
    with pytest.raises(InvalidModel, match="another file"):
        check_model(make_model(initializers=[weight, bias]))


def test_refuses_too_many_parameters():
    # Only the declared size matters here, so the tensor is left without any data
    huge = TensorProto(name="weight", data_type=TensorProto.FLOAT, dims=[20_000_000, 1])
    bias = numpy_helper.from_array(np.zeros(90, dtype=np.float32), "bias")
    with pytest.raises(InvalidModel, match="parameters"):
        check_model(make_model(initializers=[huge, bias]))


def test_refuses_files_that_are_not_models():
    with pytest.raises(InvalidModel):
        check_model(b"this is not a model")


def test_refuses_values_that_are_not_numbers():
    initializers = [
        numpy_helper.from_array(np.full((90, 53), np.nan, dtype=np.float32), "weight"),
        numpy_helper.from_array(np.zeros(90, dtype=np.float32), "bias"),
    ]
    with pytest.raises(InvalidModel, match="finite"):
        check_model(make_model(initializers=initializers))


def test_refuses_absurdly_large_outputs():
    initializers = [
        numpy_helper.from_array(np.full((90, 53), 1e30, dtype=np.float32), "weight"),
        numpy_helper.from_array(np.zeros(90, dtype=np.float32), "bias"),
    ]
    policy = Policy(make_model(initializers=initializers))
    with pytest.raises(InvalidModel, match="finite"):
        policy.act(np.ones((2, 53), dtype=np.float32), np.ones((2, 90), dtype=bool))


def test_refuses_models_whose_values_would_be_huge():
    """A few parameters can still describe a network that needs gigabytes for one decision."""
    nodes = [helper.make_node("Gemm", ["obs", "weight", "bias"], ["x0"], transB=1)]
    previous = "x0"
    for i in range(24):  # Doubling 24 times: 90 x 2^24 values per car
        nodes.append(helper.make_node("Concat", [previous, previous], [f"x{i + 1}"], axis=1))
        previous = f"x{i + 1}"
    nodes.append(helper.make_node("Slice", [previous, "start", "end", "axes"], ["logits"]))
    rng = np.random.default_rng(0)
    initializers = [
        numpy_helper.from_array(rng.normal(size=(90, 53)).astype(np.float32), "weight"),
        numpy_helper.from_array(np.zeros(90, dtype=np.float32), "bias"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "start"),
        numpy_helper.from_array(np.array([90], dtype=np.int64), "end"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axes"),
    ]
    with pytest.raises(InvalidModel, match="values per car"):
        check_model(make_model(nodes=nodes, initializers=initializers))


def test_refuses_sparse_tensors_and_dropout():
    nodes = [
        helper.make_node("Gemm", ["obs", "weight", "bias"], ["hidden"], transB=1),
        helper.make_node("Dropout", ["hidden"], ["logits"]),
    ]
    with pytest.raises(InvalidModel, match="Dropout"):
        check_model(make_model(nodes=nodes))

    model = onnx.load_model_from_string(make_model())
    sparse = model.graph.sparse_initializer.add()
    sparse.dims.extend([100_000, 100_000])
    with pytest.raises(InvalidModel, match="sparse"):
        check_model(model.SerializeToString())


def test_constants_inside_operations_are_checked_too():
    bad = helper.make_tensor("bad", TensorProto.FLOAT, [90], [float("nan")] * 90)
    nodes = [
        helper.make_node("Constant", [], ["offset"], value=bad),
        helper.make_node("Gemm", ["obs", "weight", "bias"], ["hidden"], transB=1),
        helper.make_node("Add", ["hidden", "offset"], ["logits"]),
    ]
    with pytest.raises(InvalidModel, match="finite"):
        check_model(make_model(nodes=nodes))


def test_a_slow_model_is_refused(monkeypatch):
    import boost_arena.policy as policy_module

    monkeypatch.setattr(policy_module, "MAX_SECONDS_PER_BATCH", 0.0)
    with pytest.raises(InvalidModel, match="takes"):
        Policy(make_model())
    assert Policy(make_model(), timed=False) is not None
