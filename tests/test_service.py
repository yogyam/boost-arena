"""Runs the scoring service's commands end to end on local files: open with the key, score and
duel without it in separate processes, then check everything against the repository."""

import json
import os

import numpy as np
import onnx
import pytest
from onnx import numpy_helper

from boost_arena.cli import PRIVATE_KEY_VARIABLE, main
from boost_arena.policy import uniform_model
from boost_arena.sealed import generate_key_pair
from boost_arena.submissions import make_submission


def distinct_model(name: str) -> bytes:
    """The random bot with a tiny bias, so two submissions do not hold the same file."""
    model = onnx.load_model_from_string(uniform_model())
    bias = numpy_helper.from_array(np.full(90, 1e-3 * (sum(map(ord, name)) % 7), dtype=np.float32), "bias")
    model.graph.initializer[1].CopyFrom(bias)
    return model.SerializeToString()


def submit(tmp_path, public, name, github):
    model = tmp_path / f"{name}.onnx"
    model.write_bytes(distinct_model(name))
    manifest = make_submission(str(model), public, name, "Tester", github, str(tmp_path / "submissions"))
    folder = tmp_path / "submissions" / manifest.slug
    sealed = tmp_path / "submissions" / f"{manifest.slug}.sealed"
    hosted = tmp_path / "hosted"
    hosted.mkdir(exist_ok=True)
    sealed.rename(hosted / sealed.name)
    data = json.loads((folder / "submission.json").read_text())
    data["model_url"] = "file://" + str(hosted / sealed.name)
    (folder / "submission.json").write_text(json.dumps(data))
    return manifest.slug


def test_open_score_duel_and_validate(tmp_path, monkeypatch, capsys):
    private, public = generate_key_pair()
    for name, github in (("Alpha", "a"), ("Beta", "b")):
        submit(tmp_path, public, name, github)
    submissions, results, replays, duels = (str(tmp_path / p) for p in ("submissions", "results", "replays", "duels"))
    for folder in (results, replays, duels):
        os.makedirs(folder)
    models, output = str(tmp_path / "opened"), str(tmp_path / "new_results")

    # Opening is the only step with the key, and it removes the key from the environment
    monkeypatch.setenv(PRIVATE_KEY_VARIABLE, private)
    assert (
        main(
            [
                "open-submissions",
                "--submissions",
                submissions,
                "--results",
                results,
                "--replays",
                replays,
                "--duels",
                duels,
                "--models",
                models,
                "--output",
                output,
                "--allow-local",
            ]
        )
        == 0
    )
    assert PRIVATE_KEY_VARIABLE not in os.environ
    assert sorted(os.listdir(models)) == ["alpha.json", "alpha.onnx", "beta.json", "beta.onnx", "results"]

    # Scoring runs each model in its own process
    assert main(["process-submissions", "--models", models, "--output", output, "--episodes", "2"]) == 0
    for slug in ("alpha", "beta"):
        document = json.load(open(os.path.join(output, f"{slug}.json")))
        assert "error" not in document and len(document["results"]) == 6
        assert os.path.isfile(os.path.join(output, f"{slug}.replays.json.gz"))

    # Duels see the results just produced
    merged = str(tmp_path / "merged")
    os.makedirs(merged)
    for name in os.listdir(output):
        if name.endswith(".json"):
            with open(os.path.join(output, name), "rb") as src, open(os.path.join(merged, name), "wb") as dst:
                dst.write(src.read())
    assert (
        main(
            [
                "process-duels",
                "--submissions",
                submissions,
                "--results",
                merged,
                "--duels",
                duels,
                "--models",
                models,
                "--output",
                output,
                "--episodes",
                "2",
            ]
        )
        == 0
    )
    assert sorted(os.listdir(os.path.join(output, "duels"))) == ["alpha__beta.json", "alpha__beta.replays.json.gz"]

    # The publish job checks everything against the repository. These results are not official (2 episodes)
    files = [os.path.join(output, n) for n in ("alpha.json", "beta.json", "alpha.replays.json.gz", "beta.replays.json.gz")]
    assert main(["validate-results", "--submissions", submissions, "--results", results, "--allow-unofficial"] + files) == 0
    assert main(["validate-results", "--submissions", submissions, "--results", results] + files) == 1
    for name in ("alpha.json", "beta.json"):
        with open(os.path.join(output, name), "rb") as src, open(os.path.join(results, name), "wb") as dst:
            dst.write(src.read())
    duel_files = [os.path.join(output, "duels", n) for n in ("alpha__beta.json", "alpha__beta.replays.json.gz")]
    assert main(["validate-results", "--submissions", submissions, "--results", results, "--allow-unofficial"] + duel_files) == 0

    # A result that is not for the manifest in the repository is refused
    data = json.load(open(os.path.join(output, "alpha.json")))
    data["manifest"]["description"] = "forged"
    with open(os.path.join(output, "alpha.json"), "w") as f:
        json.dump(data, f)
    assert (
        main(
            [
                "validate-results",
                "--submissions",
                submissions,
                "--results",
                results,
                "--allow-unofficial",
                os.path.join(output, "alpha.json"),
            ]
        )
        == 1
    )
    assert "differs" in capsys.readouterr().out

    # Nothing is pending afterwards, and the opened models are gone from the published output
    assert not any(name.endswith(".onnx") for name in os.listdir(output))


def test_the_key_is_required(monkeypatch):
    monkeypatch.delenv(PRIVATE_KEY_VARIABLE, raising=False)
    with pytest.raises(SystemExit):
        main(["open-submissions", "--submissions", "nowhere", "--results", "nowhere", "--models", "x", "--output", "y"])
