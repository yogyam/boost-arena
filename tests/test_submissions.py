"""Checks the submission flow from sealing a model to a published result, using local files."""

import json
import os

import pytest

from boost_arena.policy import uniform_model
from boost_arena.sealed import generate_key_pair
from boost_arena.site import build_site
from boost_arena.submissions import (
    REPLAYS_SUFFIX, SubmissionError, load_manifest, make_submission, read_replays, score_submission,
    submissions_to_score, validate_replays_document, validate_result_document, verify_submission, write_replays,
)


@pytest.fixture
def keys():
    return generate_key_pair()


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "bot.onnx"
    path.write_bytes(uniform_model())
    return str(path)


def make(tmp_path, keys, model_path, name="Test Bot", **kwargs):
    _, public = keys
    submissions = tmp_path / "submissions"
    manifest = make_submission(model_path, public, name, "Tester", str(submissions), **kwargs)
    folder = submissions / manifest.slug
    # The sealed file is moved out of the folder, as if hosted elsewhere, and the manifest points at it
    sealed = folder / f"{manifest.slug}.sealed"
    hosted = tmp_path / "hosted"
    hosted.mkdir(exist_ok=True)
    sealed.rename(hosted / sealed.name)
    data = json.loads((folder / "submission.json").read_text())
    data["model_url"] = "file://" + str(hosted / sealed.name)
    (folder / "submission.json").write_text(json.dumps(data))
    return str(folder)


def test_submit_verify_score_and_publish(tmp_path, keys, model_path):
    private, public = keys
    folder = make(tmp_path, keys, model_path, description="Acts at random")

    manifest = load_manifest(folder)
    assert manifest.slug == "test-bot" and manifest.description == "Acts at random"
    assert verify_submission(folder, public_key=public, allow_local=True)["name"] == "Test Bot"

    results = tmp_path / "results"
    assert submissions_to_score(str(tmp_path / "submissions"), str(results)) == [folder]

    document, replays = score_submission(folder, private, allow_local=True, episodes=4)
    assert "error" not in document, document.get("error")
    assert len(document["results"]) == 6
    assert 0 <= document["overall_score"] <= 100
    assert document["official"] is False   # Only four episodes
    validate_result_document(document)
    validate_replays_document(replays)
    assert all(len(episodes) == 4 for episodes in replays["tasks"].values())   # Fewer episodes than are usually recorded

    results.mkdir()
    (results / "test-bot.json").write_text(json.dumps(document))
    replay_folder = tmp_path / "replays"
    replay_folder.mkdir()
    write_replays(replays, str(replay_folder / ("test-bot" + REPLAYS_SUFFIX)))
    assert submissions_to_score(str(tmp_path / "submissions"), str(results), str(replay_folder)) == []
    assert submissions_to_score(str(tmp_path / "submissions"), str(results), str(tmp_path / "nowhere")) == [folder]

    page = build_site(str(results), str(tmp_path / "site"), str(replay_folder))
    text = open(page, encoding="utf-8").read()
    assert "Test Bot" in text and "Acts at random" in text and "not endorsed by Epic" in text
    assert 'replay.html?bot=test-bot' in text
    assert (tmp_path / "site" / "replays" / ("test-bot" + REPLAYS_SUFFIX)).is_file()
    assert "Empty-net finish" in open(tmp_path / "site" / "replay.html", encoding="utf-8").read()
    assert read_replays(str(replay_folder / ("test-bot" + REPLAYS_SUFFIX))) == replays


def test_the_wrong_key_gives_a_recorded_failure(tmp_path, keys, model_path):
    folder = make(tmp_path, keys, model_path)
    other_private, _ = generate_key_pair()
    document, replays = score_submission(folder, other_private, allow_local=True, episodes=2)
    assert replays is None
    assert "not sealed for this key" in document["error"]
    validate_result_document(document)


def test_a_changed_manifest_is_refused(tmp_path, keys, model_path):
    _, public = keys
    folder = make(tmp_path, keys, model_path)
    path = os.path.join(folder, "submission.json")
    data = json.load(open(path))
    data["sealed_sha256"] = "0" * 64
    json.dump(data, open(path, "w"))
    with pytest.raises(SubmissionError, match="SHA-256"):
        verify_submission(folder, public_key=public, allow_local=True)


def test_extra_files_and_fields_are_refused(tmp_path, keys, model_path):
    folder = make(tmp_path, keys, model_path)
    open(os.path.join(folder, "extra.txt"), "w").write("hello")
    with pytest.raises(SubmissionError, match="may only hold"):
        load_manifest(folder)
    os.remove(os.path.join(folder, "extra.txt"))

    path = os.path.join(folder, "submission.json")
    data = json.load(open(path))
    data["surprise"] = "<script>"
    json.dump(data, open(path, "w"))
    with pytest.raises(SubmissionError, match="not allowed"):
        load_manifest(folder)


def test_web_addresses_must_be_https(tmp_path, keys, model_path):
    folder = make(tmp_path, keys, model_path)
    path = os.path.join(folder, "submission.json")
    data = json.load(open(path))
    data["homepage"] = "javascript:alert(1)"
    json.dump(data, open(path, "w"))
    with pytest.raises(SubmissionError, match="https"):
        load_manifest(folder)


def test_local_files_are_refused_by_default(tmp_path, keys, model_path):
    _, public = keys
    folder = make(tmp_path, keys, model_path)
    with pytest.raises(SubmissionError, match="https"):
        verify_submission(folder, public_key=public)


def test_replay_validation_catches_bad_shapes():
    good = {"slug": "x", "fps": 15, "interface_version": 1, "task_set_version": 1,
            "tasks": {"save": [{"episode": 0, "outcome": "success", "seconds": 1.0,
                                "frames": [{"ball": [0, 0, 93.15], "cars": [[0, 0, 17, 1, 0, 0, 0, 0, 1, 1]]}]}]}}
    validate_replays_document(good)
    bad = json.loads(json.dumps(good))
    bad["tasks"]["save"][0]["frames"][0]["cars"][0][0] = "nan"
    with pytest.raises(SubmissionError, match="number"):
        validate_replays_document(bad)
    bad = json.loads(json.dumps(good))
    bad["tasks"]["evil"] = []
    with pytest.raises(SubmissionError, match="unknown tasks"):
        validate_replays_document(bad)
    bad = json.loads(json.dumps(good))
    bad["tasks"]["save"][0]["note"] = "<script>"
    with pytest.raises(SubmissionError, match="exactly"):
        validate_replays_document(bad)


def test_result_validation_catches_bad_shapes():
    with pytest.raises(SubmissionError):
        validate_result_document({"scored_at": "x", "benchmark_version": "y", "manifest_sha256": "z", "results": [], "overall_score": 1, "official": True})
    with pytest.raises(SubmissionError, match="Unexpected"):
        validate_result_document({"scored_at": "x", "benchmark_version": "y", "manifest_sha256": "z", "error": "e", "evil": 1})


def test_names_in_the_page_are_escaped(tmp_path, keys, model_path):
    private, _ = keys
    folder = make(tmp_path, keys, model_path, name='<img src=x onerror=alert(1)> Bot')
    document, _ = score_submission(folder, private, allow_local=True, episodes=2)
    results = tmp_path / "results"
    results.mkdir()
    (results / "img-src-x-onerror-alert-1-bot.json").write_text(json.dumps(document))
    text = open(build_site(str(results), str(tmp_path / "site")), encoding="utf-8").read()
    assert "<img src=x" not in text and "&lt;img src=x" in text
