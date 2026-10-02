"""Checks the submission flow from sealing a model to a published result, using local files,
and that what the scoring job hands back is checked before it is published."""

import copy
import json
import os

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from boost_arena.policy import uniform_model
from boost_arena.runner import OFFICIAL_EPISODES, OFFICIAL_SEED, SEASON
from boost_arena.sealed import generate_key_pair
from boost_arena.site import build_site
from boost_arena.submissions import (
    MODEL_SUFFIX,
    REPLAYS_SUFFIX,
    SubmissionError,
    check_not_a_copy,
    check_pull_request,
    load_manifest,
    make_submission,
    open_submissions,
    read_replays,
    score_model,
    score_submission,
    submissions_to_score,
    validate_published_result,
    validate_replays_document,
    validate_result_document,
    verify_submission,
    write_replays,
)


@pytest.fixture
def keys():
    return generate_key_pair()


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "bot.onnx"
    path.write_bytes(uniform_model())
    return str(path)


def make(tmp_path, keys, model_path, name="Test Bot", github="tester", **kwargs):
    _, public = keys
    submissions = tmp_path / "submissions"
    manifest = make_submission(model_path, public, name, "Tester", github, str(submissions), **kwargs)
    folder = submissions / manifest.slug
    # The sealed file is moved out of the folder, as if hosted elsewhere, and the manifest points at it
    sealed = submissions / f"{manifest.slug}.sealed"
    hosted = tmp_path / "hosted"
    hosted.mkdir(exist_ok=True)
    sealed.rename(hosted / sealed.name)
    data = json.loads((folder / "submission.json").read_text())
    data["model_url"] = "file://" + str(hosted / sealed.name)
    (folder / "submission.json").write_text(json.dumps(data))
    return str(folder)


def edit_manifest(folder, **changes):
    path = os.path.join(folder, "submission.json")
    with open(path) as f:
        data = json.load(f)
    data.update(changes)
    with open(path, "w") as f:
        json.dump(data, f)


def test_submit_verify_score_and_publish(tmp_path, keys, model_path):
    private, public = keys
    folder = make(tmp_path, keys, model_path, description="Acts at random")

    manifest = load_manifest(folder)
    assert manifest.slug == "test-bot" and manifest.description == "Acts at random" and manifest.github == "tester"
    assert verify_submission(folder, public_key=public, allow_local=True)["name"] == "Test Bot"

    results = tmp_path / "results"
    assert submissions_to_score(str(tmp_path / "submissions"), str(results)) == [folder]

    document, replays = score_submission(folder, private, allow_local=True, episodes=4)
    assert "error" not in document, document.get("error")
    assert len(document["results"]) == 6
    assert 0 <= document["overall_score"] <= 100
    assert document["official"] is False  # Only four episodes
    validate_result_document(document)
    validate_replays_document(replays)
    assert all(len(episodes) == 4 for episodes in replays["tasks"].values())  # Fewer episodes than are usually recorded

    results.mkdir()
    (results / "test-bot.json").write_text(json.dumps(document))
    replay_folder = tmp_path / "replays"
    replay_folder.mkdir()
    write_replays(replays, str(replay_folder / ("test-bot" + REPLAYS_SUFFIX)))
    assert submissions_to_score(str(tmp_path / "submissions"), str(results), str(replay_folder)) == []
    assert submissions_to_score(str(tmp_path / "submissions"), str(results), str(tmp_path / "nowhere")) == [folder]

    # A result from another season is scored again
    stale = dict(document, season=SEASON + 1)
    (results / "test-bot.json").write_text(json.dumps(stale))
    assert submissions_to_score(str(tmp_path / "submissions"), str(results), str(replay_folder)) == [folder]
    (results / "test-bot.json").write_text(json.dumps(document))

    page = build_site(str(results), str(tmp_path / "site"), str(replay_folder))
    text = open(page, encoding="utf-8").read()
    assert "Test Bot" in text and "Acts at random" in text and "not endorsed by Epic" in text
    assert "replay.html?bot=test-bot" in text
    assert (tmp_path / "site" / "replays" / ("test-bot" + REPLAYS_SUFFIX)).is_file()
    assert (tmp_path / "site" / "vendor" / "three.module.js").is_file()
    viewer = open(tmp_path / "site" / "replay.html", encoding="utf-8").read()
    assert "Empty-net finish" in viewer and "https://" not in viewer.split("<script")[1]  # No script from other hosts
    assert read_replays(str(replay_folder / ("test-bot" + REPLAYS_SUFFIX))) == replays


def test_the_service_opens_then_scores_without_the_key(tmp_path, keys, model_path):
    """On the service, the step with the private key only opens models; scoring runs without it."""
    private, _ = keys
    folder = make(tmp_path, keys, model_path)
    models = tmp_path / "opened"
    output = tmp_path / "new_results"
    assert open_submissions([folder], private, str(models), str(output), allow_local=True) == ["test-bot"]
    assert (models / ("test-bot" + MODEL_SUFFIX)).read_bytes() == open(model_path, "rb").read()
    started = json.loads((models / "test-bot.json").read_text())
    assert started["manifest"]["slug"] == "test-bot" and "results" not in started

    document, replays = score_model(started, (models / ("test-bot" + MODEL_SUFFIX)).read_bytes(), episodes=2)
    assert "error" not in document and replays["slug"] == "test-bot"

    # A submission that cannot be opened gets its error written for publishing
    other_private, _ = generate_key_pair()
    assert open_submissions([folder], other_private, str(models), str(output), allow_local=True) == []
    failed = json.loads((output / "test-bot.json").read_text())
    assert "not sealed for this key" in failed["error"]
    validate_result_document(failed)


def test_the_wrong_key_gives_a_recorded_failure(tmp_path, keys, model_path):
    folder = make(tmp_path, keys, model_path)
    other_private, _ = generate_key_pair()
    document, replays = score_submission(folder, other_private, allow_local=True, episodes=2)
    assert replays is None
    assert "not sealed for this key" in document["error"]
    validate_result_document(document)


def test_a_model_that_fails_while_running_is_a_recorded_failure(tmp_path, keys, model_path):
    """A model can pass every check and still break on a real observation. That must not stop the service."""
    private, public = keys
    # Log(obs + 0.5): fine on the zero observation used by the checks, NaN once a position goes below -0.5
    weight = numpy_helper.from_array(np.zeros((90, 53), dtype=np.float32), "weight")
    bias = numpy_helper.from_array(np.zeros(90, dtype=np.float32), "bias")
    half = numpy_helper.from_array(np.array(0.5, dtype=np.float32), "half")
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["obs", "half"], ["shifted"]),
            helper.make_node("Log", ["shifted"], ["logged"]),
            helper.make_node("Gemm", ["logged", "weight", "bias"], ["logits"], transB=1),
        ],
        "fragile",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["N", 53])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["N", 90])],
        [weight, bias, half],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "fragile.onnx"
    path.write_bytes(model.SerializeToString())

    folder = make(tmp_path, keys, str(path), name="Fragile")
    document, replays = score_submission(folder, private, allow_local=True, episodes=2)
    assert replays is None
    assert "not a finite number" in document["error"] or "failed while being scored" in document["error"]
    validate_result_document(document)


def test_a_copied_submission_is_refused(tmp_path, keys, model_path):
    """Pointing a new manifest at someone else's sealed file would claim their score."""
    private, public = keys
    theirs = make(tmp_path, keys, model_path, name="Their Bot", github="them")
    their_manifest = json.load(open(os.path.join(theirs, "submission.json")))

    mine = make(tmp_path, keys, model_path, name="My Bot", github="me")
    edit_manifest(
        mine,
        model_url=their_manifest["model_url"],
        sealed_sha256=their_manifest["sealed_sha256"],
        model_sha256=their_manifest["model_sha256"],
        model_bytes=their_manifest["model_bytes"],
    )
    with pytest.raises(SubmissionError, match="already submitted"):
        check_not_a_copy(load_manifest(mine), str(tmp_path / "submissions"))
    with pytest.raises(SubmissionError, match="already submitted"):
        verify_submission(mine, public_key=public, allow_local=True, submissions_folder=str(tmp_path / "submissions"))

    # Even past that check, the sealed file does not open for another submission
    document, _ = score_submission(mine, private, allow_local=True, episodes=2)
    assert "not sealed for this key and this submission" in document["error"]

    # And the same name as another bot is refused too
    edit_manifest(mine, name="their bot", model_sha256="1" * 64, sealed_sha256="2" * 64)
    with pytest.raises(SubmissionError, match="already used"):
        check_not_a_copy(load_manifest(mine), str(tmp_path / "submissions"))


def test_pull_request_checks(tmp_path, keys, model_path):
    folder = make(tmp_path, keys, model_path, github="tester")
    submissions, results = str(tmp_path / "submissions"), str(tmp_path / "results")
    check_pull_request([folder], "tester", submissions, results)
    check_pull_request([folder], "Tester", submissions, results)  # Logins are not case sensitive
    with pytest.raises(SubmissionError, match="pull request is from"):
        check_pull_request([folder], "someone-else", submissions, results)
    with pytest.raises(SubmissionError, match="valid GitHub login"):
        check_pull_request([folder], "", submissions, results)

    # Three scorings in the period are the limit, counting the ones this pull request asks for
    os.makedirs(results)
    for i in range(2):
        other = make(tmp_path, keys, model_path, name=f"Older {i}", github="tester")
        (tmp_path / "results" / f"older-{i}.json").write_text(
            json.dumps({"scored_at": "2999-01-01T00:00:00Z", "manifest": load_manifest(other).to_dict()})
        )
    check_pull_request([folder], "tester", submissions, results)
    third = make(tmp_path, keys, model_path, name="Older 2", github="tester")
    (tmp_path / "results" / "older-2.json").write_text(
        json.dumps({"scored_at": "2999-01-01T00:00:00Z", "manifest": load_manifest(third).to_dict()})
    )
    with pytest.raises(SubmissionError, match="limit"):
        check_pull_request([folder], "tester", submissions, results)
    check_pull_request([folder], "tester", submissions, results, exempt=("Tester",))  # Baselines are exempt
    # Updating one of those three is fine: it is scored again, not in addition
    check_pull_request([third], "tester", submissions, results)
    # Old scorings do not count
    (tmp_path / "results" / "older-2.json").write_text(
        json.dumps({"scored_at": "2000-01-01T00:00:00Z", "manifest": load_manifest(third).to_dict()})
    )
    check_pull_request([folder], "tester", submissions, results)


def test_make_submission_refuses_what_the_pull_request_would(tmp_path, keys, model_path):
    _, public = keys
    with pytest.raises(SubmissionError, match="GitHub login"):
        make_submission(model_path, public, "Bot", "Tester", "not a login!", str(tmp_path / "s"))
    with pytest.raises(SubmissionError, match="https"):
        make_submission(model_path, public, "Bot", "Tester", "tester", str(tmp_path / "s"), homepage="http://x")
    with pytest.raises(SubmissionError, match="characters"):
        make_submission(model_path, public, "B" * 41, "Tester", "tester", str(tmp_path / "s"))


def test_a_changed_manifest_is_refused(tmp_path, keys, model_path):
    _, public = keys
    folder = make(tmp_path, keys, model_path)
    edit_manifest(folder, sealed_sha256="0" * 64)
    with pytest.raises(SubmissionError, match="SHA-256"):
        verify_submission(folder, public_key=public, allow_local=True)


def test_extra_files_and_fields_are_refused(tmp_path, keys, model_path):
    folder = make(tmp_path, keys, model_path)
    with open(os.path.join(folder, "extra.txt"), "w") as f:
        f.write("hello")
    with pytest.raises(SubmissionError, match="may only hold"):
        load_manifest(folder)
    os.remove(os.path.join(folder, "extra.txt"))

    edit_manifest(folder, surprise="<script>")
    with pytest.raises(SubmissionError, match="not allowed"):
        load_manifest(folder)


def test_web_addresses_must_be_https(tmp_path, keys, model_path):
    folder = make(tmp_path, keys, model_path)
    edit_manifest(folder, homepage="javascript:alert(1)")
    with pytest.raises(SubmissionError, match="https"):
        load_manifest(folder)


def test_local_files_are_refused_by_default(tmp_path, keys, model_path):
    _, public = keys
    folder = make(tmp_path, keys, model_path)
    with pytest.raises(SubmissionError, match="https"):
        verify_submission(folder, public_key=public)


def test_downloads_only_from_public_https(tmp_path, keys, model_path):
    from boost_arena.submissions import fetch

    for url in (
        "http://example.com/x",
        "https://localhost/x",
        "https://127.0.0.1/x",
        "https://169.254.169.254/latest",
        "https://10.0.0.1/x",
        "ftp://example.com/x",
    ):
        with pytest.raises(SubmissionError):
            fetch(url)


def test_replay_validation_catches_bad_shapes():
    good = {
        "slug": "x",
        "fps": 15,
        "interface_version": 1,
        "task_set_version": 1,
        "season": SEASON,
        "tasks": {
            "save": [
                {
                    "episode": 0,
                    "outcome": "success",
                    "seconds": 1.0,
                    "frames": [{"ball": [0, 0, 93.15], "cars": [[0, 0, 17, 1, 0, 0, 0, 0, 1, 1]]}],
                }
            ]
        },
    }
    validate_replays_document(good)
    bad = json.loads(json.dumps(good))
    bad["tasks"]["save"][0]["frames"][0]["cars"][0][0] = "nan"
    with pytest.raises(SubmissionError, match="number"):
        validate_replays_document(bad)
    bad = json.loads(json.dumps(good))
    bad["tasks"]["save"][0]["frames"][0]["cars"][0][0] = True
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


@pytest.fixture(scope="module")
def official_document(tmp_path_factory):
    """A real result document, patched to look official so the publish checks can be tried on it."""
    tmp_path = tmp_path_factory.mktemp("official")
    private, public = generate_key_pair()
    path = tmp_path / "bot.onnx"
    path.write_bytes(uniform_model())
    folder = make(tmp_path, (private, public), str(path), name="Official Bot")
    document, _ = score_submission(folder, private, allow_local=True, episodes=2)
    assert "error" not in document
    for result in document["results"]:
        result["episodes"] = OFFICIAL_EPISODES
        result["successes"] = round(result["success_rate"] * OFFICIAL_EPISODES)
        result["conceded"] = 0
        result["timeouts"] = OFFICIAL_EPISODES - result["successes"]
        result["success_rate"] = result["successes"] / OFFICIAL_EPISODES
        result["success_rate_low"], result["success_rate_high"] = 0.0, 1.0
        result["seed"] = OFFICIAL_SEED
        result["official"] = True
    document["overall_score"] = 100 * sum(r["success_rate"] for r in document["results"]) / 6
    document["official"] = True
    validate_result_document(document, official_only=True)
    return document, folder


def forged(document, **changes):
    copy_ = copy.deepcopy(document)
    copy_.update(changes)
    return copy_


def test_result_validation_catches_forged_documents(official_document):
    """The scoring job has run a stranger's model, so its output is checked for consistency, not just shape."""
    document, _ = official_document
    fine = forged(document)
    validate_result_document(fine, official_only=True)

    with pytest.raises(SubmissionError, match="Unexpected"):
        validate_result_document(forged(document, evil=1))
    with pytest.raises(SubmissionError, match="average"):
        validate_result_document(forged(document, overall_score=100))
    with pytest.raises(SubmissionError, match="season"):
        validate_result_document(forged(document, season=SEASON + 1))
    with pytest.raises(SubmissionError, match="both"):
        validate_result_document(forged(document, error="x"))

    bad = forged(document)
    bad["results"][0]["task"] = "not_a_task"
    with pytest.raises(SubmissionError, match="every task once"):
        validate_result_document(bad)
    bad = forged(document)
    bad["results"].pop()
    with pytest.raises(SubmissionError, match="every task once"):
        validate_result_document(bad)
    bad = forged(document)
    bad["results"][0]["success_rate"] = 1.0
    with pytest.raises(SubmissionError, match="does not match the counts|interval"):
        validate_result_document(bad)
    bad = forged(document)
    bad["results"][0]["episodes"] = True
    with pytest.raises(SubmissionError, match="whole number"):
        validate_result_document(bad)
    bad = forged(document)
    bad["results"][0]["seed"] = OFFICIAL_SEED + 1
    with pytest.raises(SubmissionError, match="official settings"):
        validate_result_document(bad)
    bad = forged(document)
    bad["results"][0]["official"] = False
    with pytest.raises(SubmissionError, match="'official' does not match"):
        validate_result_document(bad)
    bad = forged(document)
    bad["results"][0]["official"] = False
    bad["official"] = False
    validate_result_document(bad)
    with pytest.raises(SubmissionError, match="Only official"):
        validate_result_document(bad, official_only=True)
    bad = forged(document)
    bad["manifest"]["slug"] = "someone-elses-bot"
    validate_result_document(bad)  # The shape is fine; the publish check below catches it


def test_published_results_must_match_the_repository(tmp_path, official_document):
    document, folder = official_document
    submissions = os.path.dirname(folder)
    results = tmp_path / "new_results"
    results.mkdir()

    path = results / "official-bot.json"
    path.write_text(json.dumps(document))
    validate_published_result(str(path), submissions)

    (results / "other-bot.json").write_text(json.dumps(document))
    with pytest.raises(SubmissionError, match="no submission"):
        validate_published_result(str(results / "other-bot.json"), submissions)

    path.write_text(json.dumps(forged(document, manifest_sha256="0" * 64)))
    with pytest.raises(SubmissionError, match="not for the manifest"):
        validate_published_result(str(path), submissions)

    changed = forged(document)
    changed["manifest"]["description"] = "a better description"
    path.write_text(json.dumps(changed))
    with pytest.raises(SubmissionError, match="differs"):
        validate_published_result(str(path), submissions)


def test_names_in_the_page_are_escaped(tmp_path, keys, model_path):
    private, _ = keys
    folder = make(tmp_path, keys, model_path, name="<img src=x onerror=alert(1)> Bot")
    document, _ = score_submission(folder, private, allow_local=True, episodes=2)
    results = tmp_path / "results"
    results.mkdir()
    (results / "img-src-x-onerror-alert-1-bot.json").write_text(json.dumps(document))
    text = open(build_site(str(results), str(tmp_path / "site")), encoding="utf-8").read()
    assert "<img src=x" not in text and "&lt;img src=x" in text


def test_a_flagged_entry_is_marked(tmp_path, keys, model_path):
    private, _ = keys
    folder = make(tmp_path, keys, model_path)
    document, _ = score_submission(folder, private, allow_local=True, episodes=2)
    results = tmp_path / "results"
    results.mkdir()
    (results / "test-bot.json").write_text(json.dumps(document))
    flags = tmp_path / "flags.json"
    flags.write_text(json.dumps({"test-bot": "https://github.com/yogyam/boost-arena/issues/1", "other": "javascript:x"}))
    text = open(build_site(str(results), str(tmp_path / "site"), flags_path=str(flags)), encoding="utf-8").read()
    assert 'class="flag" href="https://github.com/yogyam/boost-arena/issues/1"' in text
