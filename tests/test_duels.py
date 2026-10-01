"""Checks duels, their rating, and the service's bookkeeping."""

import json
import os

import pytest

from boost_arena.duel_service import duel_points, pairs_to_play, play_pair, validate_duel_document, validate_duel_replays_document
from boost_arena.duels import DUEL_KINDS, play_duel, run_pair
from boost_arena.policy import Policy, uniform_model
from boost_arena.rating import expected_share, fit_strengths, ratings
from boost_arena.submissions import SubmissionError


@pytest.fixture(scope="module")
def random_bot():
    return Policy(uniform_model())


def test_points_add_up_and_repeat(random_bot):
    first = play_duel("penalty", random_bot, random_bot, episodes=6, arenas=3)
    assert first.episodes == 6
    assert first.blue_points + first.orange_points + first.draws == 6
    assert first.draws == 0   # In a penalty duel the keeper gets the point when nobody scores
    second = play_duel("penalty", random_bot, random_bot, episodes=6, arenas=6)
    assert (first.blue_points, first.orange_points) == (second.blue_points, second.orange_points)

    kickoff = play_duel("kickoff", random_bot, random_bot, episodes=4, arenas=2, record_first=2)
    assert kickoff.episodes == 4 and len(kickoff.replays) == 2
    assert all(len(r["frames"][0]["cars"]) == 2 for r in kickoff.replays)


def test_a_pair_plays_both_ways(random_bot):
    result = run_pair(random_bot, random_bot, episodes=3, arenas=3)
    assert set(result["kinds"]) == set(DUEL_KINDS)
    for kind in result["kinds"].values():
        assert kind["a_points"] + kind["b_points"] + kind["draws"] == 6
    assert result["a_points"] == sum(k["a_points"] for k in result["kinds"].values())
    assert result["official"] is False


def test_rating_orders_bots_by_strength():
    points = {("strong", "middle"): 80, ("middle", "strong"): 20, ("middle", "weak"): 75, ("weak", "middle"): 25,
              ("strong", "weak"): 95, ("weak", "strong"): 5}
    rating = ratings(points)
    assert rating["strong"] > rating["middle"] > rating["weak"]
    assert 0.7 < expected_share(rating["strong"], rating["middle"]) < 0.9
    assert abs(sum(fit_strengths(points).values()) / 3 - 1) < 1   # Normalised around 1


def test_an_unbeaten_bot_gets_a_finite_rating():
    rating = ratings({("a", "b"): 50, ("b", "a"): 0})
    assert all(abs(v) < 10_000 for v in rating.values())


def test_service_bookkeeping(tmp_path, random_bot):
    submissions = tmp_path / "submissions"
    results = tmp_path / "results"
    duels = tmp_path / "duels"
    for folder in (submissions, results, duels):
        folder.mkdir()
    for slug in ("alpha", "beta", "gamma"):
        (submissions / slug).mkdir()
        (submissions / slug / "submission.json").write_text(json.dumps({"slug": slug}))
    import hashlib
    for slug in ("alpha", "beta"):   # gamma has no result, so it does not duel
        digest = hashlib.sha256((submissions / slug / "submission.json").read_bytes()).hexdigest()
        (results / f"{slug}.json").write_text(json.dumps({"manifest_sha256": digest, "results": []}))

    assert pairs_to_play(str(submissions), str(results), str(duels)) == [("alpha", "beta")]

    document, replays = play_pair(str(submissions), "beta", "alpha", {"alpha": random_bot, "beta": random_bot}, episodes=2)
    assert document["a"] == "alpha" and document["b"] == "beta"
    validate_duel_document(document)
    validate_duel_replays_document(replays)
    (duels / "alpha__beta.json").write_text(json.dumps(document))
    assert pairs_to_play(str(submissions), str(results), str(duels)) == []
    assert duel_points(str(duels)) == {("alpha", "beta"): document["a_points"], ("beta", "alpha"): document["b_points"]}

    # An updated bot plays again, once it has been scored again
    (submissions / "alpha" / "submission.json").write_text(json.dumps({"slug": "alpha", "v": 2}))
    assert pairs_to_play(str(submissions), str(results), str(duels)) == []   # Not scored yet, so it sits out
    digest = hashlib.sha256((submissions / "alpha" / "submission.json").read_bytes()).hexdigest()
    (results / "alpha.json").write_text(json.dumps({"manifest_sha256": digest, "results": []}))
    assert pairs_to_play(str(submissions), str(results), str(duels)) == [("alpha", "beta")]


def test_validation_refuses_bad_duels():
    with pytest.raises(SubmissionError):
        validate_duel_document({"a": "b", "b": "a"})
    with pytest.raises(SubmissionError, match="fields"):
        validate_duel_replays_document({"pair": "a__b", "a": "a", "b": "b", "fps": 15, "duel_set_version": 1, "kinds": {}, "extra": 1})
