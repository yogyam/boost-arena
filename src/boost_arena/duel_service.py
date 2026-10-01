"""Duels on the scoring service: which pairs still need playing, playing them, and checking the results.

A duel result is `duels/<a>__<b>.json`, with the slugs in alphabetical order. It records the
manifest checksums of both bots, so a bot that is updated plays everyone again.
"""

import datetime
import gzip
import json
import os
import re

from . import __version__
from .duels import DUEL_KINDS, DUEL_SET_VERSION, OFFICIAL_EPISODES, run_pair
from .policy import InvalidModel, Policy
from .runner import FRAMES_PER_SECOND, OFFICIAL_SEED, SEASON
from .submissions import (
    MAX_REPLAY_FRAMES,
    MODEL_SUFFIX,
    RECORDED_EPISODES,
    REPLAYS_SUFFIX,
    SLUG_PATTERN,
    SubmissionError,
    manifest_digest,
    open_submission,
    submission_folders,
)

DUEL_REPLAYS = 2  # Per kind and direction


def pair_name(a: str, b: str) -> str:
    a, b = sorted((a, b))
    return f"{a}__{b}"


def scored_bots(submissions_folder: str, results_folder: str) -> list:
    """Slugs of the bots that have a successful, current result, in alphabetical order."""
    bots = []
    for folder in submission_folders(submissions_folder):
        slug = os.path.basename(folder)
        result_path = os.path.join(results_folder, f"{slug}.json")
        if not os.path.isfile(result_path):
            continue
        with open(result_path, encoding="utf-8") as f:
            try:
                document = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
        if "error" not in document and document.get("manifest_sha256") == manifest_digest(folder):
            bots.append(slug)
    return bots


def pairs_to_play(submissions_folder: str, results_folder: str, duels_folder: str) -> list:
    """Pairs of scored bots with no duel result, or whose result is from an older version of either bot."""
    bots = scored_bots(submissions_folder, results_folder)
    digests = {slug: manifest_digest(os.path.join(submissions_folder, slug)) for slug in bots}
    pending = []
    for i, a in enumerate(bots):
        for b in bots[i + 1 :]:
            path = os.path.join(duels_folder, pair_name(a, b) + ".json")
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    try:
                        existing = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        existing = {}
                if (
                    existing.get("a_manifest_sha256") == digests[a]
                    and existing.get("b_manifest_sha256") == digests[b]
                    and existing.get("duel_set_version") == DUEL_SET_VERSION
                    and existing.get("season") == SEASON
                ):
                    continue
            pending.append((a, b))
    return pending


def open_bots_for_duels(
    submissions_folder: str,
    results_folder: str,
    duels_folder: str,
    private_key: str,
    models_folder: str,
    max_pairs: int,
    allow_local: bool = False,
) -> list:
    """Opens the models of every bot in the pairs about to be played into `models_folder/<slug>.onnx`,
    unless already there. A bot that cannot be opened is reported and its pairs are skipped."""
    os.makedirs(models_folder, exist_ok=True)
    skipped = []
    for a, b in pairs_to_play(submissions_folder, results_folder, duels_folder)[:max_pairs]:
        for slug in (a, b):
            path = os.path.join(models_folder, slug + MODEL_SUFFIX)
            if os.path.isfile(path) or slug in skipped:
                continue
            try:
                _, model, _ = open_submission(os.path.join(submissions_folder, slug), private_key, allow_local=allow_local)
            except SubmissionError as e:
                print(f"{slug}: cannot duel, {e}")
                skipped.append(slug)
                continue
            with open(path, "wb") as f:
                f.write(model)
    return skipped


def load_bot(models_folder: str, slug: str) -> Policy:
    """A bot's opened model, as left by `open_bots_for_duels`."""
    path = os.path.join(models_folder, slug + MODEL_SUFFIX)
    if not os.path.isfile(path):
        raise SubmissionError(f"{slug}: the model was not opened")
    try:
        return Policy.from_file(path)
    except InvalidModel as e:
        raise SubmissionError(f"{slug}: the model is not acceptable: {e}") from None


def play_pair(submissions_folder: str, a: str, b: str, policies: dict, episodes: int = None) -> tuple:
    """Plays one pair. Returns the duel document and the replays document, or raises SubmissionError
    if one of the models fails: a duel is never published half-played."""
    a, b = sorted((a, b))
    try:
        result = run_pair(policies[a], policies[b], episodes=episodes or OFFICIAL_EPISODES, record_first=DUEL_REPLAYS)
    except Exception as e:  # A stranger's model; whatever it does, the service goes on
        raise SubmissionError(f"{a} v {b}: a model failed during the duel ({type(e).__name__})") from None
    replays = result.pop("replays")
    document = {
        "a": a,
        "b": b,
        "a_manifest_sha256": manifest_digest(os.path.join(submissions_folder, a)),
        "b_manifest_sha256": manifest_digest(os.path.join(submissions_folder, b)),
        "played_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark_version": __version__,
        **result,
    }
    replay_document = {
        "pair": pair_name(a, b),
        "a": a,
        "b": b,
        "fps": FRAMES_PER_SECOND,
        "duel_set_version": DUEL_SET_VERSION,
        "season": SEASON,
        "kinds": replays,
    }
    return document, replay_document


def write_duel_replays(replays: dict, path: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(replays, f, separators=(",", ":"))


def read_duel_replays(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _check_slug(value):
    if not isinstance(value, str) or not SLUG_PATTERN.match(value):
        raise SubmissionError("A duel names a bot with a bad slug")


def _whole(value) -> bool:
    return type(value) is int and value >= 0


def validate_duel_document(document: dict, official_only: bool = False) -> None:
    """Checks a duel result from the scoring job before it is published: the right shape, and
    numbers that agree with each other."""
    if not isinstance(document, dict):
        raise SubmissionError("A duel result must be a JSON object")
    allowed = {
        "a",
        "b",
        "a_manifest_sha256",
        "b_manifest_sha256",
        "played_at",
        "benchmark_version",
        "episodes_per_direction",
        "seed",
        "season",
        "duel_set_version",
        "official",
        "kinds",
        "a_points",
        "b_points",
    }
    if set(document) != allowed:
        raise SubmissionError(f"A duel result has the wrong fields: {sorted(set(document) ^ allowed)}")
    _check_slug(document["a"])
    _check_slug(document["b"])
    if document["a"] >= document["b"]:
        raise SubmissionError("A duel must name the bots in alphabetical order")
    for field in ("a_manifest_sha256", "b_manifest_sha256"):
        if not isinstance(document[field], str) or not re.fullmatch(r"[0-9a-f]{64}", document[field]):
            raise SubmissionError(f"'{field}' must be a SHA-256")
    for field in ("played_at", "benchmark_version"):
        if not isinstance(document[field], str) or len(document[field]) > 64:
            raise SubmissionError(f"'{field}' is missing or not short text")
    for field in ("episodes_per_direction", "seed", "season", "duel_set_version", "a_points", "b_points"):
        if not _whole(document[field]):
            raise SubmissionError(f"'{field}' must be a whole number")
    if type(document["official"]) is not bool:
        raise SubmissionError("'official' must be true or false")
    if (
        document["episodes_per_direction"] == 0
        or document["duel_set_version"] != DUEL_SET_VERSION
        or document["season"] != SEASON
    ):
        raise SubmissionError("The duel is from another duel set or season")
    if document["official"] != (document["episodes_per_direction"] == OFFICIAL_EPISODES and document["seed"] == OFFICIAL_SEED):
        raise SubmissionError("'official' does not match the duel's settings")
    if official_only and not document["official"]:
        raise SubmissionError("Only official duels are published")
    if not isinstance(document["kinds"], dict) or set(document["kinds"]) != set(DUEL_KINDS):
        raise SubmissionError("A duel result must hold every duel kind")
    total_a = total_b = 0
    for kind in document["kinds"].values():
        if not isinstance(kind, dict) or set(kind) != {"a_points", "b_points", "draws", "a_as_blue", "b_as_blue"}:
            raise SubmissionError("A duel kind has the wrong fields")
        for field in ("a_points", "b_points", "draws"):
            if not _whole(kind[field]):
                raise SubmissionError(f"'{field}' must be a whole number")
        halves = {"a": 0, "b": 0, "draws": 0}
        for half in ("a_as_blue", "b_as_blue"):
            if not isinstance(kind[half], dict) or set(kind[half]) != {"a", "b", "draws"}:
                raise SubmissionError("A duel half has the wrong fields")
            for key, value in kind[half].items():
                if not _whole(value):
                    raise SubmissionError("Duel points must be whole numbers")
                halves[key] += value
            if sum(kind[half].values()) != document["episodes_per_direction"]:
                raise SubmissionError("A duel half does not add up to the episodes played")
        if (halves["a"], halves["b"], halves["draws"]) != (kind["a_points"], kind["b_points"], kind["draws"]):
            raise SubmissionError("Duel halves do not add up to the kind's totals")
        total_a += kind["a_points"]
        total_b += kind["b_points"]
    if (total_a, total_b) != (document["a_points"], document["b_points"]):
        raise SubmissionError("Duel totals do not match the kinds")


def validate_published_duel(path: str, submissions_folder: str, results_folder: str, official_only: bool = True) -> dict:
    """A duel file about to be published must be named after its pair and describe both bots as
    they are scored in the repository."""
    with open(path, encoding="utf-8") as f:
        document = json.load(f)
    validate_duel_document(document, official_only=official_only)
    if os.path.basename(path) != pair_name(document["a"], document["b"]) + ".json":
        raise SubmissionError("A duel file must be named after its pair")
    scored = scored_bots(submissions_folder, results_folder)
    for side in ("a", "b"):
        slug = document[side]
        if slug not in scored:
            raise SubmissionError(f"'{slug}' has no current score, so it cannot have duelled")
        if document[f"{side}_manifest_sha256"] != manifest_digest(os.path.join(submissions_folder, slug)):
            raise SubmissionError(f"The duel is not for the manifest of '{slug}' in the repository")
    return document


def validate_published_duel_replays(path: str) -> dict:
    document = read_duel_replays(path)
    validate_duel_replays_document(document)
    if os.path.basename(path) != document["pair"] + REPLAYS_SUFFIX:
        raise SubmissionError("A duel replays file must be named after its pair")
    return document


def validate_duel_replays_document(document: dict) -> None:
    if not isinstance(document, dict) or set(document) != {"pair", "a", "b", "fps", "duel_set_version", "season", "kinds"}:
        raise SubmissionError("A duel replays file has the wrong fields")
    _check_slug(document["a"])
    _check_slug(document["b"])
    if document["pair"] != pair_name(document["a"], document["b"]):
        raise SubmissionError("The duel replays name does not match its bots")
    expected = {f"{kind}_{side}" for kind in DUEL_KINDS for side in ("a_blue", "b_blue")}
    if not isinstance(document["kinds"], dict) or set(document["kinds"]) - expected:
        raise SubmissionError("Duel replays refer to unknown duel kinds")
    for replays in document["kinds"].values():
        if not isinstance(replays, list) or len(replays) > max(DUEL_REPLAYS, RECORDED_EPISODES):
            raise SubmissionError("Too many duel replays")
        for replay in replays:
            if not isinstance(replay, dict) or set(replay) != {"episode", "frames", "outcome", "seconds"}:
                raise SubmissionError("A duel replay has the wrong fields")
            if replay["outcome"] not in ("blue", "orange", "draw"):
                raise SubmissionError("A duel replay has a bad outcome")
            frames = replay["frames"]
            if not isinstance(frames, list) or not (1 <= len(frames) <= MAX_REPLAY_FRAMES):
                raise SubmissionError("A duel replay has no frames or too many")
            for frame in frames:
                if not isinstance(frame, dict) or set(frame) != {"ball", "cars"} or len(frame["cars"]) != 2:
                    raise SubmissionError("A duel replay frame has the wrong shape")
                for values in [frame["ball"]] + frame["cars"]:
                    if not isinstance(values, list) or not all(
                        isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) < 1e5 for v in values
                    ):
                        raise SubmissionError("A duel replay frame holds something that is not a sensible number")


def duel_points(duels_folder: str) -> dict:
    """`points[(a, b)]` from every duel file: the points a took from b."""
    points = {}
    if not os.path.isdir(duels_folder):
        return points
    for name in sorted(os.listdir(duels_folder)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(duels_folder, name), encoding="utf-8") as f:
            document = json.load(f)
        points[(document["a"], document["b"])] = points.get((document["a"], document["b"]), 0) + document["a_points"]
        points[(document["b"], document["a"])] = points.get((document["b"], document["a"]), 0) + document["b_points"]
    return points
