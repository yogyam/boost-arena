"""Duels on the scoring service: which pairs still need playing, playing them, and checking the results.

A duel result is `duels/<a>__<b>.json`, with the slugs in alphabetical order. It records the
manifest checksums of both bots, so a bot that is updated plays everyone again.
"""

import datetime
import gzip
import json
import os

from . import __version__
from .duels import DUEL_KINDS, DUEL_SET_VERSION, OFFICIAL_EPISODES, run_pair
from .policy import InvalidModel, Policy, check_model
from .sealed import SealedFileError, open_sealed
from .submissions import (
    REPLAYS_SUFFIX, MAX_REPLAY_FRAMES, RECORDED_EPISODES, SLUG_PATTERN, SubmissionError, fetch, load_manifest, manifest_digest, sha256,
)

DUEL_REPLAYS = 2   # Per kind and direction


def pair_name(a: str, b: str) -> str:
    a, b = sorted((a, b))
    return f"{a}__{b}"


def scored_bots(submissions_folder: str, results_folder: str) -> list:
    """Slugs of the bots that have a successful, current result, in alphabetical order."""
    bots = []
    if not os.path.isdir(submissions_folder):
        return bots
    for slug in sorted(os.listdir(submissions_folder)):
        folder = os.path.join(submissions_folder, slug)
        result_path = os.path.join(results_folder, f"{slug}.json")
        if not os.path.isdir(folder) or slug.startswith(".") or not os.path.isfile(result_path):
            continue
        with open(result_path, "r", encoding="utf-8") as f:
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
        for b in bots[i + 1:]:
            path = os.path.join(duels_folder, pair_name(a, b) + ".json")
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        existing = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        existing = {}
                if existing.get("a_manifest_sha256") == digests[a] and existing.get("b_manifest_sha256") == digests[b] \
                        and existing.get("duel_set_version") == DUEL_SET_VERSION:
                    continue
            pending.append((a, b))
    return pending


def open_bot(submissions_folder: str, slug: str, private_key: str, allow_local: bool = False) -> Policy:
    """Downloads and opens a bot's sealed model, checking it the same way scoring does."""
    manifest = load_manifest(os.path.join(submissions_folder, slug))
    sealed = fetch(manifest.model_url, allow_local=allow_local)
    if sha256(sealed) != manifest.sealed_sha256:
        raise SubmissionError(f"{slug}: the sealed file has changed since it was submitted")
    try:
        model = open_sealed(sealed, private_key)
    except SealedFileError as e:
        raise SubmissionError(f"{slug}: {e}") from None
    if sha256(model) != manifest.model_sha256:
        raise SubmissionError(f"{slug}: the opened model does not match the manifest")
    try:
        check_model(model)
        return Policy(model)
    except InvalidModel as e:
        raise SubmissionError(f"{slug}: the model is not acceptable: {e}") from None


def play_pair(submissions_folder: str, a: str, b: str, policies: dict, episodes: int = None) -> tuple:
    """Plays one pair. Returns the duel document and the replays document."""
    a, b = sorted((a, b))
    result = run_pair(policies[a], policies[b], episodes=episodes or OFFICIAL_EPISODES, record_first=DUEL_REPLAYS)
    replays = result.pop("replays")
    document = {
        "a": a, "b": b,
        "a_manifest_sha256": manifest_digest(os.path.join(submissions_folder, a)),
        "b_manifest_sha256": manifest_digest(os.path.join(submissions_folder, b)),
        "played_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark_version": __version__,
        **result,
    }
    replay_document = {"pair": pair_name(a, b), "a": a, "b": b, "fps": 15, "duel_set_version": DUEL_SET_VERSION, "kinds": replays}
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


def validate_duel_document(document: dict) -> None:
    """Checks a duel result from the scoring job before it is published."""
    if not isinstance(document, dict):
        raise SubmissionError("A duel result must be a JSON object")
    allowed = {"a", "b", "a_manifest_sha256", "b_manifest_sha256", "played_at", "benchmark_version", "episodes_per_direction",
               "seed", "duel_set_version", "official", "kinds", "a_points", "b_points"}
    if set(document) != allowed:
        raise SubmissionError(f"A duel result has the wrong fields: {sorted(set(document) ^ allowed)}")
    _check_slug(document["a"])
    _check_slug(document["b"])
    if document["a"] >= document["b"]:
        raise SubmissionError("A duel must name the bots in alphabetical order")
    for field in ("a_manifest_sha256", "b_manifest_sha256", "played_at", "benchmark_version"):
        if not isinstance(document[field], str) or len(document[field]) > 64:
            raise SubmissionError(f"'{field}' is missing or not short text")
    for field in ("episodes_per_direction", "seed", "duel_set_version", "a_points", "b_points"):
        if not isinstance(document[field], int) or document[field] < 0:
            raise SubmissionError(f"'{field}' must be a whole number")
    if not isinstance(document["official"], bool):
        raise SubmissionError("'official' must be true or false")
    if not isinstance(document["kinds"], dict) or set(document["kinds"]) != set(DUEL_KINDS):
        raise SubmissionError("A duel result must hold every duel kind")
    total_a = total_b = 0
    for kind in document["kinds"].values():
        if not isinstance(kind, dict) or set(kind) != {"a_points", "b_points", "draws", "a_as_blue", "b_as_blue"}:
            raise SubmissionError("A duel kind has the wrong fields")
        for field in ("a_points", "b_points", "draws"):
            if not isinstance(kind[field], int) or kind[field] < 0:
                raise SubmissionError(f"'{field}' must be a whole number")
        for half in ("a_as_blue", "b_as_blue"):
            if not isinstance(kind[half], dict) or set(kind[half]) != {"a", "b", "draws"}:
                raise SubmissionError("A duel half has the wrong fields")
            for value in kind[half].values():
                if not isinstance(value, int) or value < 0:
                    raise SubmissionError("Duel points must be whole numbers")
        if kind["a_points"] + kind["b_points"] + kind["draws"] != 2 * document["episodes_per_direction"]:
            raise SubmissionError("Duel points do not add up to the episodes played")
        total_a += kind["a_points"]
        total_b += kind["b_points"]
    if (total_a, total_b) != (document["a_points"], document["b_points"]):
        raise SubmissionError("Duel totals do not match the kinds")


def validate_duel_replays_document(document: dict) -> None:
    if not isinstance(document, dict) or set(document) != {"pair", "a", "b", "fps", "duel_set_version", "kinds"}:
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
                    if not isinstance(values, list) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v) < 1e5 for v in values):
                        raise SubmissionError("A duel replay frame holds something that is not a sensible number")


def duel_points(duels_folder: str) -> dict:
    """`points[(a, b)]` from every duel file: the points a took from b."""
    points = {}
    if not os.path.isdir(duels_folder):
        return points
    for name in sorted(os.listdir(duels_folder)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(duels_folder, name), "r", encoding="utf-8") as f:
            document = json.load(f)
        points[(document["a"], document["b"])] = points.get((document["a"], document["b"]), 0) + document["a_points"]
        points[(document["b"], document["a"])] = points.get((document["b"], document["a"]), 0) + document["b_points"]
    return points
