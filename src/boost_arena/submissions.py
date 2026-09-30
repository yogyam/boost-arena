"""Submissions: making one, checking one, and scoring the ones that are new.

A submission is a folder `submissions/<slug>/` holding one file, `submission.json`. It names
the bot and points at the sealed model file, which the entrant hosts anywhere public. The
scoring service opens the sealed file with the project's private key, scores the model and
writes `results/<slug>.json`. The model itself is never stored by the project.
"""

import datetime
import gzip
import hashlib
import json
import os
import re
import urllib.request
from dataclasses import dataclass

from . import __version__, interface
from .policy import InvalidModel, Policy, check_model
from .sealed import SealedFileError, is_sealed, open_sealed, seal

MANIFEST_NAME = "submission.json"
REPLAYS_SUFFIX = ".replays.json.gz"
RECORDED_EPISODES = 5          # Per task. Every bot's replays are of the same situations
MAX_REPLAY_BYTES = 4 * 1024 * 1024
MAX_REPLAY_FRAMES = 2000
MAX_SEALED_BYTES = 64 * 1024 * 1024 + 4096
DOWNLOAD_TIMEOUT = 120
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MAX_NAME = 40
MAX_TEXT = 300


class SubmissionError(Exception):
    """Something about the submission is not right. The message says what."""


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not SLUG_PATTERN.match(slug):
        raise SubmissionError("The bot name must contain at least one letter or digit")
    return slug[:MAX_NAME].rstrip("-")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Manifest:
    name: str
    slug: str
    author: str
    model_url: str
    sealed_sha256: str
    model_sha256: str
    model_bytes: int
    sealed_with: str          # The public key the model was sealed with
    interface_version: int
    submitted: str            # Date, YYYY-MM-DD
    description: str = ""
    homepage: str = ""
    public_model_url: str = ""  # If the entrant chooses to publish their model themselves

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict):
        if not isinstance(data, dict):
            raise SubmissionError("The manifest must be a JSON object")
        required = {"name", "slug", "author", "model_url", "sealed_sha256", "model_sha256", "model_bytes",
                    "sealed_with", "interface_version", "submitted"}
        optional = {"description", "homepage", "public_model_url"}
        missing = required - set(data)
        unknown = set(data) - required - optional
        if missing:
            raise SubmissionError(f"The manifest is missing: {', '.join(sorted(missing))}")
        if unknown:
            raise SubmissionError(f"The manifest has fields that are not allowed: {', '.join(sorted(unknown))}")
        manifest = cls(**data)
        manifest.validate()
        return manifest

    def validate(self):
        for field in ("name", "slug", "author", "model_url", "sealed_sha256", "model_sha256", "sealed_with",
                      "submitted", "description", "homepage", "public_model_url"):
            if not isinstance(getattr(self, field), str):
                raise SubmissionError(f"'{field}' must be text")
        if not (1 <= len(self.name) <= MAX_NAME) or not self.name.strip():
            raise SubmissionError(f"The bot name must be 1 to {MAX_NAME} characters")
        if not (1 <= len(self.author) <= MAX_NAME):
            raise SubmissionError(f"The author must be 1 to {MAX_NAME} characters")
        for field in ("description", "homepage", "public_model_url"):
            if len(getattr(self, field)) > MAX_TEXT:
                raise SubmissionError(f"'{field}' must be at most {MAX_TEXT} characters")
        if not SLUG_PATTERN.match(self.slug) or len(self.slug) > MAX_NAME:
            raise SubmissionError("The slug must be lowercase letters, digits and single hyphens")
        for field in ("model_url", "homepage", "public_model_url"):
            value = getattr(self, field)
            # A file:// address is only ever accepted by the tests, fetch() refuses it otherwise
            if value and not value.startswith("https://") and not (field == "model_url" and value.startswith("file://")):
                raise SubmissionError(f"'{field}' must be an https:// address")
        for field in ("sealed_sha256", "model_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", getattr(self, field)):
                raise SubmissionError(f"'{field}' must be a 64-character hexadecimal SHA-256")
        if not isinstance(self.model_bytes, int) or not (0 < self.model_bytes <= MAX_SEALED_BYTES):
            raise SubmissionError("'model_bytes' must be a positive size within the file limit")
        if self.interface_version != interface.INTERFACE_VERSION:
            raise SubmissionError(f"Interface version {self.interface_version} is not the current version {interface.INTERFACE_VERSION}")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.submitted):
            raise SubmissionError("'submitted' must be a date, YYYY-MM-DD")


def load_manifest(folder: str) -> Manifest:
    path = os.path.join(folder, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise SubmissionError(f"No {MANIFEST_NAME} in {folder}")
    others = [name for name in os.listdir(folder) if name != MANIFEST_NAME and not name.startswith(".")]
    if others:
        raise SubmissionError(f"A submission folder may only hold {MANIFEST_NAME}, found: {', '.join(others)}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise SubmissionError(f"{MANIFEST_NAME} is not valid JSON: {e}") from None
    manifest = Manifest.from_dict(data)
    if manifest.slug != os.path.basename(os.path.normpath(folder)):
        raise SubmissionError(f"The folder is named '{os.path.basename(os.path.normpath(folder))}' but the slug is '{manifest.slug}'")
    return manifest


def manifest_digest(folder: str) -> str:
    with open(os.path.join(folder, MANIFEST_NAME), "rb") as f:
        return sha256(f.read())


# ---- Making a submission ----

def make_submission(model_path: str, public_key: str, name: str, author: str, output_folder: str,
                    description: str = "", homepage: str = "", public_model_url: str = "",
                    model_url: str = "") -> Manifest:
    """Seals the model and writes the sealed file and manifest into `output_folder/<slug>/`.

    `model_url` is where the entrant will host the sealed file. It can be filled in later.
    """
    with open(model_path, "rb") as f:
        model = f.read()
    try:
        check_model(model)
        Policy(model)
    except InvalidModel as e:
        raise SubmissionError(f"The model is not acceptable: {e}") from None

    sealed = seal(model, public_key)
    slug = slugify(name)
    folder = os.path.join(output_folder, slug)
    os.makedirs(folder, exist_ok=True)

    sealed_path = os.path.join(folder, f"{slug}.sealed")
    with open(sealed_path, "wb") as f:
        f.write(sealed)

    manifest = Manifest(
        name=name.strip(), slug=slug, author=author.strip(), model_url=model_url,
        sealed_sha256=sha256(sealed), model_sha256=sha256(model), model_bytes=len(sealed),
        sealed_with=public_key.strip(), interface_version=interface.INTERFACE_VERSION,
        submitted=datetime.date.today().isoformat(),
        description=description.strip(), homepage=homepage.strip(), public_model_url=public_model_url.strip(),
    )
    with open(os.path.join(folder, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
        f.write("\n")
    return manifest


# ---- Fetching and checking ----

def fetch(url: str, allow_local: bool = False) -> bytes:
    """Downloads the sealed file, refusing anything over the size limit."""
    if url.startswith("file://"):
        if not allow_local:
            raise SubmissionError("Local files are not accepted, the sealed model must be at an https:// address")
        with open(url[len("file://"):], "rb") as f:
            return f.read(MAX_SEALED_BYTES + 1)

    if not url.startswith("https://"):
        raise SubmissionError("The sealed model must be at an https:// address")
    request = urllib.request.Request(url, headers={"User-Agent": f"boost-arena/{__version__}"})
    try:
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
            data = response.read(MAX_SEALED_BYTES + 1)
    except Exception as e:
        raise SubmissionError(f"The sealed model could not be downloaded: {e}") from None
    if len(data) > MAX_SEALED_BYTES:
        raise SubmissionError("The sealed model is larger than the limit")
    return data


def verify_submission(folder: str, public_key: str = None, allow_local: bool = False) -> dict:
    """Everything that can be checked without the private key. Returns a short description."""
    manifest = load_manifest(folder)
    if public_key is not None and manifest.sealed_with.strip() != public_key.strip():
        raise SubmissionError("The model was sealed with a different public key. Seal it again with the current one")

    sealed = fetch(manifest.model_url, allow_local=allow_local)
    if len(sealed) != manifest.model_bytes:
        raise SubmissionError(f"The sealed file is {len(sealed)} bytes but the manifest says {manifest.model_bytes}")
    if sha256(sealed) != manifest.sealed_sha256:
        raise SubmissionError("The sealed file does not match the SHA-256 in the manifest")
    if not is_sealed(sealed):
        raise SubmissionError("The file at model_url is not a sealed model")
    return {"slug": manifest.slug, "name": manifest.name, "author": manifest.author, "sealed_bytes": len(sealed)}


# ---- Scoring ----

def score_submission(folder: str, private_key: str, allow_local: bool = False, episodes: int = None, on_progress=None):
    """Opens, checks and scores one submission.

    Returns the result document, which records any failure, and the replays document (or None).
    """
    from .runner import FRAMES_PER_SECOND, OFFICIAL_EPISODES, overall_score, run_task
    from .tasks import TASK_SET_VERSION, TASKS

    document = {
        "scored_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark_version": __version__,
        "manifest_sha256": manifest_digest(folder),
    }
    try:
        manifest = load_manifest(folder)
        document["manifest"] = manifest.to_dict()
        verify_submission(folder, allow_local=allow_local)

        sealed = fetch(manifest.model_url, allow_local=allow_local)
        try:
            model = open_sealed(sealed, private_key)
        except SealedFileError as e:
            raise SubmissionError(str(e)) from None
        if sha256(model) != manifest.model_sha256:
            raise SubmissionError("The opened model does not match the SHA-256 in the manifest")

        try:
            document["model"] = check_model(model)
            policy = Policy(model)
        except InvalidModel as e:
            raise SubmissionError(f"The model is not acceptable: {e}") from None

        results = []
        replays = {"slug": manifest.slug, "fps": FRAMES_PER_SECOND, "interface_version": interface.INTERFACE_VERSION,
                   "task_set_version": TASK_SET_VERSION, "tasks": {}}
        for key, task in TASKS.items():
            result = run_task(policy, task, episodes=episodes or OFFICIAL_EPISODES, on_progress=on_progress,
                              record_first=RECORDED_EPISODES)
            results.append(result)
            replays["tasks"][key] = result.replays
        document["results"] = [r.to_dict() for r in results]
        document["overall_score"] = overall_score(results)
        document["official"] = all(r.official for r in results)
    except SubmissionError as e:
        document["error"] = str(e)
        return document, None
    return document, replays


def write_replays(replays: dict, path: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(replays, f, separators=(",", ":"))


def read_replays(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def validate_replays_document(document: dict) -> None:
    """Checks a replays document from the scoring job: numbers of the right shape, nothing else."""
    from .tasks import TASKS

    if not isinstance(document, dict) or set(document) != {"slug", "fps", "interface_version", "task_set_version", "tasks"}:
        raise SubmissionError("A replays file must hold exactly slug, fps, interface_version, task_set_version and tasks")
    if not isinstance(document["slug"], str) or not SLUG_PATTERN.match(document["slug"]):
        raise SubmissionError("The replays slug is not valid")
    for field in ("fps", "interface_version", "task_set_version"):
        if not isinstance(document[field], int):
            raise SubmissionError(f"'{field}' must be a whole number")
    if not isinstance(document["tasks"], dict) or set(document["tasks"]) - set(TASKS):
        raise SubmissionError("The replays refer to unknown tasks")

    def numbers(values, count):
        if not isinstance(values, list) or len(values) != count:
            raise SubmissionError("A replay frame has the wrong shape")
        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or abs(value) > 1e5:
                raise SubmissionError("A replay frame holds something that is not a sensible number")

    for replays in document["tasks"].values():
        if not isinstance(replays, list) or len(replays) > RECORDED_EPISODES:
            raise SubmissionError("Too many replays for a task")
        for replay in replays:
            if not isinstance(replay, dict) or set(replay) != {"episode", "frames", "outcome", "seconds"}:
                raise SubmissionError("A replay must hold exactly episode, frames, outcome and seconds")
            if not isinstance(replay["episode"], int) or replay["outcome"] not in ("success", "conceded", "timeout"):
                raise SubmissionError("A replay has a bad episode number or outcome")
            if not isinstance(replay["seconds"], (int, float)) or not (0 <= replay["seconds"] <= 120):
                raise SubmissionError("A replay has a bad duration")
            frames = replay["frames"]
            if not isinstance(frames, list) or not (1 <= len(frames) <= MAX_REPLAY_FRAMES):
                raise SubmissionError("A replay has no frames or too many")
            for frame in frames:
                if not isinstance(frame, dict) or set(frame) != {"ball", "cars"}:
                    raise SubmissionError("A replay frame must hold exactly ball and cars")
                numbers(frame["ball"], 3)
                if not isinstance(frame["cars"], list) or not (1 <= len(frame["cars"]) <= 2):
                    raise SubmissionError("A replay frame must hold one or two cars")
                for car in frame["cars"]:
                    numbers(car, 10)


def submissions_to_score(submissions_folder: str, results_folder: str, replays_folder: str = None) -> list:
    """The submission folders that have no result yet, whose manifest changed since they were scored,
    or that were scored before replays were kept."""
    pending = []
    if not os.path.isdir(submissions_folder):
        return pending
    for slug in sorted(os.listdir(submissions_folder)):
        folder = os.path.join(submissions_folder, slug)
        if not os.path.isdir(folder) or slug.startswith("."):
            continue
        result_path = os.path.join(results_folder, f"{slug}.json")
        if os.path.isfile(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                try:
                    document = json.load(f)
                except (json.JSONDecodeError, OSError):
                    document = {}
            up_to_date = document.get("manifest_sha256") == manifest_digest(folder)
            has_replays = replays_folder is None or "error" in document or os.path.isfile(os.path.join(replays_folder, slug + REPLAYS_SUFFIX))
            if up_to_date and has_replays:
                continue
        pending.append(folder)
    return pending


def validate_result_document(document: dict) -> None:
    """Checks a result document produced by the scoring job before it is published.

    The job that scores a submission has run a stranger's model, so what it hands back is
    treated as untrusted data and must have exactly the expected shape.
    """
    if not isinstance(document, dict):
        raise SubmissionError("A result must be a JSON object")
    allowed = {"scored_at", "benchmark_version", "manifest_sha256", "manifest", "model", "results",
               "overall_score", "official", "error"}
    unknown = set(document) - allowed
    if unknown:
        raise SubmissionError(f"Unexpected fields in a result: {', '.join(sorted(unknown))}")
    for field in ("scored_at", "benchmark_version", "manifest_sha256"):
        if not isinstance(document.get(field), str) or len(document[field]) > 64:
            raise SubmissionError(f"'{field}' is missing or not short text")
    if "manifest" in document:
        Manifest.from_dict(document["manifest"])
    if "error" in document:
        if not isinstance(document["error"], str) or len(document["error"]) > 500:
            raise SubmissionError("'error' must be short text")
        return
    if not isinstance(document.get("results"), list) or not document["results"]:
        raise SubmissionError("'results' is missing")
    for result in document["results"]:
        if not isinstance(result, dict):
            raise SubmissionError("Each result must be an object")
        for field in ("task", "episodes", "successes", "conceded", "timeouts", "success_rate", "success_rate_low",
                      "success_rate_high", "mean_seconds_to_score", "time_limit", "seed", "sampled", "official",
                      "interface_version", "task_set_version", "simulator_version", "benchmark_version"):
            if field not in result:
                raise SubmissionError(f"A task result is missing '{field}'")
        if not isinstance(result["task"], str) or len(result["task"]) > 40:
            raise SubmissionError("A task key is not short text")
        for field in ("episodes", "successes", "conceded", "timeouts"):
            if not isinstance(result[field], int) or result[field] < 0:
                raise SubmissionError(f"'{field}' must be a whole number")
        for field in ("success_rate", "success_rate_low", "success_rate_high"):
            if not isinstance(result[field], (int, float)) or not (0 <= result[field] <= 1):
                raise SubmissionError(f"'{field}' must be between 0 and 1")
        if result["successes"] + result["conceded"] + result["timeouts"] != result["episodes"]:
            raise SubmissionError("The episode counts do not add up")
    if not isinstance(document.get("overall_score"), (int, float)) or not (0 <= document["overall_score"] <= 100):
        raise SubmissionError("'overall_score' must be between 0 and 100")
    if not isinstance(document.get("official"), bool):
        raise SubmissionError("'official' must be true or false")
