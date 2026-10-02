"""Submissions: making one, checking one, opening one, scoring one, and checking what the service produced.

A submission is a folder `submissions/<slug>/` holding one file, `submission.json`. It names
the bot and points at the sealed model file, which the entrant hosts anywhere public. The
scoring service opens the sealed file with the project's private key, scores the model and
writes `results/<slug>.json`. The model itself is never stored by the project.

On the service, opening and scoring are separate steps: only the opening step has the
private key, and only the scoring step runs the stranger's model.
"""

import datetime
import gzip
import hashlib
import importlib.metadata
import ipaddress
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from . import __version__, interface
from .policy import InvalidModel, Policy, check_model
from .sealed import SealedFileError, is_sealed, open_sealed, seal

MANIFEST_NAME = "submission.json"
REPLAYS_SUFFIX = ".replays.json.gz"
MODEL_SUFFIX = ".onnx"
RECORDED_EPISODES = 5  # Per task. Every bot's replays are of the same situations
MAX_REPLAY_BYTES = 4 * 1024 * 1024
MAX_REPLAY_FRAMES = 2000
MAX_SEALED_BYTES = 64 * 1024 * 1024 + 4096
DOWNLOAD_TIMEOUT = 30  # Per read, seconds
DOWNLOAD_DEADLINE = 300  # For the whole file, seconds
MAX_REDIRECTS = 5
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
GITHUB_LOGIN_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")
MAX_NAME = 40
MAX_TEXT = 300
SCORINGS_PER_PERIOD = 3  # Scored submissions one person may have in a period, see RULES.md
SCORING_PERIOD_DAYS = 30


class SubmissionError(Exception):
    """Something about the submission is not right. The message says what."""


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
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
    github: str  # The GitHub login that submits the bot; the sealed file is bound to it
    model_url: str
    sealed_sha256: str
    model_sha256: str
    model_bytes: int
    sealed_with: str  # The public key the model was sealed with
    interface_version: int
    submitted: str  # Date, YYYY-MM-DD
    description: str = ""
    homepage: str = ""
    public_model_url: str = ""  # If the entrant chooses to publish their model themselves

    REQUIRED = frozenset(
        {
            "name",
            "slug",
            "author",
            "github",
            "model_url",
            "sealed_sha256",
            "model_sha256",
            "model_bytes",
            "sealed_with",
            "interface_version",
            "submitted",
        }
    )
    OPTIONAL = frozenset({"description", "homepage", "public_model_url"})

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict):
        if not isinstance(data, dict):
            raise SubmissionError("The manifest must be a JSON object")
        missing = cls.REQUIRED - set(data)
        unknown = set(data) - cls.REQUIRED - cls.OPTIONAL
        if missing:
            raise SubmissionError(f"The manifest is missing: {', '.join(sorted(missing))}")
        if unknown:
            raise SubmissionError(f"The manifest has fields that are not allowed: {', '.join(sorted(unknown))}")
        manifest = cls(**data)
        manifest.validate()
        return manifest

    def validate(self):
        for field in (
            "name",
            "slug",
            "author",
            "github",
            "model_url",
            "sealed_sha256",
            "model_sha256",
            "sealed_with",
            "submitted",
            "description",
            "homepage",
            "public_model_url",
        ):
            if not isinstance(getattr(self, field), str):
                raise SubmissionError(f"'{field}' must be text")
        if not (1 <= len(self.name) <= MAX_NAME) or not self.name.strip() or self.name != self.name.strip():
            raise SubmissionError(f"The bot name must be 1 to {MAX_NAME} characters, without spaces at the ends")
        if not (1 <= len(self.author) <= MAX_NAME) or not self.author.strip():
            raise SubmissionError(f"The author must be 1 to {MAX_NAME} characters")
        if not GITHUB_LOGIN_PATTERN.match(self.github):
            raise SubmissionError("'github' must be the GitHub login of the person submitting")
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
        if type(self.model_bytes) is not int or not (0 < self.model_bytes <= MAX_SEALED_BYTES):
            raise SubmissionError("'model_bytes' must be a positive size within the file limit")
        if self.interface_version != interface.INTERFACE_VERSION:
            raise SubmissionError(
                f"Interface version {self.interface_version} is not the current version {interface.INTERFACE_VERSION}"
            )
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", self.submitted):
            raise SubmissionError("'submitted' must be a date, YYYY-MM-DD")


def load_manifest(folder: str) -> Manifest:
    path = os.path.join(folder, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise SubmissionError(f"No {MANIFEST_NAME} in {folder}")
    others = [name for name in os.listdir(folder) if name != MANIFEST_NAME and not name.startswith(".")]
    if others:
        raise SubmissionError(f"A submission folder may only hold {MANIFEST_NAME}, found: {', '.join(others)}")
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise SubmissionError(f"{MANIFEST_NAME} is not valid JSON: {e}") from None
    manifest = Manifest.from_dict(data)
    if manifest.slug != os.path.basename(os.path.normpath(folder)):
        raise SubmissionError(
            f"The folder is named '{os.path.basename(os.path.normpath(folder))}' but the slug is '{manifest.slug}'"
        )
    return manifest


def manifest_digest(folder: str) -> str:
    with open(os.path.join(folder, MANIFEST_NAME), "rb") as f:
        return sha256(f.read())


def submission_folders(submissions_folder: str) -> list:
    """Every submission folder, by slug, skipping anything that is not named like one."""
    if not os.path.isdir(submissions_folder):
        return []
    return [
        os.path.join(submissions_folder, slug)
        for slug in sorted(os.listdir(submissions_folder))
        if SLUG_PATTERN.match(slug) and os.path.isdir(os.path.join(submissions_folder, slug))
    ]


# ---- Making a submission ----


def make_submission(
    model_path: str,
    public_key: str,
    name: str,
    author: str,
    github: str,
    output_folder: str,
    description: str = "",
    homepage: str = "",
    public_model_url: str = "",
    model_url: str = "",
) -> Manifest:
    """Seals the model. Writes the manifest into `output_folder/<slug>/` and the sealed file
    beside that folder, as `output_folder/<slug>.sealed`, so the folder is exactly what goes
    into the pull request and can be checked with `verify-submission` as it is.

    `model_url` is where the entrant will host the sealed file. It can be filled in later.
    """
    with open(model_path, "rb") as f:
        model = f.read()
    try:
        check_model(model)
        Policy(model)
    except InvalidModel as e:
        raise SubmissionError(f"The model is not acceptable: {e}") from None

    slug = slugify(name)
    try:
        sealed = seal(model, public_key, slug, github.strip())
    except SealedFileError as e:
        raise SubmissionError(str(e)) from None

    manifest = Manifest(
        name=name.strip(),
        slug=slug,
        author=author.strip(),
        github=github.strip(),
        model_url=model_url.strip(),
        sealed_sha256=sha256(sealed),
        model_sha256=sha256(model),
        model_bytes=len(sealed),
        sealed_with=public_key.strip(),
        interface_version=interface.INTERFACE_VERSION,
        submitted=datetime.date.today().isoformat(),
        description=description.strip(),
        homepage=homepage.strip(),
        public_model_url=public_model_url.strip(),
    )
    manifest.validate()  # The same checks the pull request will run, so problems show up now

    folder = os.path.join(output_folder, slug)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(output_folder, f"{slug}.sealed"), "wb") as f:
        f.write(sealed)
    with open(os.path.join(folder, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
        f.write("\n")
    return manifest


# ---- Fetching and checking ----


def _is_public_host(host: str) -> bool:
    """Only public addresses are downloaded from: nothing on the machine or its network."""
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False
    if not infos:
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            return False
    return True


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme != "https" or not _is_public_host(parsed.hostname or ""):
            raise urllib.error.HTTPError(newurl, code, "redirect to an address that is not public https", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url: str, allow_local: bool = False) -> bytes:
    """Downloads the sealed file: public https only, within the size limit and a time limit."""
    if url.startswith("file://"):
        if not allow_local:
            raise SubmissionError("Local files are not accepted, the sealed model must be at an https:// address")
        with open(url[len("file://") :], "rb") as f:
            return f.read(MAX_SEALED_BYTES + 1)

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise SubmissionError("The sealed model must be at an https:// address")
    if not _is_public_host(parsed.hostname):
        raise SubmissionError("The sealed model must be at a public address")

    opener = urllib.request.build_opener(_HttpsOnlyRedirects())
    request = urllib.request.Request(url, headers={"User-Agent": f"boost-arena/{__version__}"})
    deadline = time.monotonic() + DOWNLOAD_DEADLINE
    chunks, size = [], 0
    try:
        with opener.open(request, timeout=DOWNLOAD_TIMEOUT) as response:
            while size <= MAX_SEALED_BYTES:
                if time.monotonic() > deadline:
                    raise SubmissionError("Downloading the sealed model took too long")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
    except SubmissionError:
        raise
    except urllib.error.HTTPError as e:
        raise SubmissionError(f"The sealed model could not be downloaded (HTTP {e.code})") from None
    except Exception as e:
        raise SubmissionError(f"The sealed model could not be downloaded ({type(e).__name__})") from None
    if size > MAX_SEALED_BYTES:
        raise SubmissionError("The sealed model is larger than the limit")
    return b"".join(chunks)


def verify_submission(folder: str, public_key: str = None, allow_local: bool = False, submissions_folder: str = None) -> dict:
    """Everything that can be checked without the private key. Returns a short description, with the sealed bytes."""
    manifest = load_manifest(folder)
    if public_key is not None and manifest.sealed_with.strip() != public_key.strip():
        raise SubmissionError("The model was sealed with a different public key. Seal it again with the current one")
    if submissions_folder is not None:
        check_not_a_copy(manifest, submissions_folder)

    sealed = fetch(manifest.model_url, allow_local=allow_local)
    if len(sealed) != manifest.model_bytes:
        raise SubmissionError(f"The sealed file is {len(sealed)} bytes but the manifest says {manifest.model_bytes}")
    if sha256(sealed) != manifest.sealed_sha256:
        raise SubmissionError("The sealed file does not match the SHA-256 in the manifest")
    if not is_sealed(sealed):
        raise SubmissionError("The file at model_url is not a sealed model")
    return {
        "slug": manifest.slug,
        "name": manifest.name,
        "author": manifest.author,
        "github": manifest.github,
        "sealed_bytes": len(sealed),
        "sealed": sealed,
    }


def check_not_a_copy(manifest: Manifest, submissions_folder: str) -> None:
    """A manifest may not point at another submission's model: that would be claiming someone else's score."""
    for other_folder in submission_folders(submissions_folder):
        if os.path.basename(other_folder) == manifest.slug:
            continue
        try:
            with open(os.path.join(other_folder, MANIFEST_NAME), encoding="utf-8") as f:
                other = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(other, dict):
            continue
        if manifest.model_sha256 == other.get("model_sha256") or manifest.sealed_sha256 == other.get("sealed_sha256"):
            raise SubmissionError(f"This model is already submitted as '{os.path.basename(other_folder)}'")
        if manifest.name.strip().lower() == str(other.get("name", "")).strip().lower():
            raise SubmissionError(f"The name '{manifest.name}' is already used by '{os.path.basename(other_folder)}'")


def check_pull_request(
    folders: list, github_login: str, submissions_folder: str, results_folder: str, exempt: tuple = ()
) -> None:
    """What the pull request check knows that the manifest alone does not: who opened it, and what they already have.

    `exempt` logins are not held to the scoring limit: the project's own baseline bots (see RULES.md).
    """
    if not GITHUB_LOGIN_PATTERN.match(github_login or ""):
        raise SubmissionError("The pull request author is not a valid GitHub login")
    changed = set()
    for folder in folders:
        manifest = load_manifest(folder)
        if manifest.github.lower() != github_login.lower():
            raise SubmissionError(
                f"{manifest.slug}: the manifest names '{manifest.github}' but the pull request is from '{github_login}'"
            )
        changed.add(manifest.slug)
    if github_login.lower() in {login.lower() for login in exempt}:
        return

    # The scorings this person already had in the period, plus the ones this pull request asks for
    since = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=SCORING_PERIOD_DAYS)
    recent = 0
    if os.path.isdir(results_folder):
        for name in os.listdir(results_folder):
            if not name.endswith(".json") or name[:-5] in changed:
                continue
            try:
                with open(os.path.join(results_folder, name), encoding="utf-8") as f:
                    document = json.load(f)
                login = document["manifest"]["github"]
                scored_at = datetime.datetime.strptime(document["scored_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.UTC)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if login.lower() == github_login.lower() and scored_at >= since:
                recent += 1
    if recent + len(changed) > SCORINGS_PER_PERIOD:
        raise SubmissionError(
            f"'{github_login}' would have {recent + len(changed)} scored submissions in {SCORING_PERIOD_DAYS} days, "
            f"the limit is {SCORINGS_PER_PERIOD}"
        )


# ---- Opening (needs the private key) ----


def open_submission(folder: str, private_key: str, allow_local: bool = False) -> tuple:
    """Downloads, opens and checks one submission's model. Returns (manifest, model bytes, model description).

    Whether the model is a copy of another submission's is a pull request check, not one made
    here: a sealed file only opens for the submission it was sealed for anyway."""
    manifest = load_manifest(folder)
    sealed = verify_submission(folder, allow_local=allow_local)["sealed"]
    try:
        model = open_sealed(sealed, private_key, manifest.slug, manifest.github)
    except SealedFileError as e:
        raise SubmissionError(str(e)) from None
    if sha256(model) != manifest.model_sha256:
        raise SubmissionError("The opened model does not match the SHA-256 in the manifest")
    try:
        info = check_model(model)
        Policy(model)
    except InvalidModel as e:
        raise SubmissionError(f"The model is not acceptable: {e}") from None
    return manifest, model, info


def library_versions() -> dict:
    """What the scoring ran on, so a result can be traced when a dependency changes."""
    import platform

    versions = {"python": platform.python_version(), "platform": platform.system().lower()}
    for name in ("rocketsim", "onnxruntime", "numpy", "onnx"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def new_document(folder: str) -> dict:
    """The start of a result document for a submission folder."""
    return {
        "scored_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark_version": __version__,
        "manifest_sha256": manifest_digest(folder),
    }


def open_submissions(folders: list, private_key: str, models_folder: str, output_folder: str, allow_local: bool = False) -> list:
    """Opens every submission into `models_folder/<slug>.onnx`. A submission that cannot be
    opened gets its error written as a result document in `output_folder` instead.

    Returns the slugs that were opened.
    """
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    opened = []
    for folder in folders:
        slug = os.path.basename(os.path.normpath(folder))
        document = new_document(folder)
        try:
            manifest, model, info = open_submission(folder, private_key, allow_local=allow_local)
            document["manifest"] = manifest.to_dict()
            document["model"] = info
            with open(os.path.join(models_folder, slug + MODEL_SUFFIX), "wb") as f:
                f.write(model)
            with open(os.path.join(models_folder, slug + ".json"), "w", encoding="utf-8") as f:
                json.dump(document, f, indent=2)
            opened.append(slug)
        except SubmissionError as e:
            document["error"] = str(e)
            with open(os.path.join(output_folder, f"{slug}.json"), "w", encoding="utf-8") as f:
                json.dump(document, f, indent=2)
                f.write("\n")
    return opened


# ---- Scoring (no key needed) ----


def score_model(document: dict, model: bytes, episodes: int = None, on_progress=None) -> tuple:
    """Scores an opened model. `document` is the result document started when it was opened.

    Returns the finished document, which records any failure, and the replays document (or None).
    Nothing the model does can raise: the model is a stranger's, and a failure is a result.
    """
    from .runner import FRAMES_PER_SECOND, OFFICIAL_EPISODES, SEASON, overall_score, run_task
    from .tasks import TASK_SET_VERSION, TASKS

    slug = document["manifest"]["slug"]
    try:
        policy = Policy(model)
        results = []
        replays = {
            "slug": slug,
            "fps": FRAMES_PER_SECOND,
            "interface_version": interface.INTERFACE_VERSION,
            "task_set_version": TASK_SET_VERSION,
            "season": SEASON,
            "tasks": {},
        }
        for key, task in TASKS.items():
            result = run_task(
                policy, task, episodes=episodes or OFFICIAL_EPISODES, on_progress=on_progress, record_first=RECORDED_EPISODES
            )
            results.append(result)
            replays["tasks"][key] = result.replays
        document["results"] = [r.to_dict() for r in results]
        document["overall_score"] = overall_score(results)
        document["official"] = all(r.official for r in results)
        document["season"] = SEASON
        document["libraries"] = library_versions()
    except InvalidModel as e:
        document["error"] = f"The model is not acceptable: {e}"
        return document, None
    except Exception as e:  # The model is a stranger's; whatever it does, scoring must go on
        document["error"] = f"The model failed while being scored ({type(e).__name__})"
        return document, None
    return document, replays


def score_submission(folder: str, private_key: str, allow_local: bool = False, episodes: int = None, on_progress=None) -> tuple:
    """Opens, checks and scores one submission in one go, as a local tool would."""
    document = new_document(folder)
    try:
        manifest, model, info = open_submission(folder, private_key, allow_local=allow_local)
    except SubmissionError as e:
        document["error"] = str(e)
        return document, None
    document["manifest"] = manifest.to_dict()
    document["model"] = info
    return score_model(document, model, episodes=episodes, on_progress=on_progress)


def write_replays(replays: dict, path: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(replays, f, separators=(",", ":"))


def read_replays(path: str) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def submissions_to_score(submissions_folder: str, results_folder: str, replays_folder: str = None) -> list:
    """The submission folders that have no result yet, whose manifest changed since they were scored,
    whose result is from another season, or that were scored before replays were kept."""
    from .runner import SEASON

    pending = []
    for folder in submission_folders(submissions_folder):
        slug = os.path.basename(folder)
        result_path = os.path.join(results_folder, f"{slug}.json")
        if os.path.isfile(result_path):
            with open(result_path, encoding="utf-8") as f:
                try:
                    document = json.load(f)
                except (json.JSONDecodeError, OSError):
                    document = {}
            up_to_date = document.get("manifest_sha256") == manifest_digest(folder)
            current_season = "error" in document or document.get("season") == SEASON
            has_replays = (
                replays_folder is None
                or "error" in document
                or os.path.isfile(os.path.join(replays_folder, slug + REPLAYS_SUFFIX))
            )
            if up_to_date and current_season and has_replays:
                continue
        pending.append(folder)
    return pending


# ---- Checking what the scoring job produced ----


def _whole_number(value) -> bool:
    return type(value) is int


def _number(value) -> bool:
    return type(value) in (int, float)


def validate_replays_document(document: dict) -> None:
    """Checks a replays document from the scoring job: numbers of the right shape, nothing else."""
    from .tasks import TASKS

    if not isinstance(document, dict) or set(document) != {
        "slug",
        "fps",
        "interface_version",
        "task_set_version",
        "season",
        "tasks",
    }:
        raise SubmissionError("A replays file must hold exactly slug, fps, interface_version, task_set_version, season and tasks")
    if not isinstance(document["slug"], str) or not SLUG_PATTERN.match(document["slug"]):
        raise SubmissionError("The replays slug is not valid")
    for field in ("fps", "interface_version", "task_set_version", "season"):
        if not _whole_number(document[field]):
            raise SubmissionError(f"'{field}' must be a whole number")
    if not isinstance(document["tasks"], dict) or set(document["tasks"]) - set(TASKS):
        raise SubmissionError("The replays refer to unknown tasks")

    def numbers(values, count):
        if not isinstance(values, list) or len(values) != count:
            raise SubmissionError("A replay frame has the wrong shape")
        for value in values:
            if not _number(value) or abs(value) > 1e5:
                raise SubmissionError("A replay frame holds something that is not a sensible number")

    for replays in document["tasks"].values():
        if not isinstance(replays, list) or len(replays) > RECORDED_EPISODES:
            raise SubmissionError("Too many replays for a task")
        for replay in replays:
            if not isinstance(replay, dict) or set(replay) != {"episode", "frames", "outcome", "seconds"}:
                raise SubmissionError("A replay must hold exactly episode, frames, outcome and seconds")
            if not _whole_number(replay["episode"]) or replay["outcome"] not in ("success", "conceded", "timeout"):
                raise SubmissionError("A replay has a bad episode number or outcome")
            if not _number(replay["seconds"]) or not (0 <= replay["seconds"] <= 120):
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


def validate_result_document(document: dict, official_only: bool = False) -> None:
    """Checks a result document produced by the scoring job before it is published.

    The job that scores a submission has run a stranger's model, so what it hands back is
    treated as untrusted data: it must have exactly the expected shape, and its numbers must
    agree with each other. With `official_only`, a result that is not official is refused.
    """
    from .runner import OFFICIAL_EPISODES, OFFICIAL_SEED, SEASON
    from .sim import SIMULATOR_VERSION
    from .tasks import TASK_SET_VERSION, TASKS

    if not isinstance(document, dict):
        raise SubmissionError("A result must be a JSON object")
    allowed = {
        "scored_at",
        "benchmark_version",
        "manifest_sha256",
        "manifest",
        "model",
        "results",
        "overall_score",
        "official",
        "season",
        "libraries",
        "error",
    }
    unknown = set(document) - allowed
    if unknown:
        raise SubmissionError(f"Unexpected fields in a result: {', '.join(sorted(unknown))}")
    for field in ("scored_at", "benchmark_version", "manifest_sha256"):
        if not isinstance(document.get(field), str) or len(document[field]) > 64:
            raise SubmissionError(f"'{field}' is missing or not short text")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", document["scored_at"]):
        raise SubmissionError("'scored_at' must be a UTC timestamp")
    if not re.fullmatch(r"[0-9a-f]{64}", document["manifest_sha256"]):
        raise SubmissionError("'manifest_sha256' must be a SHA-256")
    if "manifest" in document:
        Manifest.from_dict(document["manifest"])
    if "model" in document:
        model = document["model"]
        if not isinstance(model, dict) or set(model) != {"parameters", "operations", "file_bytes", "input_name", "output_name"}:
            raise SubmissionError("'model' has the wrong fields")
        for field in ("parameters", "operations", "file_bytes"):
            if not _whole_number(model[field]) or model[field] < 0:
                raise SubmissionError(f"model '{field}' must be a whole number")
        for field in ("input_name", "output_name"):
            if not isinstance(model[field], str) or len(model[field]) > 200:
                raise SubmissionError(f"model '{field}' must be short text")
    if "libraries" in document:
        libraries = document["libraries"]
        if (
            not isinstance(libraries, dict)
            or len(libraries) > 10
            or any(not isinstance(k, str) or not isinstance(v, str) or len(k) > 40 or len(v) > 40 for k, v in libraries.items())
        ):
            raise SubmissionError("'libraries' must map short names to short versions")
    if "error" in document:
        if not isinstance(document["error"], str) or not (0 < len(document["error"]) <= 500):
            raise SubmissionError("'error' must be short text")
        if "results" in document or "overall_score" in document:
            raise SubmissionError("A result cannot hold both an error and scores")
        return

    for field in ("manifest", "model", "results", "overall_score", "official", "season"):
        if field not in document:
            raise SubmissionError(f"A scored result must hold '{field}'")
    results = document["results"]
    if not isinstance(results, list) or [r.get("task") if isinstance(r, dict) else None for r in results] != list(TASKS):
        raise SubmissionError("'results' must hold every task once, in order")
    rates = []
    for result in results:
        expected = {
            "task",
            "episodes",
            "successes",
            "conceded",
            "timeouts",
            "success_rate",
            "success_rate_low",
            "success_rate_high",
            "mean_seconds_to_score",
            "time_limit",
            "seed",
            "sampled",
            "official",
            "season",
            "interface_version",
            "task_set_version",
            "simulator_version",
            "benchmark_version",
        }
        if set(result) != expected:
            raise SubmissionError(f"A task result has the wrong fields: {sorted(set(result) ^ expected)}")
        for field in ("episodes", "successes", "conceded", "timeouts", "seed", "season", "interface_version", "task_set_version"):
            if not _whole_number(result[field]) or result[field] < 0:
                raise SubmissionError(f"'{field}' must be a whole number")
        for field in ("success_rate", "success_rate_low", "success_rate_high"):
            if not _number(result[field]) or not (0 <= result[field] <= 1):
                raise SubmissionError(f"'{field}' must be between 0 and 1")
        for field in ("mean_seconds_to_score", "time_limit"):
            if not _number(result[field]) or not (0 <= result[field] <= 3600):
                raise SubmissionError(f"'{field}' must be a sensible number of seconds")
        for field in ("sampled", "official"):
            if type(result[field]) is not bool:
                raise SubmissionError(f"'{field}' must be true or false")
        for field in ("simulator_version", "benchmark_version"):
            if not isinstance(result[field], str) or len(result[field]) > 64:
                raise SubmissionError(f"'{field}' must be short text")
        if result["episodes"] == 0 or result["successes"] + result["conceded"] + result["timeouts"] != result["episodes"]:
            raise SubmissionError("The episode counts do not add up")
        if abs(result["success_rate"] - result["successes"] / result["episodes"]) > 1e-9:
            raise SubmissionError("The success rate does not match the counts")
        if not (result["success_rate_low"] <= result["success_rate"] <= result["success_rate_high"]):
            raise SubmissionError("The confidence interval does not contain the success rate")
        if result["time_limit"] != TASKS[result["task"]].time_limit and result["official"]:
            raise SubmissionError("An official result must use the task's time limit")
        if (result["interface_version"], result["task_set_version"], result["simulator_version"]) != (
            interface.INTERFACE_VERSION,
            TASK_SET_VERSION,
            SIMULATOR_VERSION,
        ):
            raise SubmissionError("A task result is from another version of the benchmark")
        if result["official"] and (
            result["episodes"] != OFFICIAL_EPISODES
            or result["seed"] != OFFICIAL_SEED
            or not result["sampled"]
            or result["season"] != SEASON
        ):
            raise SubmissionError("A result marked official was not made with the official settings")
        rates.append(result["success_rate"])
    if not _number(document["overall_score"]) or abs(document["overall_score"] - 100 * sum(rates) / len(rates)) > 1e-6:
        raise SubmissionError("'overall_score' is not the average of the task success rates")
    if type(document["official"]) is not bool or document["official"] != all(r["official"] for r in results):
        raise SubmissionError("'official' does not match the task results")
    if not _whole_number(document["season"]) or document["season"] != SEASON:
        raise SubmissionError("The result is from another season")
    if official_only and not document["official"]:
        raise SubmissionError("Only official results are published")


def validate_published_result(path: str, submissions_folder: str, official_only: bool = True) -> dict:
    """A result file about to be published must belong to the submission it is named after,
    and describe the manifest as it is in the repository."""
    with open(path, encoding="utf-8") as f:
        document = json.load(f)
    validate_result_document(document, official_only=official_only)
    slug = os.path.basename(path)[: -len(".json")]
    if not SLUG_PATTERN.match(slug):
        raise SubmissionError("A result file must be named after a slug")
    folder = os.path.join(submissions_folder, slug)
    if not os.path.isdir(folder):
        raise SubmissionError(f"There is no submission '{slug}'")
    if document["manifest_sha256"] != manifest_digest(folder):
        raise SubmissionError(f"The result for '{slug}' is not for the manifest in the repository")
    if "manifest" in document:
        with open(os.path.join(folder, MANIFEST_NAME), encoding="utf-8") as f:
            if document["manifest"] != json.load(f):
                raise SubmissionError(f"The result for '{slug}' holds a manifest that differs from the repository's")
    return document


def validate_published_replays(path: str, submissions_folder: str) -> dict:
    if os.path.getsize(path) > MAX_REPLAY_BYTES:
        raise SubmissionError("The replays file is too large")
    document = read_replays(path)
    validate_replays_document(document)
    name = os.path.basename(path)
    if not name.endswith(REPLAYS_SUFFIX) or name[: -len(REPLAYS_SUFFIX)] != document["slug"]:
        raise SubmissionError("A replays file must be named after its slug")
    if not os.path.isdir(os.path.join(submissions_folder, document["slug"])):
        raise SubmissionError(f"There is no submission '{document['slug']}'")
    return document
