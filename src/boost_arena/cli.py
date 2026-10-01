"""Command line for entrants and for the scoring service."""

import argparse
import json
import sys
import time

import os

from . import __version__
from .policy import InvalidModel, Policy, check_model, uniform_model
from .tasks import TASKS

PUBLIC_KEY_FILE = "PUBLIC_KEY"
PRIVATE_KEY_VARIABLE = "BOOST_ARENA_PRIVATE_KEY"


def _read_public_key(path):
    if not os.path.isfile(path):
        raise SystemExit(f"No public key file at {path}. Pass --public-key, or run this from a clone of the repository")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _progress(name):
    def show(done, total):
        if sys.stderr.isatty() and (done % 50 == 0 or done == total):
            print(f"\r{name}: {done}/{total} episodes", end="", file=sys.stderr, flush=True)
    return show


def _cmd_check(args):
    with open(args.model, "rb") as f:
        data = f.read()
    try:
        info = check_model(data)
        Policy(data)
    except InvalidModel as e:
        print(f"Not accepted: {e}")
        return 1
    print(f"Accepted: {info['parameters']:,} parameters, {info['operations']} operations, {info['file_bytes'] / 1e6:.1f} MB")
    return 0


def _cmd_tasks(args):
    for task in TASKS.values():
        opponent = "against the reference keeper" if task.with_opponent else "no opponent"
        print(f"{task.key:14s} {task.name} ({task.time_limit:.0f} s, {opponent})")
        print(f"{'':14s} {task.description}")
    return 0


def _cmd_make_random_bot(args):
    with open(args.output, "wb") as f:
        f.write(uniform_model())
    print(f"Wrote {args.output}: a bot that picks among its allowed actions with equal chance")
    return 0


def _cmd_score(args):
    from .runner import OFFICIAL_EPISODES, overall_score, run_task  # Loads the simulator, so only when needed

    try:
        policy = Policy.from_file(args.model)
    except InvalidModel as e:
        print(f"Not accepted: {e}")
        return 1

    episodes = args.episodes or OFFICIAL_EPISODES
    keys = args.task or list(TASKS)
    results = []
    started = time.time()
    for key in keys:
        task = TASKS[key]

        def progress(done, total, name=task.name):
            if sys.stderr.isatty() and (done % 20 == 0 or done == total):
                print(f"\r{name}: {done}/{total} episodes", end="", file=sys.stderr, flush=True)

        results.append(run_task(
            policy, task, episodes=episodes, seed=args.seed, sampled=not args.most_likely,
            time_limit=args.time_limit, on_progress=progress,
        ))
        if sys.stderr.isatty():
            print("\r" + " " * 60 + "\r", end="", file=sys.stderr)

    print(f"{'Task':20s} {'Success':>8s}   {'95% interval':14s} {'Time to score':>14s}   Own goal  Out of time")
    for result in results:
        task = TASKS[result.task]
        seconds = "" if task.survive or not result.mean_seconds_to_score else f"{result.mean_seconds_to_score:.1f} s"
        print(
            f"{task.name:20s} {result.success_rate:>8.1%}   "
            f"{f'{result.success_rate_low:.1%} to {result.success_rate_high:.1%}':14s} {seconds:>14s}   "
            f"{result.conceded:>8d}  {result.timeouts:>11d}"
        )

    complete = set(keys) == set(TASKS)
    official = complete and all(result.official for result in results)
    if complete:
        print(f"\nOverall score: {overall_score(results):.1f} out of 100")
    print(f"{episodes} episodes per task, {time.time() - started:.0f} s")
    if not official:
        print("Not an official result: official scoring runs every task with the standard settings")

    if args.output:
        document = {"official_settings": official, "results": [r.to_dict() for r in results]}
        if complete:
            document["overall_score"] = overall_score(results)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2)
        print(f"Saved to {args.output}")
    return 0


def _cmd_submit(args):
    from .submissions import SubmissionError, make_submission

    try:
        manifest = make_submission(
            args.model, _read_public_key(args.public_key), args.name, args.author, args.output,
            description=args.description or "", homepage=args.homepage or "",
            public_model_url=args.public_model_url or "", model_url=args.model_url or "",
        )
    except SubmissionError as e:
        print(f"Not accepted: {e}")
        return 1

    folder = os.path.join(args.output, manifest.slug)
    print(f"Wrote {folder}/")
    print(f"  {manifest.slug}.sealed    the sealed model, {manifest.model_bytes / 1e6:.1f} MB. Put this somewhere public over https")
    print(f"  submission.json   the manifest. Copy it to submissions/{manifest.slug}/ in a pull request")
    if not manifest.model_url:
        print("Then fill in model_url in submission.json with the address of the sealed file.")
    return 0


def _cmd_verify_submission(args):
    from .submissions import SubmissionError, verify_submission

    public_key = _read_public_key(args.public_key) if os.path.isfile(args.public_key) else None
    failed = 0
    for folder in args.folder:
        try:
            info = verify_submission(folder, public_key=public_key, allow_local=args.allow_local)
            print(f"OK   {folder}: {info['name']} by {info['author']}, sealed model {info['sealed_bytes'] / 1e6:.1f} MB")
        except SubmissionError as e:
            print(f"FAIL {folder}: {e}")
            failed += 1
    return 1 if failed else 0


def _cmd_process_submissions(args):
    from .submissions import REPLAYS_SUFFIX, score_submission, submissions_to_score, write_replays

    private_key = os.environ.get(PRIVATE_KEY_VARIABLE, "").strip()
    if not private_key:
        raise SystemExit(f"The private key must be in the {PRIVATE_KEY_VARIABLE} environment variable")

    pending = submissions_to_score(args.submissions, args.results, args.replays)
    if args.only:
        pending = [folder for folder in pending if os.path.basename(os.path.normpath(folder)) in args.only]
    if not pending:
        print("Nothing new to score")
        return 0

    os.makedirs(args.output, exist_ok=True)
    for folder in pending:
        slug = os.path.basename(os.path.normpath(folder))
        started = time.time()
        document, replays = score_submission(folder, private_key, allow_local=args.allow_local, episodes=args.episodes,
                                             on_progress=_progress(slug))
        if sys.stderr.isatty():
            print("\r" + " " * 60 + "\r", end="", file=sys.stderr)
        with open(os.path.join(args.output, f"{slug}.json"), "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2)
            f.write("\n")
        if replays is not None:
            write_replays(replays, os.path.join(args.output, slug + REPLAYS_SUFFIX))
        if "error" in document:
            print(f"{slug}: not scored, {document['error']}")
        else:
            print(f"{slug}: overall {document['overall_score']:.1f} in {time.time() - started:.0f} s")
    return 0


def _cmd_duel(args):
    from .duels import DUEL_KINDS, run_pair

    try:
        a, b = Policy.from_file(args.a), Policy.from_file(args.b)
    except InvalidModel as e:
        print(f"Not accepted: {e}")
        return 1
    started = time.time()
    result = run_pair(a, b, episodes=args.episodes)
    name_a, name_b = os.path.basename(args.a), os.path.basename(args.b)
    for kind, tally in result["kinds"].items():
        print(f"{DUEL_KINDS[kind]:14s} {name_a} {tally['a_points']} - {tally['b_points']} {name_b}"
              + (f", {tally['draws']} draws" if tally["draws"] else ""))
    print(f"{'Total':14s} {name_a} {result['a_points']} - {result['b_points']} {name_b}")
    print(f"{args.episodes} episodes per direction, {time.time() - started:.0f} s")
    return 0


def _cmd_process_duels(args):
    from .duel_service import open_bot, pairs_to_play, play_pair, write_duel_replays
    from .submissions import SubmissionError

    private_key = os.environ.get(PRIVATE_KEY_VARIABLE, "").strip()
    if not private_key:
        raise SystemExit(f"The private key must be in the {PRIVATE_KEY_VARIABLE} environment variable")

    pending = pairs_to_play(args.submissions, args.results, args.duels)
    if not pending:
        print("No duels to play")
        return 0
    if len(pending) > args.max_pairs:
        print(f"{len(pending)} pairs to play, doing {args.max_pairs} this time")
        pending = pending[: args.max_pairs]

    policies = {}
    os.makedirs(os.path.join(args.output, "duels"), exist_ok=True)
    for a, b in pending:
        try:
            for slug in (a, b):
                if slug not in policies:
                    policies[slug] = open_bot(args.submissions, slug, private_key, allow_local=args.allow_local)
        except SubmissionError as e:
            print(f"{a} vs {b}: skipped, {e}")
            continue
        started = time.time()
        document, replays = play_pair(args.submissions, a, b, policies, episodes=args.episodes)
        name = document["a"] + "__" + document["b"]
        with open(os.path.join(args.output, "duels", name + ".json"), "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2)
            f.write("\n")
        write_duel_replays(replays, os.path.join(args.output, "duels", name + ".replays.json.gz"))
        print(f"{a} {document['a_points']} - {document['b_points']} {b} in {time.time() - started:.0f} s")
    return 0


def _cmd_validate_results(args):
    from .duel_service import read_duel_replays, validate_duel_document, validate_duel_replays_document
    from .submissions import MAX_REPLAY_BYTES, REPLAYS_SUFFIX, SubmissionError, read_replays, validate_replays_document, validate_result_document

    failed = 0
    for path in args.file:
        try:
            is_duel = os.path.basename(os.path.dirname(os.path.abspath(path))) == "duels"
            if path.endswith(REPLAYS_SUFFIX):
                if os.path.getsize(path) > MAX_REPLAY_BYTES:
                    raise SubmissionError("The replays file is too large")
                if is_duel:
                    validate_duel_replays_document(read_duel_replays(path))
                else:
                    validate_replays_document(read_replays(path))
            elif is_duel:
                with open(path, "r", encoding="utf-8") as f:
                    validate_duel_document(json.load(f))
            else:
                with open(path, "r", encoding="utf-8") as f:
                    validate_result_document(json.load(f))
            print(f"OK   {path}")
        except (SubmissionError, json.JSONDecodeError, OSError, EOFError, ValueError) as e:
            print(f"FAIL {path}: {e}")
            failed += 1
    return 1 if failed else 0


def _cmd_build_site(args):
    from .site import build_site

    path = build_site(args.results, args.output, args.replays, args.duels)
    print(f"Wrote {path}")
    return 0


def _cmd_keygen(args):
    from .sealed import generate_key_pair

    private, public = generate_key_pair()
    print("Public key (commit it as PUBLIC_KEY):")
    print(public)
    print(f"\nPrivate key (store it as the {PRIVATE_KEY_VARIABLE} secret and keep a copy somewhere safe, never in the repository):")
    print(private)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="boost-arena", description="Score a bot on the Boost Arena benchmark")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Check that a model file is an acceptable submission")
    check.add_argument("model")
    check.set_defaults(run=_cmd_check)

    tasks = commands.add_parser("tasks", help="List the tasks")
    tasks.set_defaults(run=_cmd_tasks)

    random_bot = commands.add_parser("make-random-bot", help="Write a bot that acts at random, as a baseline")
    random_bot.add_argument("output")
    random_bot.set_defaults(run=_cmd_make_random_bot)

    score = commands.add_parser("score", help="Score a model on one or more tasks")
    score.add_argument("model")
    score.add_argument("--task", action="append", choices=list(TASKS), help="Task to run, repeat for several (default: all)")
    score.add_argument("--episodes", type=int, default=None, help="Episodes per task (default: 1000)")
    score.add_argument("--seed", type=int, default=0)
    score.add_argument("--most-likely", action="store_true",
                       help="Always take the most likely action, instead of drawing actions at random as in training")
    score.add_argument("--time-limit", type=float, default=None, help="Change the time limit of every task")
    score.add_argument("--output", help="Also save the results to this JSON file")
    score.set_defaults(run=_cmd_score)

    submit = commands.add_parser("submit", help="Seal a model and write the manifest for a submission")
    submit.add_argument("model", help="The ONNX file")
    submit.add_argument("--name", required=True, help="The bot's name")
    submit.add_argument("--author", required=True, help="Your name or handle")
    submit.add_argument("--description", help="One line about the bot, optional")
    submit.add_argument("--homepage", help="An https:// link about the bot, optional")
    submit.add_argument("--public-model-url", help="Where you publish the model yourself, if you do, optional")
    submit.add_argument("--model-url", help="Where the sealed file will be hosted, if you already know")
    submit.add_argument("--public-key", default=PUBLIC_KEY_FILE, help=f"The project's public key file (default: {PUBLIC_KEY_FILE})")
    submit.add_argument("--output", default="my_submission", help="Folder to write into (default: my_submission)")
    submit.set_defaults(run=_cmd_submit)

    verify = commands.add_parser("verify-submission", help="Check submission folders, without the private key")
    verify.add_argument("folder", nargs="+")
    verify.add_argument("--public-key", default=PUBLIC_KEY_FILE)
    verify.add_argument("--allow-local", action="store_true", help=argparse.SUPPRESS)
    verify.set_defaults(run=_cmd_verify_submission)

    process = commands.add_parser("process-submissions", help="Score the submissions that have no result yet (needs the private key)")
    process.add_argument("--submissions", default="submissions")
    process.add_argument("--results", default="results")
    process.add_argument("--replays", default="replays")
    process.add_argument("--output", default="new_results", help="Where to write the new result and replay files")
    process.add_argument("--only", nargs="*", help="Only these slugs")
    process.add_argument("--episodes", type=int, default=None, help=argparse.SUPPRESS)
    process.add_argument("--allow-local", action="store_true", help=argparse.SUPPRESS)
    process.set_defaults(run=_cmd_process_submissions)

    duel = commands.add_parser("duel", help="Play two models against each other")
    duel.add_argument("a")
    duel.add_argument("b")
    duel.add_argument("--episodes", type=int, default=100, help="Per direction of each duel kind (official: 200)")
    duel.set_defaults(run=_cmd_duel)

    process_duels = commands.add_parser("process-duels", help="Play the duels that have not been played (needs the private key)")
    process_duels.add_argument("--submissions", default="submissions")
    process_duels.add_argument("--results", default="results")
    process_duels.add_argument("--duels", default="duels")
    process_duels.add_argument("--output", default="new_results")
    process_duels.add_argument("--max-pairs", type=int, default=15, help="At most this many pairs per run")
    process_duels.add_argument("--episodes", type=int, default=None, help=argparse.SUPPRESS)
    process_duels.add_argument("--allow-local", action="store_true", help=argparse.SUPPRESS)
    process_duels.set_defaults(run=_cmd_process_duels)

    validate = commands.add_parser("validate-results", help="Check result and replay files before they are published")
    validate.add_argument("file", nargs="+")
    validate.set_defaults(run=_cmd_validate_results)

    site = commands.add_parser("build-site", help="Build the leaderboard website")
    site.add_argument("--results", default="results")
    site.add_argument("--replays", default="replays")
    site.add_argument("--duels", default="duels")
    site.add_argument("--output", default="site")
    site.set_defaults(run=_cmd_build_site)

    keygen = commands.add_parser("keygen", help="Make a new key pair for the scoring service")
    keygen.set_defaults(run=_cmd_keygen)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
