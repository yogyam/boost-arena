"""Command line: boost-arena check | tasks | score | make-random-bot"""

import argparse
import json
import sys
import time

from . import __version__
from .policy import InvalidModel, Policy, check_model, uniform_model
from .tasks import TASKS


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

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
