"""Command line: boost-arena check | tasks | score"""

import argparse
import json
import sys
import time

from . import __version__
from .policy import InvalidModel, Policy, check_model
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
        opponent = "with an opponent" if task.with_opponent else "no opponent"
        print(f"{task.key:16s} {task.name} ({task.time_limit:.0f} s limit, {opponent})")
        print(f"{'':16s} {task.description}")
    return 0


def _cmd_score(args):
    from .runner import run_task  # Loads the simulator, so only when needed

    try:
        policy = Policy.from_file(args.model)
    except InvalidModel as e:
        print(f"Not accepted: {e}")
        return 1

    keys = args.task or list(TASKS)
    results = []
    for key in keys:
        task = TASKS[key]
        started = time.time()

        def progress(done, total, name=task.name):
            if sys.stderr.isatty() and (done % 20 == 0 or done == total):
                print(f"\r{name}: {done}/{total} episodes", end="", file=sys.stderr, flush=True)

        result = run_task(
            policy, task, episodes=args.episodes, seed=args.seed, sampled=not args.most_likely,
            time_limit=args.time_limit, on_progress=progress,
        )
        if sys.stderr.isatty():
            print(file=sys.stderr)
        results.append(result)

        print(f"{task.name}")
        print(f"  Success rate     {result.success_rate:.1%}  (95% interval {result.success_rate_low:.1%} to {result.success_rate_high:.1%})")
        print(f"  Time to score    {result.mean_seconds_to_score:.1f} s on average, limit {result.time_limit:.0f} s")
        print(f"  Episodes         {result.episodes}: {result.successes} scored, {result.own_goals} own goals, {result.timeouts} ran out of time")
        print(f"  Took             {time.time() - started:.0f} s")
        if not result.official:
            print("  Not an official result: the time limit was changed")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"results": [r.to_dict() for r in results]}, f, indent=2)
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

    score = commands.add_parser("score", help="Score a model on one or more tasks")
    score.add_argument("model")
    score.add_argument("--task", action="append", choices=sorted(TASKS), help="Task to run, repeat for several (default: all)")
    score.add_argument("--episodes", type=int, default=1000)
    score.add_argument("--seed", type=int, default=0)
    score.add_argument("--most-likely", action="store_true",
                       help="Always take the most likely action, instead of drawing actions at random as in training")
    score.add_argument("--time-limit", type=float, default=None, help="Change the time limit (the result is then not official)")
    score.add_argument("--output", help="Also save the results to this JSON file")
    score.set_defaults(run=_cmd_score)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
