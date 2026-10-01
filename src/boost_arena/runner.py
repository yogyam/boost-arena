"""Plays a task many times and scores the result."""

import math
import zlib
from dataclasses import asdict, dataclass, field

import numpy as np

from . import __version__, interface
from .policy import Policy
from .sim import SIMULATOR_VERSION, Game
from .tasks import CONCEDED, SUCCESS, TASK_SET_VERSION, Task

OFFICIAL_EPISODES = 1000

# The situations a bot is scored on come from a seed that changes every season, so a bot
# cannot be tuned to the exact situations for long. RULES.md says when a season changes.
SEASON = 1
OFFICIAL_SEEDS = {1: 0}
OFFICIAL_SEED = OFFICIAL_SEEDS[SEASON]


@dataclass
class TaskResult:
    task: str
    episodes: int
    successes: int
    conceded: int  # Episodes that ended with the ball in the bot's own goal
    timeouts: int  # Episodes that ran out of time without success
    success_rate: float
    success_rate_low: float  # 95% confidence interval
    success_rate_high: float
    mean_seconds_to_score: float  # Over episodes won by scoring, 0 if there were none
    time_limit: float
    seed: int
    sampled: bool
    official: bool
    season: int = SEASON
    interface_version: int = interface.INTERFACE_VERSION
    task_set_version: int = TASK_SET_VERSION
    simulator_version: str = SIMULATOR_VERSION
    benchmark_version: str = __version__
    replays: list = field(default_factory=list, repr=False, compare=False)  # Not part of the result document

    def to_dict(self):
        document = asdict(self)
        del document["replays"]
        return document


FRAMES_PER_SECOND = interface.TICK_RATE // interface.TICK_SKIP


def _frame(ball, cars):
    """One replay frame: the ball's position and each car's position, orientation and state, rounded."""
    return {
        "ball": [round(float(v), 1) for v in ball.pos],
        "cars": [
            [round(float(v), 1) for v in car.pos]
            + [round(float(v), 3) for v in car.forward]
            + [round(float(v), 3) for v in car.up]
            + [int(car.is_on_ground) + 2 * int(car.is_demoed)]
            for car in cars
        ],
    }


def overall_score(results) -> float:
    """One number for a bot: its average success rate over the tasks, out of 100."""
    return 100 * sum(r.success_rate for r in results) / len(results) if results else 0.0


def _wilson_interval(successes, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _episode_rngs(task: Task, seed: int, episode: int):
    """Separate random generators for setting up an episode and for choosing actions in it.

    They depend only on the task, the seed and the episode number, so an episode plays out
    the same way however many are run side by side.
    """
    task_id = zlib.crc32(task.key.encode())
    return np.random.default_rng([seed, task_id, episode, 0]), np.random.default_rng([seed, task_id, episode, 1])


def run_task(
    policy: Policy,
    task: Task,
    episodes: int = OFFICIAL_EPISODES,
    seed: int = OFFICIAL_SEED,
    sampled: bool = True,
    arenas: int = 32,
    time_limit: float = None,
    on_progress=None,
    record_first: int = 0,
) -> TaskResult:
    """Scores `policy` on `task`.

    `time_limit` overrides the task's own limit. A result with an override, with a different
    number of episodes than the official one, or with a different seed, is not official.

    With `record_first`, the first that many episodes are recorded frame by frame and
    returned in the result's `replays`.
    """
    limit = task.time_limit if time_limit is None else time_limit
    arenas = min(arenas, episodes)
    games = [Game(with_opponent=task.with_opponent) for _ in range(arenas)]

    next_episode = 0
    action_rngs = [None] * arenas
    kept = [None] * arenas
    active = [False] * arenas
    recording = [None] * arenas
    replays = []

    def start(slot):
        nonlocal next_episode
        if next_episode >= episodes:
            active[slot] = False
            return
        setup_rng, action_rngs[slot] = _episode_rngs(task, seed, next_episode)
        kept[slot] = task.setup(games[slot], setup_rng)
        active[slot] = True
        if next_episode < record_first:
            game = games[slot]
            recording[slot] = {"episode": next_episode, "frames": [_frame(game.ball_info(), game.car_infos())]}
        else:
            recording[slot] = None
        next_episode += 1

    for slot in range(arenas):
        start(slot)

    successes = conceded = timeouts = finished = scored = 0
    seconds_to_score = 0.0

    while any(active):
        slots = [slot for slot in range(arenas) if active[slot]]

        states = []
        for slot in slots:
            ball, cars = games[slot].ball_info(), games[slot].car_infos()
            states.append((ball, cars))
        # The bot always drives car 0
        obs = np.stack([interface.build_observation(ball, cars, 0) for ball, cars in states])
        masks = np.stack([interface.action_mask(cars[0]) for _, cars in states])

        actions = policy.act(obs, masks, [action_rngs[slot] for slot in slots] if sampled else None)

        for slot, action, (ball, cars) in zip(slots, actions, states, strict=True):
            game = games[slot]
            game.step([action] + [0] * (len(game.cars) - 1), scripted=task.scripted(game, kept[slot], ball, cars))
            if recording[slot] is not None:
                recording[slot]["frames"].append(_frame(game.ball_info(), game.car_infos()))

            outcome = task.outcome(game)
            if outcome is None and game.seconds < limit:
                continue

            if outcome == SUCCESS:
                successes += 1
                scored += 1
                seconds_to_score += game.seconds
            elif outcome == CONCEDED:
                conceded += 1
            elif task.survive:
                successes += 1
            else:
                timeouts += 1

            finished += 1
            if recording[slot] is not None:
                replay = recording[slot]
                replay["outcome"] = outcome if outcome else ("success" if task.survive else "timeout")
                replay["seconds"] = round(game.seconds, 3)
                replays.append(replay)
            if on_progress:
                on_progress(finished, episodes)
            start(slot)

    low, high = _wilson_interval(successes, finished)
    return TaskResult(
        task=task.key,
        episodes=finished,
        successes=successes,
        conceded=conceded,
        timeouts=timeouts,
        success_rate=successes / finished if finished else 0.0,
        success_rate_low=low,
        success_rate_high=high,
        mean_seconds_to_score=seconds_to_score / scored if scored else 0.0,
        time_limit=limit,
        seed=seed,
        sampled=sampled,
        official=time_limit is None and episodes == OFFICIAL_EPISODES and sampled and seed == OFFICIAL_SEED,
        replays=sorted(replays, key=lambda r: r["episode"]),
    )
