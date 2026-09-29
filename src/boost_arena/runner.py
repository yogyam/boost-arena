"""Plays a task many times and scores the result."""

import math
import zlib
from dataclasses import asdict, dataclass

import numpy as np

from . import __version__, interface
from .interface import BLUE
from .policy import Policy
from .sim import SIMULATOR_VERSION, Game
from .tasks import Task


@dataclass
class TaskResult:
    task: str
    episodes: int
    successes: int
    own_goals: int
    timeouts: int
    success_rate: float
    success_rate_low: float   # 95% confidence interval
    success_rate_high: float
    mean_seconds_to_score: float  # Over successful episodes, 0 if there were none
    time_limit: float
    seed: int
    sampled: bool
    official: bool
    interface_version: int = interface.INTERFACE_VERSION
    simulator_version: str = SIMULATOR_VERSION
    benchmark_version: str = __version__

    def to_dict(self):
        return asdict(self)


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


def run_task(policy: Policy, task: Task, episodes: int = 1000, seed: int = 0, sampled: bool = True,
             arenas: int = 32, time_limit: float = None, on_progress=None) -> TaskResult:
    """Scores `policy` on `task`.

    `time_limit` overrides the task's own limit. A result with an override is not official.
    """
    if task.with_opponent:
        raise NotImplementedError("Tasks with an opponent are not available yet")

    limit = task.time_limit if time_limit is None else time_limit
    arenas = min(arenas, episodes)
    games = [Game(with_opponent=False) for _ in range(arenas)]

    next_episode = 0
    action_rngs = [None] * arenas
    active = [False] * arenas

    def start(slot):
        nonlocal next_episode
        if next_episode >= episodes:
            active[slot] = False
            return
        setup_rng, action_rngs[slot] = _episode_rngs(task, seed, next_episode)
        task.setup(games[slot], setup_rng)
        active[slot] = True
        next_episode += 1

    for slot in range(arenas):
        start(slot)

    successes = own_goals = timeouts = finished = 0
    seconds_to_score = 0.0

    while any(active):
        slots = [slot for slot in range(arenas) if active[slot]]
        observed = [games[slot].observe() for slot in slots]
        obs = np.concatenate([o for o, _ in observed])
        masks = np.concatenate([m for _, m in observed])

        actions = policy.act(obs, masks, [action_rngs[slot] for slot in slots] if sampled else None)

        for slot, action in zip(slots, actions):
            game = games[slot]
            game.step([action])

            scorer = game.scoring_team()
            if scorer is None and game.seconds < limit:
                continue

            if scorer == BLUE:
                successes += 1
                seconds_to_score += game.seconds
            elif scorer is not None:
                own_goals += 1
            else:
                timeouts += 1

            finished += 1
            if on_progress:
                on_progress(finished, episodes)
            start(slot)

    low, high = _wilson_interval(successes, finished)
    return TaskResult(
        task=task.key,
        episodes=finished,
        successes=successes,
        own_goals=own_goals,
        timeouts=timeouts,
        success_rate=successes / finished if finished else 0.0,
        success_rate_low=low,
        success_rate_high=high,
        mean_seconds_to_score=seconds_to_score / successes if successes else 0.0,
        time_limit=limit,
        seed=seed,
        sampled=sampled,
        official=time_limit is None,
    )
