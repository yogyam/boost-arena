"""Duels: two bots against each other, directly.

Two kinds of duel, both from the situations the tasks use:

- **Penalty duel.** One bot attacks from the penalty situation, the other keeps goal. Then
  they swap. A goal is a point for the attacker; keeping it out is a point for the keeper.
- **Kickoff duel.** Both bots start from a kickoff. The first goal within the time limit
  wins the episode; no goal is a draw.

Every pair plays the same seeded situations, both ways round. The points from every
duel feed a single rating, see rating.py.
"""

import zlib
from dataclasses import dataclass

import numpy as np

from . import interface
from .interface import BLUE, ORANGE
from .policy import Policy
from .runner import _frame
from .sim import Game
from .tasks import TASKS

DUEL_SET_VERSION = 1
OFFICIAL_EPISODES = 200      # Per direction of each duel kind
PENALTY_TIME_LIMIT = TASKS["penalty"].time_limit
KICKOFF_TIME_LIMIT = 30.0

DUEL_KINDS = {
    "penalty": "Penalty duel",
    "kickoff": "Kickoff duel",
}


@dataclass
class DuelTally:
    """Points for the blue and orange bots over a set of episodes, plus what happened."""
    episodes: int = 0
    blue_points: int = 0
    orange_points: int = 0
    draws: int = 0
    replays: list = None


def _episode_rngs(kind: str, seed: int, episode: int):
    duel_id = zlib.crc32(("duel:" + kind).encode())
    return (np.random.default_rng([seed, duel_id, episode, 0]),
            np.random.default_rng([seed, duel_id, episode, 1]),
            np.random.default_rng([seed, duel_id, episode, 2]))


def _kickoff(game: Game, rng: np.random.Generator):
    game.reset()
    game.arena.reset_kickoff(int(rng.integers(0, 2**31 - 1)))
    for i, car in enumerate(game.cars):
        state = car.get_state()
        state.boost = 100.0
        car.set_state(state)


def play_duel(kind: str, blue: Policy, orange: Policy, episodes: int = OFFICIAL_EPISODES, seed: int = 0,
              arenas: int = 32, record_first: int = 0) -> DuelTally:
    """Blue against orange in `kind`. In the penalty duel blue attacks and orange keeps goal."""
    if kind not in DUEL_KINDS:
        raise ValueError(f"Unknown duel kind {kind}")
    penalty = kind == "penalty"
    limit = PENALTY_TIME_LIMIT if penalty else KICKOFF_TIME_LIMIT

    arenas = min(arenas, episodes)
    games = [Game(with_opponent=True) for _ in range(arenas)]
    rngs = [None] * arenas
    active = [False] * arenas
    recording = [None] * arenas
    next_episode = 0
    tally = DuelTally(replays=[])

    def start(slot):
        nonlocal next_episode
        if next_episode >= episodes:
            active[slot] = False
            return
        setup_rng, blue_rng, orange_rng = _episode_rngs(kind, seed, next_episode)
        game = games[slot]
        if penalty:
            TASKS["penalty"].setup(game, setup_rng)   # Places the ball, the attacker and the keeper's start
        else:
            _kickoff(game, setup_rng)
        rngs[slot] = (blue_rng, orange_rng)
        recording[slot] = {"episode": next_episode, "frames": [_frame(game.ball_info(), game.car_infos())]} if next_episode < record_first else None
        active[slot] = True
        next_episode += 1

    for slot in range(arenas):
        start(slot)

    while any(active):
        slots = [slot for slot in range(arenas) if active[slot]]
        states = [(games[slot].ball_info(), games[slot].car_infos()) for slot in slots]
        obs = {team: np.stack([interface.build_observation(ball, cars, team) for ball, cars in states]) for team in (BLUE, ORANGE)}
        masks = {team: np.stack([interface.action_mask(cars[team]) for _, cars in states]) for team in (BLUE, ORANGE)}
        actions = {
            BLUE: blue.act(obs[BLUE], masks[BLUE], [rngs[slot][0] for slot in slots]),
            ORANGE: orange.act(obs[ORANGE], masks[ORANGE], [rngs[slot][1] for slot in slots]),
        }

        for i, slot in enumerate(slots):
            game = games[slot]
            game.step([actions[BLUE][i], actions[ORANGE][i]])
            if recording[slot] is not None:
                recording[slot]["frames"].append(_frame(game.ball_info(), game.car_infos()))

            scorer = game.scoring_team()
            if scorer is None and game.seconds < limit:
                continue

            tally.episodes += 1
            if scorer == BLUE:
                tally.blue_points += 1
                outcome = "blue"
            elif scorer == ORANGE:
                tally.orange_points += 1
                outcome = "orange"
            elif penalty:
                tally.orange_points += 1   # The keeper kept it out
                outcome = "orange"
            else:
                tally.draws += 1
                outcome = "draw"

            if recording[slot] is not None:
                replay = recording[slot]
                replay["outcome"] = outcome
                replay["seconds"] = round(game.seconds, 3)
                tally.replays.append(replay)
            start(slot)

    tally.replays.sort(key=lambda r: r["episode"])
    return tally


def run_pair(policy_a: Policy, policy_b: Policy, episodes: int = OFFICIAL_EPISODES, seed: int = 0,
             arenas: int = 32, record_first: int = 0) -> dict:
    """Every duel kind, both ways round. Returns points for A and B, per kind and in total."""
    document = {"episodes_per_direction": episodes, "seed": seed, "duel_set_version": DUEL_SET_VERSION,
                "official": episodes == OFFICIAL_EPISODES, "kinds": {}, "replays": {}}
    total_a = total_b = 0
    for kind in DUEL_KINDS:
        a_blue = play_duel(kind, policy_a, policy_b, episodes, seed, arenas, record_first)
        b_blue = play_duel(kind, policy_b, policy_a, episodes, seed, arenas, record_first)
        a_points = a_blue.blue_points + b_blue.orange_points
        b_points = a_blue.orange_points + b_blue.blue_points
        document["kinds"][kind] = {
            "a_points": a_points, "b_points": b_points, "draws": a_blue.draws + b_blue.draws,
            "a_as_blue": {"a": a_blue.blue_points, "b": a_blue.orange_points, "draws": a_blue.draws},
            "b_as_blue": {"a": b_blue.orange_points, "b": b_blue.blue_points, "draws": b_blue.draws},
        }
        if record_first:
            document["replays"][kind + "_a_blue"] = a_blue.replays
            document["replays"][kind + "_b_blue"] = b_blue.replays
        total_a += a_points
        total_b += b_points
    document["a_points"] = total_a
    document["b_points"] = total_b
    return document
