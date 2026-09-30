"""The training curriculum: what the bot is rewarded for, and which situations it practises, by phase.

The phases follow the PISTY paper's first three: ball control, then ground scoring, then
aerial play. The trainer writes its progress to a small file, which the environment
processes read so that they all switch phase together.
"""

import json
import os
import time
from dataclasses import dataclass

PROGRESS_FILE = "progress.json"
PROGRESS_FILE_VARIABLE = "BOOST_ARENA_PROGRESS_FILE"


@dataclass(frozen=True)
class Phase:
    name: str
    start: int                 # Total timesteps at which the phase begins
    learning_rate: float
    gamma: float
    rewards: dict              # Reward name -> weight
    situations: dict           # Situation name -> relative frequency


PHASES = [
    Phase(
        name="1 Ball control", start=0, learning_rate=2e-4, gamma=0.99,
        rewards={"touch": 40, "strong_touch": 10, "player_to_ball": 5, "face_ball": 0.5, "air": 0.25, "speed": 0.2,
                 "ball_to_goal": 0.2, "goal": 0},
        situations={"kickoff": 1, "random": 1},
    ),
    Phase(
        name="2 Ground scoring", start=150_000_000, learning_rate=1.85e-4, gamma=0.993,
        rewards={"touch": 3, "strong_touch": 2, "player_to_ball": 2, "face_ball": 0.2, "air": 0.5, "speed": 0.4,
                 "ball_to_goal": 4, "goal": 30},
        situations={"kickoff": 1, "random": 1, "pass": 0.5, "empty_net": 0.5},
    ),
    Phase(
        name="3 Aerial play", start=800_000_000, learning_rate=1.5e-4, gamma=0.995,
        rewards={"touch": 0.05, "strong_touch": 0.85, "player_to_ball": 0.25, "face_ball": 0.3, "air": 1.0, "speed": 0.1,
                 "ball_to_goal": 3, "goal": 40, "air_touch": 10},
        situations={"kickoff": 0.5, "random": 1, "pass": 0.4, "empty_net": 0.5, "falling_ball": 1, "cross": 0.5,
                    "save": 0.5, "penalty": 0.5},
    ),
]

ALL_REWARDS = sorted({name for phase in PHASES for name in phase.rewards})
ALL_SITUATIONS = sorted({name for phase in PHASES for name in phase.situations})


def phase_for(timesteps: int, scale: float = 1.0) -> Phase:
    """The phase in force after `timesteps`. `scale` shrinks every phase start, for quick tests."""
    current = PHASES[0]
    for phase in PHASES:
        if timesteps >= phase.start * scale:
            current = phase
    return current


def write_progress(path: str, timesteps: int, scale: float = 1.0) -> None:
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump({"timesteps": int(timesteps), "scale": scale}, f)
    os.replace(path + ".tmp", path)   # Never half-written


class ProgressReader:
    """Reads the trainer's progress file now and then, cheaply enough to call on every reset."""

    def __init__(self, path: str = None, check_every: float = 5.0):
        self.path = path or os.environ.get(PROGRESS_FILE_VARIABLE)
        self.check_every = check_every
        self._checked = 0.0
        self._timesteps = 0
        self._scale = 1.0

    @property
    def phase(self) -> Phase:
        now = time.monotonic()
        if self.path and now - self._checked > self.check_every:
            self._checked = now
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._timesteps, self._scale = int(data["timesteps"]), float(data.get("scale", 1.0))
            except (OSError, ValueError, KeyError):
                pass
        return phase_for(self._timesteps, self._scale)
