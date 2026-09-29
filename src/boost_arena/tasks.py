"""The benchmark tasks.

In every task the bot drives the blue car and attacks the goal at +Y. A task sets up a
situation from a random generator, so the same seed always gives the same situation.
"""

import math
from dataclasses import dataclass

import numpy as np

from .sim import Game

BALL_REST_HEIGHT = 93.15
CAR_REST_HEIGHT = 17.0

FIELD_HALF_WIDTH = 4096
FIELD_HALF_LENGTH = 5120


@dataclass(frozen=True)
class Task:
    key: str
    name: str
    description: str
    time_limit: float  # Seconds before an episode without a goal counts as a failure
    with_opponent: bool = False

    def setup(self, game: Game, rng: np.random.Generator):
        raise NotImplementedError


class EmptyNetFinish(Task):
    """A still ball in the attacking half, the car a short way behind it, nobody in goal."""

    def setup(self, game: Game, rng: np.random.Generator):
        game.reset(seed=int(rng.integers(0, 2**31 - 1)))

        ball = np.array([rng.uniform(-3000, 3000), rng.uniform(1000, 4000), BALL_REST_HEIGHT])
        game.set_ball(ball)

        # The car starts on the side of the ball away from the goal, give or take 40 degrees,
        # and faces the ball, give or take 30 degrees
        away_from_goal = math.atan2(ball[1] - FIELD_HALF_LENGTH, ball[0])
        direction = away_from_goal + math.radians(rng.uniform(-40, 40))
        distance = rng.uniform(700, 1500)
        car = np.array([
            np.clip(ball[0] + distance * math.cos(direction), -FIELD_HALF_WIDTH + 300, FIELD_HALF_WIDTH - 300),
            np.clip(ball[1] + distance * math.sin(direction), -FIELD_HALF_LENGTH + 300, FIELD_HALF_LENGTH - 300),
            CAR_REST_HEIGHT,
        ])
        facing = math.atan2(ball[1] - car[1], ball[0] - car[0]) + math.radians(rng.uniform(-30, 30))
        game.set_car(0, car, yaw=facing)


TASKS = {
    task.key: task
    for task in [
        EmptyNetFinish(
            key="empty_net",
            name="Empty-net finish",
            description="Score from a still ball in the attacking half, with nobody in goal.",
            time_limit=20.0,
        ),
    ]
}
