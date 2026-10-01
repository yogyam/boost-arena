"""The benchmark tasks.

In every task the bot drives the blue car. It attacks the goal at +Y and defends the one
at -Y. A task sets up a situation from a random generator, so the same seed always gives
the same situation.
"""

import math
from dataclasses import dataclass

import numpy as np

from . import keeper
from .interface import (
    BALL_REST_HEIGHT,
    BLUE,
    CAR_REST_HEIGHT,
    FIELD_HALF_LENGTH,
    FIELD_HALF_WIDTH,
    GOAL_LINE,
    GRAVITY,
    ORANGE,
)
from .sim import Game, ball_path

TASK_SET_VERSION = 1


SUCCESS = "success"
CONCEDED = "conceded"  # The ball went into the bot's own goal


def _degrees(rng, spread):
    return math.radians(rng.uniform(-spread, spread))


def _clamp_to_field(x, y, margin=300):
    return (
        float(np.clip(x, -FIELD_HALF_WIDTH + margin, FIELD_HALF_WIDTH - margin)),
        float(np.clip(y, -FIELD_HALF_LENGTH + margin, FIELD_HALF_LENGTH - margin)),
    )


def _goes_in_untouched(pos, vel, seconds, own_goal=False):
    """Whether a ball that nobody touches ends up in a goal: the bot's own, or either."""
    path = ball_path(pos, vel, seconds)
    if own_goal:
        return bool((path[:, 1] < -GOAL_LINE).any())
    return bool((np.abs(path[:, 1]) > GOAL_LINE).any())


def _velocity_towards(start, target, speed):
    """A flat velocity of `speed` from `start` towards `target`."""
    direction = np.array([target[0] - start[0], target[1] - start[1]], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return direction * speed


@dataclass(frozen=True)
class Task:
    key: str
    name: str
    description: str
    time_limit: float  # Seconds
    with_opponent: bool = False
    survive: bool = False  # True if lasting until the time limit is the success

    def setup(self, game: Game, rng: np.random.Generator):
        """Sets up the situation. May return something to keep for the episode, passed to scripted()."""
        raise NotImplementedError

    def scripted(self, game: Game, kept, ball, cars) -> dict:
        """Controls for the cars that a program drives, by car index."""
        return {}

    def outcome(self, game: Game):
        """SUCCESS or CONCEDED once the episode is decided by a goal, otherwise None."""
        scorer = game.scoring_team()
        if scorer == BLUE:
            return SUCCESS
        if scorer == ORANGE:
            return CONCEDED
        return None

    def _place_behind_ball(self, game, rng, ball, distance, spread, facing_error, speed=(0, 0)):
        """Puts the bot's car on the side of the ball away from the goal it attacks, facing the ball."""
        away_from_goal = math.atan2(ball[1] - FIELD_HALF_LENGTH, ball[0])
        direction = away_from_goal + _degrees(rng, spread)
        gap = rng.uniform(*distance)
        x, y = _clamp_to_field(ball[0] + gap * math.cos(direction), ball[1] + gap * math.sin(direction))
        facing = math.atan2(ball[1] - y, ball[0] - x) + _degrees(rng, facing_error)
        pace = rng.uniform(*speed)
        game.set_car(0, (x, y, CAR_REST_HEIGHT), yaw=facing, vel=(pace * math.cos(facing), pace * math.sin(facing), 0))


class EmptyNetFinish(Task):
    def setup(self, game, rng):
        game.reset()
        ball = np.array([rng.uniform(-3000, 3000), rng.uniform(1000, 4000), BALL_REST_HEIGHT])
        game.set_ball(ball)
        self._place_behind_ball(game, rng, ball, distance=(700, 1500), spread=40, facing_error=30)


class PassFinish(Task):
    def setup(self, game, rng):
        game.reset()

        # The pass comes in from a wing towards the channel in front of the car.
        # A pass that would roll in by itself is drawn again: the bot has to do the scoring.
        while True:
            side = rng.choice([-1, 1])
            ball = np.array([side * rng.uniform(2000, 3200), rng.uniform(1800, 3600), BALL_REST_HEIGHT])
            target = (rng.uniform(-700, 700), rng.uniform(1500, 3200))
            flat = _velocity_towards(ball, target, rng.uniform(900, 1700))
            velocity = (flat[0], flat[1], rng.uniform(0, 250))
            if not _goes_in_untouched(ball, velocity, self.time_limit):
                break
        game.set_ball(ball, velocity)

        facing = math.pi / 2 + _degrees(rng, 25)
        pace = rng.uniform(0, 600)
        game.set_car(
            0,
            (rng.uniform(-1200, 1200), rng.uniform(-1500, 300), CAR_REST_HEIGHT),
            yaw=facing,
            vel=(pace * math.cos(facing), pace * math.sin(facing), 0),
        )


class FallingBall(Task):
    def setup(self, game, rng):
        game.reset()

        while True:
            ball = np.array([rng.uniform(-1800, 1800), rng.uniform(800, 3200), rng.uniform(1200, 1800)])
            drift = rng.uniform(0, 150)
            heading = rng.uniform(-math.pi, math.pi)
            velocity = (drift * math.cos(heading), drift * math.sin(heading), rng.uniform(-1200, -400))
            if not _goes_in_untouched(ball, velocity, self.time_limit):
                break
        game.set_ball(ball, velocity)

        self._place_behind_ball(game, rng, ball, distance=(1200, 2000), spread=20, facing_error=20, speed=(400, 900))


class AerialCross(Task):
    def setup(self, game, rng):
        game.reset()

        # A high ball from a wing, arcing towards the space in front of the goal
        while True:
            side = rng.choice([-1, 1])
            ball = np.array([side * rng.uniform(2600, 3500), rng.uniform(2600, 3800), rng.uniform(250, 600)])
            target = (rng.uniform(-500, 500), rng.uniform(3800, 4600))
            flat = _velocity_towards(ball, target, rng.uniform(1000, 1600))
            velocity = (flat[0], flat[1], rng.uniform(500, 900))
            if not _goes_in_untouched(ball, velocity, self.time_limit):
                break
        game.set_ball(ball, velocity)

        facing = math.pi / 2 + _degrees(rng, 30)
        pace = rng.uniform(0, 800)
        game.set_car(
            0,
            (rng.uniform(-1000, 1000), rng.uniform(-500, 1500), CAR_REST_HEIGHT),
            yaw=facing,
            vel=(pace * math.cos(facing), pace * math.sin(facing), 0),
        )


class Save(Task):
    """A shot at the bot's goal. Every shot goes in if nobody touches it."""

    def _shot(self, rng):
        start = np.array([rng.uniform(-2200, 2200), rng.uniform(-2600, -1200), rng.uniform(BALL_REST_HEIGHT, 300)])
        aim = np.array([rng.uniform(-750, 750), -FIELD_HALF_LENGTH, rng.uniform(100, 500)])
        flat = aim[:2] - start[:2]
        seconds = float(np.linalg.norm(flat)) / rng.uniform(1400, 2600)
        velocity = np.array(
            [
                flat[0] / seconds,
                flat[1] / seconds,
                (aim[2] - start[2]) / seconds + 0.5 * GRAVITY * seconds,
            ]
        )
        return start, velocity

    def setup(self, game, rng):
        game.reset()

        # Bounces and drag can take a shot wide, so shots are drawn until one is on target
        while True:
            start, velocity = self._shot(rng)
            if _goes_in_untouched(start, velocity, 4.0, own_goal=True):
                break
        game.set_ball(start, velocity)

        x, y = rng.uniform(-500, 500), -FIELD_HALF_LENGTH + rng.uniform(150, 450)
        facing = math.atan2(start[1] - y, start[0] - x) + _degrees(rng, 30)
        game.set_car(0, (x, y, CAR_REST_HEIGHT), yaw=facing)


class Penalty(Task):
    def setup(self, game, rng):
        game.reset()

        ball = np.array([rng.uniform(-800, 800), rng.uniform(2600, 3400), BALL_REST_HEIGHT])
        game.set_ball(ball)
        self._place_behind_ball(game, rng, ball, distance=(900, 1500), spread=25, facing_error=15)

        game.set_car(1, keeper.START_POS, yaw=keeper.START_YAW)
        return keeper.Keeper()

    def scripted(self, game, kept, ball, cars):
        return {1: kept.controls(ball, cars[1])}


TASKS = {
    task.key: task
    for task in [
        EmptyNetFinish(
            key="empty_net",
            name="Empty-net finish",
            time_limit=20.0,
            description="Score from a still ball in the attacking half, with nobody in goal.",
        ),
        PassFinish(
            key="pass",
            name="Pass finish",
            time_limit=20.0,
            description="Score from a ball rolling in from the wing, with nobody in goal.",
        ),
        FallingBall(
            key="falling_ball",
            name="Falling ball",
            time_limit=20.0,
            description="Score from a ball dropping out of the air, with nobody in goal.",
        ),
        AerialCross(
            key="cross",
            name="Aerial cross",
            time_limit=15.0,
            description="Score from a high cross arcing in from the wing, with nobody in goal.",
        ),
        Save(
            key="save",
            name="Save",
            time_limit=6.0,
            survive=True,
            description="Keep out a shot that is on target, for six seconds.",
        ),
        Penalty(
            key="penalty",
            name="Penalty shoot-out",
            time_limit=10.0,
            with_opponent=True,
            description="Score from a still ball past the reference keeper.",
        ),
    ]
}
