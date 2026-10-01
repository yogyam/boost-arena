"""The Boost Arena interface, version 1.

This is what every bot sees and what it can do. It is described in words in
docs/INTERFACE.md; this file is the reference implementation. It has no dependency
on the simulator, so it can be tested and reused on its own.
"""

from dataclasses import dataclass, field

import numpy as np

INTERFACE_VERSION = 1

OBS_SIZE = 53
NUM_ACTIONS = 90
NUM_GROUND_ACTIONS = 24

# The simulator runs at 120 ticks per second. A bot chooses an action every TICK_SKIP ticks.
# The chosen action only takes effect after ACTION_DELAY ticks, like a reaction time.
TICK_RATE = 120
TICK_SKIP = 8
ACTION_DELAY = 7

BLUE = 0
ORANGE = 1

# Observation scaling
# The field, in Unreal units. Everything that needs these takes them from here
FIELD_HALF_WIDTH = 4096
FIELD_HALF_LENGTH = 5120
CEILING = 2044
GOAL_HALF_WIDTH = 893
BALL_RADIUS = 91.25
BALL_REST_HEIGHT = 93.15
CAR_REST_HEIGHT = 17.0
GRAVITY = 650.0
GOAL_LINE = 5124.25 + BALL_RADIUS  # The ball's centre is past this when it is in a goal

POS_SCALE = np.array([1 / FIELD_HALF_WIDTH, 1 / FIELD_HALF_LENGTH, 1 / CEILING], dtype=np.float32)
VEL_SCALE = np.float32(1 / 2300)
ANG_VEL_SCALE = np.float32(1 / 5.5)

# Orange sees the field rotated by half a turn, so both teams attack the same way in their own view
_INVERT = np.array([-1, -1, 1], dtype=np.float32)

# Index of each control within an action
THROTTLE, STEER, PITCH, YAW, ROLL, JUMP, BOOST, HANDBRAKE = range(8)


def _vec(values=(0, 0, 0)):
    return np.asarray(values, dtype=np.float32)


@dataclass
class BallInfo:
    pos: np.ndarray = field(default_factory=_vec)
    vel: np.ndarray = field(default_factory=_vec)
    ang_vel: np.ndarray = field(default_factory=_vec)


@dataclass
class CarInfo:
    team: int = BLUE
    pos: np.ndarray = field(default_factory=_vec)
    vel: np.ndarray = field(default_factory=_vec)
    ang_vel: np.ndarray = field(default_factory=_vec)
    forward: np.ndarray = field(default_factory=lambda: _vec((1, 0, 0)))
    up: np.ndarray = field(default_factory=lambda: _vec((0, 0, 1)))
    boost: float = 100.0
    is_on_ground: bool = True
    has_flip_or_jump: bool = True
    is_demoed: bool = False
    has_world_contact: bool = False
    world_contact_normal: np.ndarray = field(default_factory=_vec)
    # The controls this car chose on its previous decision, all zero at the start of an episode
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))


def _build_action_table():
    signed = (-1, 0, 1)
    binary = (0, 1)
    actions = []

    # Ground: throttle, steer, boost, handbrake. Yaw copies steer.
    for throttle in signed:
        for steer in signed:
            for boost in binary:
                for handbrake in binary:
                    if boost == 1 and throttle != 1:
                        continue  # Boosting already drives forward
                    actions.append((throttle, steer, 0, steer, 0, 0, boost, handbrake))
    assert len(actions) == NUM_GROUND_ACTIONS

    # Aerial: pitch, yaw, roll, jump, boost. Throttle copies boost and steer copies yaw.
    for pitch in signed:
        for yaw in signed:
            for roll in signed:
                for jump in binary:
                    for boost in binary:
                        if jump == 1 and yaw != 0:
                            continue  # A side flip only needs roll
                        if pitch == 0 and roll == 0 and jump == 0:
                            continue  # Already covered by a ground action
                        handbrake = int(jump == 1 and (pitch != 0 or yaw != 0 or roll != 0))
                        actions.append((boost, yaw, pitch, yaw, roll, jump, boost, handbrake))

    table = np.array(actions, dtype=np.float32)
    assert table.shape == (NUM_ACTIONS, 8)
    return table


ACTION_TABLE = _build_action_table()
ACTION_TABLE.setflags(write=False)


def _build_masks():
    index = np.arange(NUM_ACTIONS)
    is_ground = index < NUM_GROUND_ACTIONS
    jump = ACTION_TABLE[:, JUMP] != 0
    boost = ACTION_TABLE[:, BOOST] != 0

    # Compatibility note: the first aerial action (index 24) is left out of the airborne set.
    # The bots this interface was defined from were trained that way, so it is kept.
    air = (index > NUM_GROUND_ACTIONS) & ~jump

    # Ground actions that are also useful in the air: pure yaw, with or without boost
    yaw_only = (ACTION_TABLE[:, THROTTLE] == ACTION_TABLE[:, BOOST]) & (
        (ACTION_TABLE[:, YAW] != 0) == (ACTION_TABLE[:, HANDBRAKE] != 0)
    )
    air = air | (is_ground & yaw_only)

    return is_ground, air, jump, boost


_GROUND_MASK, _AIR_MASK, _JUMP_MASK, _BOOST_MASK = _build_masks()


def action_mask(car: CarInfo) -> np.ndarray:
    """Which of the 90 actions the car may choose right now, as 90 booleans."""
    mask = (_GROUND_MASK if car.is_on_ground else _AIR_MASK).copy()

    if car.boost == 0:
        mask &= ~_BOOST_MASK

    # Lying on its roof counts as being able to jump, which is how a car rights itself
    is_turtled = car.has_world_contact and car.world_contact_normal[2] > 0.9
    if car.has_flip_or_jump or is_turtled:
        mask |= _JUMP_MASK

    return mask


def _car_block(car: CarInfo, invert: bool) -> np.ndarray:
    flip = _INVERT if invert else np.ones(3, dtype=np.float32)
    return np.concatenate(
        [
            _vec(car.pos) * flip * POS_SCALE,
            _vec(car.forward) * flip,
            _vec(car.up) * flip,
            _vec(car.vel) * flip * VEL_SCALE,
            _vec(car.ang_vel) * flip * ANG_VEL_SCALE,
            np.array([car.is_on_ground, car.has_flip_or_jump, car.is_demoed], dtype=np.float32),
        ]
    )


def build_observation(ball: BallInfo, cars: list, index: int) -> np.ndarray:
    """The 53 numbers the car at `index` sees.

    Layout: ball (9), own previous action (8), own car (18), opponent's car (18).
    Without an opponent the last 18 numbers are zero.
    """
    me = cars[index]
    invert = me.team == ORANGE
    flip = _INVERT if invert else np.ones(3, dtype=np.float32)

    opponents = [car for car in cars if car is not me and car.team != me.team]
    if len(opponents) > 1:
        raise ValueError("Interface version 1 is 1v1: at most one opponent")

    obs = np.concatenate(
        [
            _vec(ball.pos) * flip * POS_SCALE,
            _vec(ball.vel) * flip * VEL_SCALE,
            _vec(ball.ang_vel) * flip * ANG_VEL_SCALE,
            np.asarray(me.prev_action, dtype=np.float32),
            _car_block(me, invert),
            _car_block(opponents[0], invert) if opponents else np.zeros(18, dtype=np.float32),
        ]
    ).astype(np.float32)

    assert obs.shape == (OBS_SIZE,)
    return obs
