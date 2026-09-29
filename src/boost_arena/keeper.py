"""The reference keeper: a fixed program, not a trained model.

It guards the goal at +Y. It sits across the goal mouth, side on to the shooter, and drives
forwards and backwards along the goal line to stay in front of where the ball is heading.
It jumps to meet a ball that arrives too high to block from the ground, with a small hop, a
full jump or a double jump depending on the height.
It never leaves its line, never boosts and never comes out to challenge.

Being a program, it behaves the same for every bot and never changes, which a trained
keeper could not promise. It is deliberately beatable: a hard shot into a corner gets past it.
"""

import math

import numpy as np
import RocketSim as rs

from . import interface

GOAL_LINE_Y = 5120.0
LINE_Y = GOAL_LINE_Y - 260.0   # Where the keeper patrols
REACH_X = 760.0                # It does not follow the ball past the posts

START_POS = (0.0, LINE_Y, 17.0)
START_YAW = 0.0                # Facing +X, side on to a shot coming from -Y

GRAVITY = 650.0
STEP_SECONDS = interface.TICK_SKIP / interface.TICK_RATE

# The keeper's car is flat, so it has to meet the ball at the right height, not just be under it.
# It aims its centre this far below the ball's centre.
MEET_BELOW_CENTRE = 45.0
# A ball this low is blocked without leaving the ground
GROUND_BLOCK_HEIGHT = 125.0

# The three jumps it knows, as the decisions on which the jump button is pressed
JUMPS = {
    "tap": (0,),
    "held": (0, 1, 2),
    "double": (0, 1, 2, 4),   # Released on decision 3, so pressing again is a second jump
}

_profiles = None


def _jump_profiles():
    """How high the car's centre is after each decision of each jump, measured in the simulator."""
    global _profiles
    if _profiles is None:
        from .sim import Game

        _profiles = {}
        for name, presses in JUMPS.items():
            game = Game(with_opponent=False)
            game.set_car(0, START_POS, yaw=START_YAW)
            for _ in range(10):
                game.step([0], scripted={0: rs.CarControls()})

            heights = []
            for decision in range(24):
                pressed = rs.CarControls()
                pressed.jump = decision in presses
                game.step([0], scripted={0: pressed})
                heights.append(float(game.car_infos()[0].pos[2]))
            heights = np.array(heights)
            _profiles[name] = heights[: int(heights.argmax()) + 1]   # Only the way up
    return _profiles


def _clamp(value, low, high):
    return max(low, min(high, value))


class Keeper:
    """One keeper for one episode. Make a new one, or call reset(), when an episode starts."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._jump = None       # The jump in progress, None while on the ground
        self._decision = 0      # Decisions since it took off

    def _plan_jump(self, height, seconds_away):
        """The jump that meets a ball arriving at `height`, if now is the moment to take off."""
        meet_at = height - MEET_BELOW_CENTRE
        for name, profile in _jump_profiles().items():
            if profile[-1] < meet_at:
                continue  # This jump does not get high enough
            decisions_needed = int(np.argmax(profile >= meet_at)) + 1
            if seconds_away <= (decisions_needed + 0.5) * STEP_SECONDS:
                return name
            return None  # It can reach, but it is too early
        return None  # Out of reach

    def controls(self, ball, car) -> rs.CarControls:
        """The keeper's controls for this decision. `ball` is a BallInfo and `car` the keeper's CarInfo."""
        out = rs.CarControls()
        if car.is_demoed:
            self.reset()
            return out

        # Where, when and how high the ball will cross the keeper's line, if it is coming this way
        target_x = float(ball.pos[0])
        seconds_away = math.inf
        height = float(ball.pos[2])
        if ball.vel[1] > 50 and ball.pos[1] < LINE_Y:
            seconds_away = float((LINE_Y - ball.pos[1]) / ball.vel[1])
            target_x = float(ball.pos[0] + ball.vel[0] * seconds_away)
            height = float(ball.pos[2] + ball.vel[2] * seconds_away - 0.5 * GRAVITY * seconds_away**2)
            height = max(height, 93.0)  # It bounces, it does not go through the floor
        target_x = _clamp(target_x, -REACH_X, REACH_X)

        forward = np.array([car.forward[0], car.forward[1]], dtype=np.float64)
        along = 1.0 if forward[0] >= 0 else -1.0   # Which way along the line the nose points

        if self._jump is not None:
            self._decision += 1
            if car.is_on_ground and self._decision > 3:
                self.reset()  # It has landed
            else:
                out.jump = self._decision in JUMPS[self._jump]
                return out

        if not car.is_on_ground:
            return out  # Knocked into the air, nothing to do until it lands

        # Drive along the line towards the target, slowing as it gets close
        offset = (target_x - float(car.pos[0])) * along
        speed = float(car.vel[0]) * along
        out.throttle = _clamp((offset - 0.25 * speed) / 120.0, -1.0, 1.0)

        # Hold the line: steer back towards it and keep the nose pointing along it
        heading_error = math.atan2(forward[1], forward[0] * along) * along
        line_error = (float(car.pos[1]) - LINE_Y) * along
        direction = 1.0 if speed >= 0 else -1.0
        out.steer = _clamp(direction * (-2.0 * heading_error - line_error / 250.0), -1.0, 1.0)

        # Take off for a ball that will arrive too high to block from the ground
        if height > GROUND_BLOCK_HEIGHT and abs(target_x - float(car.pos[0])) < 300:
            self._jump = self._plan_jump(height, seconds_away)
            if self._jump is not None:
                self._decision = 0
                out.jump = True
        return out
