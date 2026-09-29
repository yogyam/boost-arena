"""Development aid: how often different kinds of shot beat the reference keeper."""

import math

import numpy as np
import RocketSim as rs

from boost_arena.interface import BLUE
from boost_arena.keeper import LINE_Y, START_POS, START_YAW, Keeper
from boost_arena.sim import Game, ball_path

GOAL_LINE = 5124.25 + 91.25


def trial(ball_pos, ball_vel, seconds=5.0):
    game = Game(with_opponent=True)
    game.set_car(0, (0, -4000, 17), yaw=math.pi / 2)  # The shooter stays out of the way
    game.set_car(1, START_POS, yaw=START_YAW)
    game.set_ball(ball_pos, ball_vel)
    keeper = Keeper()
    highest = 0.0
    while game.seconds < seconds:
        cars = game.car_infos()
        highest = max(highest, float(cars[1].pos[2]))
        game.step([0, 0], scripted={0: rs.CarControls(), 1: keeper.controls(game.ball_info(), cars[1])})
        if game.scoring_team() is not None:
            return game.scoring_team() == BLUE, highest
    return False, highest


def shots(rng, speed_range, x_range, height_range, n=200):
    goals = on_target = 0
    highest = 0.0
    for _ in range(n):
        start = np.array([rng.uniform(-1500, 1500), rng.uniform(1500, 3000), 93.15])
        aim = np.array([rng.uniform(*x_range), 5120.0, rng.uniform(*height_range)])
        flat = aim[:2] - start[:2]
        t = np.hypot(*flat) / rng.uniform(*speed_range)
        vel = np.array([flat[0] / t, flat[1] / t, (aim[2] - start[2]) / t + 0.5 * 650 * t])
        if not (np.abs(ball_path(start, vel, 5.0)[:, 1]) > GOAL_LINE).any():
            continue
        on_target += 1
        scored, h = trial(start, vel)
        goals += scored
        highest = max(highest, h)
    return goals, on_target, highest


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for label, speed, xs, hs in [
        ("slow, central, low", (900, 1400), (-300, 300), (95, 150)),
        ("slow, wide, low", (900, 1400), (-800, 800), (95, 150)),
        ("fast, central, low", (2200, 3200), (-300, 300), (95, 150)),
        ("fast, corners, low", (2200, 3200), (600, 820), (95, 150)),
        ("medium, central, waist high", (1500, 2400), (-300, 300), (180, 300)),
        ("medium, central, head high", (1500, 2400), (-300, 300), (300, 450)),
        ("medium, anywhere, top of goal", (1500, 2400), (-800, 800), (450, 600)),
    ]:
        g, n, h = shots(rng, speed, xs, hs)
        print(f"{label:30s} {g:3d} of {n:3d} on-target shots beat the keeper ({g / max(n, 1):.0%}), keeper reached height {h:.0f}")
