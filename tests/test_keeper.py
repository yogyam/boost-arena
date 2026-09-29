"""Checks the reference keeper does what its description says."""

import math

import numpy as np
import RocketSim as rs

from boost_arena.interface import BLUE
from boost_arena.keeper import LINE_Y, START_POS, START_YAW, Keeper
from boost_arena.sim import Game


def shoot(ball_pos, ball_vel, seconds=5.0):
    """Fires a ball at the keeper's goal. Returns whether it went in and how far the keeper strayed from its line."""
    game = Game(with_opponent=True)
    game.set_car(0, (0, -4000, 17), yaw=math.pi / 2)
    game.set_car(1, START_POS, yaw=START_YAW)
    game.set_ball(ball_pos, ball_vel)
    keeper = Keeper()
    strayed = 0.0
    while game.seconds < seconds:
        cars = game.car_infos()
        strayed = max(strayed, abs(float(cars[1].pos[1]) - LINE_Y))
        game.step([0, 0], scripted={0: rs.CarControls(), 1: keeper.controls(game.ball_info(), cars[1])})
        if game.scoring_team() is not None:
            return game.scoring_team() == BLUE, strayed
    return False, strayed


def test_holds_its_line_with_nothing_to_do():
    scored, strayed = shoot((0, 0, 93.15), (0, 0, 0), seconds=10)
    assert not scored
    assert strayed < 5


def test_stops_a_slow_shot_down_the_middle():
    scored, _ = shoot((0, 2500, 93.15), (0, 1200, 0))
    assert not scored


def test_moves_across_for_a_slow_shot_to_one_side():
    for side in (-1, 1):
        scored, _ = shoot((0, 2000, 93.15), (side * 220, 1100, 0))
        assert not scored


def test_is_beaten_by_a_fast_shot_into_the_corner():
    start = np.array([0.0, 2500.0, 93.15])
    aim = np.array([820.0, 5120.0])
    direction = (aim - start[:2]) / np.linalg.norm(aim - start[:2])
    scored, _ = shoot(start, (direction[0] * 3200, direction[1] * 3200, 0))
    assert scored


def test_is_the_same_every_time():
    first = shoot((300, 2200, 93.15), (150, 1800, 250))
    second = shoot((300, 2200, 93.15), (150, 1800, 250))
    assert first == second
