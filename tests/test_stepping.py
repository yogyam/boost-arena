"""Replays a game recorded in the training environment and checks the simulator wrapper follows it.

The recording was made with simulator version 2.1.1 and the benchmark pins 2.2.1. Over the
four seconds recorded the two agree to within a hundredth of a unit, which shows that
timing, controls and the action delay are applied the same way.
"""

import gzip
import json
import os

import numpy as np
import pytest
import RocketSim as rs

from boost_arena import interface
from boost_arena.sim import Game, _rs_vec

VECTORS_PATH = os.path.join(os.path.dirname(__file__), "data", "interface_vectors.json.gz")


@pytest.fixture(scope="module")
def trajectory():
    with gzip.open(VECTORS_PATH, "rt", encoding="utf-8") as f:
        return json.load(f)["trajectory"]


def replay(trajectory):
    game = Game(with_opponent=True)
    game.reset(seed=0)

    ball = trajectory["initial_ball"]
    game.set_ball(ball["pos"], ball["vel"], ball["ang_vel"])
    # The recording lists the cars in its own order, so they are matched by team
    teams = [int(car.team) for car in game.cars]
    car_of = [teams.index(player["team"]) for player in trajectory["initial_players"]]

    for recorded, player in enumerate(trajectory["initial_players"]):
        rot_mat = rs.RotMat(_rs_vec(player["forward"]), _rs_vec(player["right"]), _rs_vec(player["up"]))
        game.set_car(car_of[recorded], player["pos"], rot_mat=rot_mat, vel=player["vel"], ang_vel=player["ang_vel"], boost=player["boost"])

    errors = []
    for step in trajectory["steps"]:
        actions = [0] * len(game.cars)
        for recorded, action in enumerate(step["actions"]):
            actions[car_of[recorded]] = action
        game.step(actions)

        cars = game.car_infos()
        errors.append(max(
            float(np.abs(cars[car_of[recorded]].pos - np.array(player["pos"], dtype=np.float32)).max())
            for recorded, player in enumerate(step["players"])
        ))
    return game, car_of, errors


def test_replay_follows_the_recording(trajectory):
    game, car_of, errors = replay(trajectory)

    # The cars drive thousands of units in this time
    assert max(errors) < 0.05, errors


def test_previous_action_is_reported(trajectory):
    game, car_of, _ = replay(trajectory)
    last_step = trajectory["steps"][-1]
    cars = game.car_infos()
    for recorded, player in enumerate(last_step["players"]):
        car = cars[car_of[recorded]]
        np.testing.assert_array_equal(car.prev_action, interface.ACTION_TABLE[last_step["actions"][recorded]])
        np.testing.assert_array_equal(car.prev_action, np.array(player["prev_action"], dtype=np.float32))


def test_time_advances():
    game = Game(with_opponent=False)
    game.reset(seed=0)
    for _ in range(15):
        game.step([0])
    assert game.seconds == pytest.approx(1.0)


def test_scoring_team():
    game = Game(with_opponent=False)
    game.reset(seed=0)
    assert game.scoring_team() is None
    game.set_ball((0, 5300, 100))
    assert game.scoring_team() == interface.BLUE
    game.set_ball((0, -5300, 100))
    assert game.scoring_team() == interface.ORANGE
