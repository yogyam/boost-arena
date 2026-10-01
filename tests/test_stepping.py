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

    ball = trajectory["initial_ball"]
    game.set_ball(ball["pos"], ball["vel"], ball["ang_vel"])
    # The recording lists the cars in its own order, so they are matched by team
    teams = [int(car.team) for car in game.cars]
    car_of = [teams.index(player["team"]) for player in trajectory["initial_players"]]

    for recorded, player in enumerate(trajectory["initial_players"]):
        rot_mat = rs.RotMat(_rs_vec(player["forward"]), _rs_vec(player["right"]), _rs_vec(player["up"]))
        game.set_car(
            car_of[recorded], player["pos"], rot_mat=rot_mat, vel=player["vel"], ang_vel=player["ang_vel"], boost=player["boost"]
        )

    errors = []
    for step in trajectory["steps"]:
        actions = [0] * len(game.cars)
        for recorded, action in enumerate(step["actions"]):
            actions[car_of[recorded]] = action
        game.step(actions)

        cars = game.car_infos()
        errors.append(
            max(
                float(np.abs(cars[car_of[recorded]].pos - np.array(player["pos"], dtype=np.float32)).max())
                for recorded, player in enumerate(step["players"])
            )
        )
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
    for _ in range(15):
        game.step([0])
    assert game.seconds == pytest.approx(1.0)


def test_scoring_team():
    game = Game(with_opponent=False)
    assert game.scoring_team() is None
    game.set_ball((0, 5300, 100))
    assert game.scoring_team() == interface.BLUE
    game.set_ball((0, -5300, 100))
    assert game.scoring_team() == interface.ORANGE


def test_a_demolished_car_stays_out():
    # The blue car drives into the orange one at full speed
    game = Game(with_opponent=True)
    game.set_car(0, (0, 0, 17), yaw=0.0, vel=(2300, 0, 0))
    game.set_car(1, (700, 0, 17), yaw=1.57)
    full_speed_ahead = 18  # Throttle and boost, no steering
    assert list(interface.ACTION_TABLE[full_speed_ahead]) == [1, 0, 0, 0, 0, 0, 1, 0]

    for _ in range(8):
        game.step([full_speed_ahead, 0])
    wreck = game.car_infos()[1]
    assert wreck.is_demoed
    where = wreck.pos.copy()

    for _ in range(15 * 8):  # Eight seconds, well past the simulator's usual three
        game.step([full_speed_ahead, 0])

    wreck = game.car_infos()[1]
    assert wreck.is_demoed
    np.testing.assert_array_equal(wreck.pos, where)


def test_reset_leaves_nothing_behind():
    def play(game):
        game.reset()
        game.set_ball((0, 1000, 93.15), (300, 200, 0))
        game.set_car(0, (-500, -200, 17), yaw=0.7)
        game.set_car(1, (400, 2500, 17), yaw=-2.0)
        for step in range(45):
            game.step([(step * 7) % 24, (step * 5) % 24])
        ball, cars = game.ball_info(), game.car_infos()
        return np.concatenate([ball.pos, ball.vel] + [np.concatenate([c.pos, c.vel, c.forward]) for c in cars])

    fresh = play(Game(with_opponent=True))

    used = Game(with_opponent=True)
    used.set_ball((0, 0, 500), (2000, 1500, 300))
    used.set_car(0, (1000, 1000, 17), yaw=2.0)
    used.set_car(1, (1200, 1000, 17), yaw=-1.0)
    for step in range(200):
        used.step([(step * 11) % 90 if step % 3 else 5, 3])

    np.testing.assert_array_equal(play(used), fresh)


def test_the_physics_settings_are_in_force():
    from boost_arena import sim

    game = Game(with_opponent=True)
    for _ in range(2):  # They must survive a reset too
        mutators = game.arena.get_mutator_config()
        assert mutators.boost_used_per_second == sim.BOOST_USED_PER_SECOND
        assert mutators.car_spawn_boost_amount == sim.CAR_SPAWN_BOOST
        assert mutators.respawn_delay == sim.RESPAWN_DELAY
        assert [car.get_config().hitbox_size.x for car in game.cars] == [game.cars[0].get_config().hitbox_size.x] * 2
        game.reset()

    # Two seconds of boosting uses two boost. At the simulator's usual rate it would use 67.
    game.set_car(0, (-3000, -4000, 17), yaw=0.8)
    game.set_car(1, (3000, 4000, 17), yaw=0.0)
    for _ in range(30):
        game.step([18, 0])
    assert game.car_infos()[0].boost == pytest.approx(98, abs=0.3)
