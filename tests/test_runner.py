"""Checks that scoring runs and gives the same answer every time."""

import numpy as np

from boost_arena.policy import Policy
from boost_arena.runner import run_task
from boost_arena.tasks import TASKS
from test_policy import make_model


def test_scores_repeat_and_add_up():
    policy = Policy(make_model())
    task = TASKS["empty_net"]

    # A short limit keeps the test quick, the untrained model will not score anyway
    first = run_task(policy, task, episodes=12, seed=3, arenas=4, time_limit=2.0)
    second = run_task(policy, task, episodes=12, seed=3, arenas=6, time_limit=2.0)

    assert first.episodes == 12
    assert first.successes + first.own_goals + first.timeouts == 12
    assert 0 <= first.success_rate_low <= first.success_rate <= first.success_rate_high <= 1
    assert not first.official

    # The same seed gives the same result, however many games run side by side
    assert (first.successes, first.own_goals, first.timeouts) == (second.successes, second.own_goals, second.timeouts)
    assert first.mean_seconds_to_score == second.mean_seconds_to_score


def test_situations_depend_only_on_the_seed():
    from boost_arena.sim import Game

    task = TASKS["empty_net"]
    positions = []
    for _ in range(2):
        game = Game(with_opponent=False)
        task.setup(game, np.random.default_rng(42))
        positions.append((game.ball_info().pos.copy(), game.car_infos()[0].pos.copy()))
    np.testing.assert_array_equal(positions[0][0], positions[1][0])
    np.testing.assert_array_equal(positions[0][1], positions[1][1])

    # The ball is in the attacking half and the car is further from the goal than the ball
    ball, car = positions[0]
    assert ball[1] > 0
    assert np.hypot(car[0], car[1] - 5120) > np.hypot(ball[0], ball[1] - 5120)
