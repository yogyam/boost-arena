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
    assert first.successes + first.conceded + first.timeouts == 12
    assert 0 <= first.success_rate_low <= first.success_rate <= first.success_rate_high <= 1
    assert not first.official

    # The same seed gives the same result, however many games run side by side
    assert (first.successes, first.conceded, first.timeouts) == (second.successes, second.conceded, second.timeouts)
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


def test_full_length_episodes_do_not_depend_on_how_many_run_side_by_side():
    from boost_arena.policy import uniform_model

    policy = Policy(uniform_model())
    for key in ("pass", "penalty"):
        results = [run_task(policy, TASKS[key], episodes=24, seed=1, arenas=arenas) for arenas in (1, 5, 24)]
        counts = [(r.successes, r.conceded, r.timeouts, r.mean_seconds_to_score) for r in results]
        assert counts[0] == counts[1] == counts[2]
