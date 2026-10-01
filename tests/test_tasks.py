"""Checks that every task sets up what it says it does."""

import numpy as np
import pytest
import RocketSim as rs

from boost_arena import interface
from boost_arena.policy import Policy, uniform_model
from boost_arena.runner import overall_score, run_task
from boost_arena.sim import Game
from boost_arena.tasks import CONCEDED, TASKS


def play_untouched(task, seed):
    """Sets up an episode, parks the bot's car out of the way and lets the episode run."""
    game = Game(with_opponent=task.with_opponent)
    kept = task.setup(game, np.random.default_rng(seed))
    game.set_car(0, (3800, -4800, 17), yaw=0.0)
    while game.seconds < task.time_limit:
        ball, cars = game.ball_info(), game.car_infos()
        scripted = task.scripted(game, kept, ball, cars)
        scripted[0] = rs.CarControls()
        game.step([0] * len(game.cars), scripted=scripted)
        if task.outcome(game) is not None:
            return task.outcome(game)
    return None


@pytest.mark.parametrize("key", list(TASKS))
def test_same_seed_same_situation(key):
    task = TASKS[key]
    situations = []
    for _ in range(2):
        game = Game(with_opponent=task.with_opponent)
        task.setup(game, np.random.default_rng(11))
        ball = game.ball_info()
        situations.append(np.concatenate([ball.pos, ball.vel] + [car.pos for car in game.car_infos()]))
    np.testing.assert_array_equal(situations[0], situations[1])


@pytest.mark.parametrize("key", list(TASKS))
def test_situation_is_inside_the_field(key):
    task = TASKS[key]
    for seed in range(20):
        game = Game(with_opponent=task.with_opponent)
        task.setup(game, np.random.default_rng(seed))
        for thing in [game.ball_info()] + game.car_infos():
            assert abs(thing.pos[0]) < 4096 and abs(thing.pos[1]) < 5120 and 0 < thing.pos[2] < 2044
        assert game.scoring_team() is None
        assert game.car_infos()[0].team == interface.BLUE


@pytest.mark.parametrize("key", [key for key, task in TASKS.items() if not task.survive])
def test_nothing_is_scored_without_the_bot(key):
    for seed in range(15):
        assert play_untouched(TASKS[key], seed) is None


def test_every_shot_in_the_save_task_is_on_target():
    for seed in range(15):
        assert play_untouched(TASKS["save"], seed) == CONCEDED


def test_survival_counts_as_success():
    # With no time for the shot to arrive, every episode is survived
    result = run_task(Policy(uniform_model()), TASKS["save"], episodes=6, seed=0, arenas=3, time_limit=0.2)
    assert result.successes == 6 and result.conceded == 0 and result.timeouts == 0


def test_penalty_runs_against_the_keeper():
    result = run_task(Policy(uniform_model()), TASKS["penalty"], episodes=6, seed=0, arenas=3, time_limit=1.0)
    assert result.episodes == 6
    assert result.successes + result.conceded + result.timeouts == 6


def test_overall_score_is_the_average():
    policy = Policy(uniform_model())
    results = [run_task(policy, TASKS[key], episodes=4, seed=0, arenas=4, time_limit=0.2) for key in ("empty_net", "save")]
    assert overall_score(results) == pytest.approx(50 * (results[0].success_rate + results[1].success_rate))
