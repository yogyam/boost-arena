"""Checks that the starter kit's environment follows the rulebook and that a trained network exports."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rlgym_learn_algos")

from boost_arena import interface
from boost_arena.policy import Policy
from boost_arena.starter import env as kit
from boost_arena.starter.actor import MaskedDiscreteFF
from boost_arena.starter.curriculum import PHASES, ALL_REWARDS, phase_for


def make_env(situation_rng_seed=0):
    from rlgym.api import RLGym
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence

    from boost_arena.starter.curriculum import ProgressReader

    progress = ProgressReader(path=None)
    return RLGym(
        state_mutator=MutatorSequence(FixedTeamSizeMutator(blue_size=1, orange_size=1),
                                      kit.SituationMutator(progress, np.random.default_rng(situation_rng_seed))),
        obs_builder=kit.ArenaObs(),
        action_parser=kit.ArenaAction(),
        reward_fn=kit.CurriculumReward(progress),
        termination_cond=kit.GoalCondition(),
        truncation_cond=kit.TimeoutCondition(),
        transition_engine=RocketSimEngine(rlbot_delay=False),
    )


def test_observation_has_the_rulebook_shape_and_a_mask():
    env = make_env()
    obs = env.reset()
    assert len(obs) == 2
    for agent, values in obs.items():
        assert values.shape == (interface.OBS_SIZE + interface.NUM_ACTIONS,)
        mask = values[interface.OBS_SIZE:]
        assert set(np.unique(mask)) <= {0.0, 1.0} and mask.sum() > 0
        assert values[9:17].tolist() == [0.0] * 8   # No previous action at the start


def test_previous_action_and_timing_follow_the_rulebook():
    env = make_env()
    env.reset()
    agents = env.agents
    actions = {agent: np.array([23]) for agent in agents}
    engine_actions = env.action_parser.parse_actions(actions, env.state, env.shared_info)
    for rows in engine_actions.values():
        assert rows.shape == (interface.TICK_SKIP, 8)
        np.testing.assert_array_equal(rows[: interface.ACTION_DELAY], np.zeros((interface.ACTION_DELAY, 8)))
        np.testing.assert_array_equal(rows[interface.ACTION_DELAY:], np.tile(interface.ACTION_TABLE[23], (interface.TICK_SKIP - interface.ACTION_DELAY, 1)))

    obs, *_ = env.step(actions)
    for values in obs.values():
        np.testing.assert_array_equal(values[9:17], interface.ACTION_TABLE[23])


def test_physics_settings_match_the_rules():
    from rlgym.rocket_league import common_values

    env = make_env()
    env.reset()
    for car in env.state.cars.values():
        assert car.hitbox_type == common_values.PLANK
        assert car.boost_amount == 100.0
    assert env.state.config.boost_consumption * common_values.BOOST_CONSUMPTION_RATE == pytest.approx(1.0, abs=1e-3)


def test_every_phase_situation_can_be_set_up():
    for phase in PHASES:
        for name in phase.situations:
            progress = type("P", (), {"phase": phase})()
            mutator = kit.SituationMutator(progress, np.random.default_rng(1))
            env = make_env()
            env.reset()
            state = env.state
            for _ in range(5):
                mutator.apply(state, {})
                assert abs(state.ball.position[0]) < 4096 and abs(state.ball.position[1]) < 5200
                for car in state.cars.values():
                    assert abs(car.physics.position[0]) < 4096 and abs(car.physics.position[1]) < 5200


def test_rewards_are_finite_and_use_every_weight():
    env = make_env()
    env.reset()
    for _ in range(20):
        obs, rewards, terminated, truncated = env.step({agent: np.array([18]) for agent in env.agents})
        assert all(np.isfinite(r) for r in rewards.values())
    assert set(ALL_REWARDS) >= set(PHASES[0].rewards)
    assert phase_for(0).name.startswith("1") and phase_for(10**12).name.startswith("3")


def test_actor_masks_and_exports():
    actor = MaskedDiscreteFF((64, 64), True, torch.float32, torch.device("cpu"))
    obs = np.random.default_rng(0).normal(size=(32, interface.OBS_SIZE)).astype(np.float32)
    mask = np.zeros((32, interface.NUM_ACTIONS), dtype=np.float32)
    mask[:, 5:9] = 1
    actions, log_probs = actor.get_actions(list(range(32)), np.concatenate([obs, mask], axis=1))
    assert ((actions >= 5) & (actions < 9)).all()
    assert log_probs.shape == (32,)

    data = actor.export()
    policy = Policy(data)
    exported = policy.logits(obs)
    with torch.no_grad():
        expected = actor.network(torch.as_tensor(obs)).numpy()
    np.testing.assert_allclose(exported, expected, atol=1e-4)
