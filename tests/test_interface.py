"""Checks the interface against vectors recorded from the training environment it was defined from."""

import gzip
import json
import os

import numpy as np
import pytest

from boost_arena import interface
from boost_arena.interface import BallInfo, CarInfo

VECTORS_PATH = os.path.join(os.path.dirname(__file__), "data", "interface_vectors.json.gz")


@pytest.fixture(scope="module")
def vectors():
    with gzip.open(VECTORS_PATH, "rt", encoding="utf-8") as f:
        return json.load(f)


def car_from_json(j):
    return CarInfo(
        team=j["team"],
        pos=np.array(j["pos"], dtype=np.float32),
        vel=np.array(j["vel"], dtype=np.float32),
        ang_vel=np.array(j["ang_vel"], dtype=np.float32),
        forward=np.array(j["forward"], dtype=np.float32),
        up=np.array(j["up"], dtype=np.float32),
        boost=j["boost"],
        is_on_ground=j["is_on_ground"],
        has_flip_or_jump=j["has_flip_or_jump"],
        is_demoed=j["is_demoed"],
        has_world_contact=j["has_world_contact"],
        world_contact_normal=np.array(j["world_contact_normal"], dtype=np.float32),
        prev_action=np.array(j["prev_action"], dtype=np.float32),
    )


def ball_from_json(j):
    return BallInfo(
        pos=np.array(j["pos"], dtype=np.float32),
        vel=np.array(j["vel"], dtype=np.float32),
        ang_vel=np.array(j["ang_vel"], dtype=np.float32),
    )


def test_timing(vectors):
    assert vectors["tick_skip"] == interface.TICK_SKIP
    assert vectors["action_delay"] == interface.ACTION_DELAY


def test_action_table(vectors):
    recorded = np.array(vectors["actions"], dtype=np.float32)
    assert recorded.shape == (interface.NUM_ACTIONS, 8)
    np.testing.assert_array_equal(interface.ACTION_TABLE, recorded)


def test_observations(vectors):
    assert len(vectors["samples"]) > 500
    for sample in vectors["samples"]:
        cars = [car_from_json(j) for j in sample["players"]]
        obs = interface.build_observation(ball_from_json(sample["ball"]), cars, sample["player_index"])
        np.testing.assert_allclose(obs, np.array(sample["obs"], dtype=np.float32), rtol=0, atol=1e-6)


def test_action_masks(vectors):
    seen_ground = seen_air = False
    for sample in vectors["samples"]:
        car = car_from_json(sample["players"][sample["player_index"]])
        seen_ground |= car.is_on_ground
        seen_air |= not car.is_on_ground
        mask = interface.action_mask(car)
        np.testing.assert_array_equal(mask, np.array(sample["action_mask"], dtype=bool))
    assert seen_ground and seen_air


def test_every_state_has_an_action(vectors):
    for sample in vectors["samples"]:
        car = car_from_json(sample["players"][sample["player_index"]])
        assert interface.action_mask(car).any()


def test_no_opponent_gives_zeros():
    obs = interface.build_observation(BallInfo(), [CarInfo()], 0)
    assert obs.shape == (interface.OBS_SIZE,)
    assert not obs[-18:].any()
