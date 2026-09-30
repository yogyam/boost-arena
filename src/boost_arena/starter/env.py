"""The training environment, built from RLGym parts so that it follows the Boost Arena rules.

Everything the bot sees and does here is what the benchmark gives it later: the same 53
numbers, the same 90 actions and mask, the same timing, boost and car body.
"""

import math
import os
from typing import Any, Dict, List

import numpy as np
from rlgym.api import ActionParser, DoneCondition, ObsBuilder, RewardFunction, StateMutator
from rlgym.rocket_league import common_values
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.state_mutators import KickoffMutator

from .. import interface
from ..interface import BLUE, ORANGE, BallInfo, CarInfo
from ..tasks import CAR_REST_HEIGHT, BALL_REST_HEIGHT, TASKS
from .curriculum import ALL_REWARDS, ALL_SITUATIONS, ProgressReader

TICK_SKIP = interface.TICK_SKIP
ACTION_DELAY = interface.ACTION_DELAY

# What the policy receives: the observation, then the action mask
POLICY_INPUT_SIZE = interface.OBS_SIZE + interface.NUM_ACTIONS

FIELD_HALF_WIDTH = 4096
FIELD_HALF_LENGTH = 5120
GOAL_HALF_WIDTH = 893


def _car_info(car, prev_action) -> CarInfo:
    physics = car.physics
    on_ground = car.on_ground
    # The rulebook's "on its roof" test uses the contact surface, which RLGym does not expose.
    # A car that is upside down and low is treated the same way.
    turtled = physics.up[2] < -0.5 and physics.position[2] < 60
    return CarInfo(
        team=int(car.team_num),
        pos=np.asarray(physics.position, dtype=np.float32),
        vel=np.asarray(physics.linear_velocity, dtype=np.float32),
        ang_vel=np.asarray(physics.angular_velocity, dtype=np.float32),
        forward=np.asarray(physics.forward, dtype=np.float32),
        up=np.asarray(physics.up, dtype=np.float32),
        boost=float(car.boost_amount),
        is_on_ground=bool(on_ground),
        has_flip_or_jump=bool(on_ground or car.has_flip),
        is_demoed=bool(car.is_demoed),
        has_world_contact=bool(turtled),
        world_contact_normal=np.array([0, 0, 1], dtype=np.float32) if turtled else np.zeros(3, dtype=np.float32),
        prev_action=np.asarray(prev_action, dtype=np.float32),
    )


def _ball_info(state: GameState) -> BallInfo:
    return BallInfo(
        pos=np.asarray(state.ball.position, dtype=np.float32),
        vel=np.asarray(state.ball.linear_velocity, dtype=np.float32),
        ang_vel=np.asarray(state.ball.angular_velocity, dtype=np.float32),
    )


class ArenaObs(ObsBuilder):
    """The 53 rulebook numbers, followed by the 90 action-mask flags for the policy to apply."""

    def get_obs_space(self, agent):
        return "real", POLICY_INPUT_SIZE

    def reset(self, agents, initial_state, shared_info):
        shared_info["prev_actions"] = {agent: np.zeros(8, dtype=np.float32) for agent in agents}

    def build_obs(self, agents, state, shared_info):
        prev_actions = shared_info.get("prev_actions", {})
        ball = _ball_info(state)
        ids = list(state.cars)
        cars = [_car_info(state.cars[i], prev_actions.get(i, np.zeros(8, dtype=np.float32))) for i in ids]
        out = {}
        for agent in agents:
            index = ids.index(agent)
            obs = interface.build_observation(ball, cars, index)
            mask = interface.action_mask(cars[index]).astype(np.float32)
            out[agent] = np.concatenate([obs, mask]).astype(np.float32)
        return out


class ArenaAction(ActionParser):
    """The 90-action table with the rulebook's timing: 7 ticks of the old controls, then 1 of the new."""

    def get_action_space(self, agent):
        return "discrete", interface.NUM_ACTIONS

    def reset(self, agents, initial_state, shared_info):
        shared_info["prev_actions"] = {agent: np.zeros(8, dtype=np.float32) for agent in agents}

    def parse_actions(self, actions, state, shared_info):
        prev_actions = shared_info.setdefault("prev_actions", {})
        engine_actions = {}
        for agent, action in actions.items():
            index = int(np.asarray(action).reshape(-1)[0])
            new = interface.ACTION_TABLE[index]
            old = prev_actions.get(agent, np.zeros(8, dtype=np.float32))
            engine_actions[agent] = np.vstack([np.tile(old, (ACTION_DELAY, 1)), np.tile(new, (TICK_SKIP - ACTION_DELAY, 1))]).astype(np.float32)
            prev_actions[agent] = new.copy()
        return engine_actions


class _RecordingGame:
    """Stands in for the benchmark's Game so that a task's setup can be reused to place cars and ball."""

    def __init__(self):
        self.ball = None
        self.cars = {}

    def reset(self):
        pass

    def set_ball(self, pos, vel=(0, 0, 0), ang_vel=(0, 0, 0)):
        self.ball = (np.asarray(pos, dtype=np.float32), np.asarray(vel, dtype=np.float32), np.asarray(ang_vel, dtype=np.float32))

    def set_car(self, index, pos, yaw=None, rot_mat=None, vel=(0, 0, 0), ang_vel=(0, 0, 0), boost=100.0):
        if rot_mat is not None:
            yaw = math.atan2(rot_mat.forward.y, rot_mat.forward.x)
        self.cars[index] = (np.asarray(pos, dtype=np.float32), float(yaw or 0.0), np.asarray(vel, dtype=np.float32))


class SituationMutator(StateMutator):
    """Sets up the situations of the current curriculum phase, drawn in proportion to their weights.

    Task situations are the benchmark's own, so the bot practises what it will be scored on,
    from different random draws. The orange car, which a task may leave out, starts in front
    of its own goal.
    """

    def __init__(self, progress: ProgressReader = None, rng: np.random.Generator = None):
        self.progress = progress or ProgressReader()
        self.rng = rng or np.random.default_rng()
        self.kickoff = KickoffMutator()

    def apply(self, state: GameState, shared_info):
        phase = self.progress.phase
        names = list(phase.situations)
        weights = np.array([phase.situations[n] for n in names], dtype=np.float64)
        name = names[self.rng.choice(len(names), p=weights / weights.sum())]
        shared_info["situation"] = name

        blue = [car for car in state.cars.values() if car.is_blue]
        orange = [car for car in state.cars.values() if car.is_orange]
        for car in state.cars.values():
            car.hitbox_type = common_values.PLANK
            car.boost_amount = 100.0
            car.demo_respawn_timer = 0.0
        state.config.boost_consumption = 1.0 / common_values.BOOST_CONSUMPTION_RATE   # 1 boost per second, as in the rules

        if name == "kickoff":
            self.kickoff.apply(state, shared_info)
            for car in state.cars.values():
                car.boost_amount = 100.0
            return
        if name == "random":
            self._random(state)
            return

        recorder = _RecordingGame()
        TASKS[name].setup(recorder, self.rng)
        pos, vel, ang_vel = recorder.ball
        state.ball.position = pos.copy()
        state.ball.linear_velocity = vel.copy()
        state.ball.angular_velocity = ang_vel.copy()

        for index, cars in ((0, blue), (1, orange)):
            if not cars:
                continue
            if index in recorder.cars:
                pos, yaw, vel = recorder.cars[index]
            else:
                # The task has no opponent, so the orange car waits in front of its goal
                pos = np.array([self.rng.uniform(-600, 600), FIELD_HALF_LENGTH - self.rng.uniform(200, 600), CAR_REST_HEIGHT], dtype=np.float32)
                yaw = -math.pi / 2
                vel = np.zeros(3, dtype=np.float32)
            self._place(cars[0], pos, yaw, vel)

    def _random(self, state: GameState):
        """Ball anywhere with some speed, cars anywhere on the ground, facing anywhere."""
        state.ball.position = np.array([
            self.rng.uniform(-FIELD_HALF_WIDTH + 400, FIELD_HALF_WIDTH - 400),
            self.rng.uniform(-FIELD_HALF_LENGTH + 500, FIELD_HALF_LENGTH - 500),
            self.rng.uniform(BALL_REST_HEIGHT, 1200),
        ], dtype=np.float32)
        speed = self.rng.uniform(0, 1500)
        heading = self.rng.uniform(-math.pi, math.pi)
        state.ball.linear_velocity = np.array([speed * math.cos(heading), speed * math.sin(heading), self.rng.uniform(-300, 600)], dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        for car in state.cars.values():
            pos = np.array([
                self.rng.uniform(-FIELD_HALF_WIDTH + 400, FIELD_HALF_WIDTH - 400),
                self.rng.uniform(-FIELD_HALF_LENGTH + 500, FIELD_HALF_LENGTH - 500),
                CAR_REST_HEIGHT,
            ], dtype=np.float32)
            self._place(car, pos, self.rng.uniform(-math.pi, math.pi), np.zeros(3, dtype=np.float32))

    @staticmethod
    def _place(car, pos, yaw, vel):
        car.physics.position = np.asarray(pos, dtype=np.float32).copy()
        car.physics.linear_velocity = np.asarray(vel, dtype=np.float32).copy()
        car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
        car.physics.euler_angles = np.array([0, yaw, 0], dtype=np.float32)
        car.boost_amount = 100.0


class CurriculumReward(RewardFunction):
    """The weighted sum of a few simple rewards, with the weights of the current phase."""

    def __init__(self, progress: ProgressReader = None):
        self.progress = progress or ProgressReader()
        self._last_ball_speed = 0.0

    def reset(self, agents, initial_state, shared_info):
        self._last_ball_speed = float(np.linalg.norm(initial_state.ball.linear_velocity))
        self.weights = self.progress.phase.rewards

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        ball = state.ball
        ball_speed = float(np.linalg.norm(ball.linear_velocity))
        scorer = state.scoring_team if state.goal_scored else None
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            physics = car.physics
            sign = 1.0 if car.is_blue else -1.0          # Blue attacks +Y
            goal_y = sign * FIELD_HALF_LENGTH

            parts = {}
            to_ball = ball.position - physics.position
            distance = float(np.linalg.norm(to_ball)) or 1.0
            direction = to_ball / distance
            touched = car.ball_touches > 0

            parts["touch"] = 1.0 if touched else 0.0
            parts["strong_touch"] = max(0.0, (ball_speed - self._last_ball_speed) / 1500.0) if touched else 0.0
            parts["player_to_ball"] = float(np.dot(physics.linear_velocity, direction)) / common_values.CAR_MAX_SPEED
            parts["face_ball"] = float(np.dot(physics.forward, direction))
            parts["air"] = 0.0 if car.on_ground else (1.0 if physics.position[2] > 500 else 0.3)
            parts["speed"] = float(np.linalg.norm(physics.linear_velocity)) / common_values.CAR_MAX_SPEED
            parts["air_touch"] = 1.0 if touched and not car.on_ground and ball.position[2] > 500 else 0.0

            # Ball velocity towards the goal the car attacks, minus the same for the other goal
            to_goal = np.array([0 - ball.position[0], goal_y - ball.position[1], 0.0])
            to_goal /= float(np.linalg.norm(to_goal)) or 1.0
            parts["ball_to_goal"] = float(np.dot(ball.linear_velocity, to_goal)) / common_values.BALL_MAX_SPEED
            if scorer is None:
                parts["goal"] = 0.0
            else:
                scored = (scorer == common_values.BLUE_TEAM) == car.is_blue
                parts["goal"] = (1.0 + ball_speed / common_values.BALL_MAX_SPEED) * (1.0 if scored else -1.0)

            rewards[agent] = float(sum(self.weights.get(name, 0.0) * parts[name] for name in ALL_REWARDS))
        self._last_ball_speed = ball_speed
        return rewards


class GoalCondition(DoneCondition):
    def reset(self, agents, initial_state, shared_info):
        pass

    def is_done(self, agents, state, shared_info):
        return {agent: state.goal_scored for agent in agents}


class TimeoutCondition(DoneCondition):
    """Ends the episode after `seconds`, or after `no_touch_seconds` without anyone touching the ball."""

    def __init__(self, seconds=30.0, no_touch_seconds=15.0):
        self.seconds = seconds
        self.no_touch_seconds = no_touch_seconds

    def reset(self, agents, initial_state, shared_info):
        self._start = initial_state.tick_count
        self._last_touch = initial_state.tick_count

    def is_done(self, agents, state, shared_info):
        if any(car.ball_touches > 0 for car in state.cars.values()):
            self._last_touch = state.tick_count
        elapsed = (state.tick_count - self._start) / common_values.TICKS_PER_SECOND
        since_touch = (state.tick_count - self._last_touch) / common_values.TICKS_PER_SECOND
        done = elapsed >= self.seconds or since_touch >= self.no_touch_seconds
        return {agent: done for agent in agents}


def build_env():
    """One 1v1 training environment. Used by every environment process."""
    from rlgym.api import RLGym
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence

    progress = ProgressReader()
    return RLGym(
        state_mutator=MutatorSequence(FixedTeamSizeMutator(blue_size=1, orange_size=1), SituationMutator(progress)),
        obs_builder=ArenaObs(),
        action_parser=ArenaAction(),
        reward_fn=CurriculumReward(progress),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(),
        transition_engine=RocketSimEngine(rlbot_delay=False),   # The rulebook's delay is applied by ArenaAction
    )
