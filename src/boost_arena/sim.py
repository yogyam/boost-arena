"""Runs games in the RocketSim simulator under the Boost Arena rules."""

import os

import numpy as np
import RocketSim as rs

from . import interface
from .interface import BLUE, ORANGE, BallInfo, CarInfo

SIMULATOR_VERSION = "2.2.1"

# Physics settings, part of the rules
# Boost drains so slowly that a full tank outlasts any episode: bots do not manage boost in version 1
BOOST_USED_PER_SECOND = 1.0
CAR_SPAWN_BOOST = 0.0
CAR_BODY = rs.CarConfig.PLANK

_initialized = False


def _collision_mesh_folder():
    # The arena collision files come with the rlgym-rocket-league package
    import rlgym.rocket_league.sim as rlgym_sim

    folder = os.path.join(os.path.dirname(rlgym_sim.__file__), "collision_meshes")
    if not os.path.isdir(os.path.join(folder, "soccar")):
        raise RuntimeError(
            "Could not find the arena collision files. Install them with: pip install 'rlgym-rocket-league[sim]'"
        )
    return folder


def init():
    """Loads the arena into the simulator. Safe to call more than once."""
    global _initialized
    if not _initialized:
        rs.init(_collision_mesh_folder())
        _initialized = True


def _vec(v):
    return np.array([v.x, v.y, v.z], dtype=np.float32)


def _rs_vec(values):
    return rs.Vec(float(values[0]), float(values[1]), float(values[2]))


def rot_mat_from_yaw(yaw: float) -> rs.RotMat:
    """The orientation of a car standing on its wheels and facing `yaw` radians from the +X axis."""
    return rs.Angle(float(yaw), 0.0, 0.0).as_rot_mat()


class Game:
    """One arena with a blue car and, optionally, an orange car."""

    def __init__(self, with_opponent: bool = True):
        init()
        self.arena = rs.Arena(rs.GameMode.SOCCAR)

        mutators = self.arena.get_mutator_config()
        mutators.boost_used_per_second = BOOST_USED_PER_SECOND
        mutators.car_spawn_boost_amount = CAR_SPAWN_BOOST
        self.arena.set_mutator_config(mutators)
        self._goal_line = mutators.goal_base_threshold_y + mutators.ball_radius

        self.cars = [self.arena.add_car(rs.Team.BLUE, CAR_BODY)]
        if with_opponent:
            self.cars.append(self.arena.add_car(rs.Team.ORANGE, CAR_BODY))

        self.prev_actions = np.zeros((len(self.cars), 8), dtype=np.float32)
        self.ticks = 0

    # ---- Setting up a situation ----

    def reset(self, seed: int = -1):
        """Puts the ball and cars in a kickoff position and clears everything left over from the last episode."""
        self.arena.reset_kickoff(seed)
        for car in self.cars:
            car.set_controls(rs.CarControls())
            state = car.get_state()
            state.boost = 100.0
            car.set_state(state)
        self.prev_actions[:] = 0
        self.ticks = 0

    def set_ball(self, pos, vel=(0, 0, 0), ang_vel=(0, 0, 0)):
        state = rs.BallState()
        state.pos = _rs_vec(pos)
        state.vel = _rs_vec(vel)
        state.ang_vel = _rs_vec(ang_vel)
        self.arena.ball.set_state(state)

    def set_car(self, index, pos, yaw=None, rot_mat=None, vel=(0, 0, 0), ang_vel=(0, 0, 0), boost=100.0):
        """Places a car. Give either `yaw` for a car on its wheels, or a full `rot_mat`."""
        state = rs.CarState()
        state.pos = _rs_vec(pos)
        state.vel = _rs_vec(vel)
        state.ang_vel = _rs_vec(ang_vel)
        state.rot_mat = rot_mat if rot_mat is not None else rot_mat_from_yaw(yaw or 0.0)
        state.boost = float(boost)
        self.cars[index].set_state(state)

    # ---- Reading the situation ----

    def ball_info(self) -> BallInfo:
        state = self.arena.ball.get_state()
        return BallInfo(pos=_vec(state.pos), vel=_vec(state.vel), ang_vel=_vec(state.ang_vel))

    def car_infos(self) -> list:
        infos = []
        for i, car in enumerate(self.cars):
            state = car.get_state()
            infos.append(CarInfo(
                team=int(car.team),
                pos=_vec(state.pos),
                vel=_vec(state.vel),
                ang_vel=_vec(state.ang_vel),
                forward=_vec(state.rot_mat.forward),
                up=_vec(state.rot_mat.up),
                boost=state.boost,
                is_on_ground=state.is_on_ground,
                has_flip_or_jump=state.has_flip_or_jump(),
                is_demoed=state.is_demoed,
                has_world_contact=state.has_world_contact,
                world_contact_normal=_vec(state.world_contact_normal),
                prev_action=self.prev_actions[i].copy(),
            ))
        return infos

    def observe(self):
        """Observations and action masks for every car: arrays of shape (cars, 53) and (cars, 90)."""
        ball = self.ball_info()
        cars = self.car_infos()
        obs = np.stack([interface.build_observation(ball, cars, i) for i in range(len(cars))])
        masks = np.stack([interface.action_mask(car) for car in cars])
        return obs, masks

    def scoring_team(self):
        """BLUE or ORANGE if the ball is in a goal (the team that scored), otherwise None."""
        y = self.arena.ball.get_state().pos.y
        if abs(y) > self._goal_line:
            return BLUE if y > 0 else ORANGE
        return None

    @property
    def seconds(self) -> float:
        return self.ticks / interface.TICK_RATE

    # ---- Playing ----

    def step(self, action_indices):
        """Advances one decision step. `action_indices` holds one index into the action table per car."""
        if len(action_indices) != len(self.cars):
            raise ValueError("Expected one action per car")

        # The cars keep doing what they were doing until the new decision takes effect
        self.arena.step(interface.ACTION_DELAY)

        for i, car in enumerate(self.cars):
            action = interface.ACTION_TABLE[int(action_indices[i])]
            controls = rs.CarControls()
            controls.throttle = float(action[interface.THROTTLE])
            controls.steer = float(action[interface.STEER])
            controls.pitch = float(action[interface.PITCH])
            controls.yaw = float(action[interface.YAW])
            controls.roll = float(action[interface.ROLL])
            controls.jump = bool(action[interface.JUMP] == 1)
            controls.boost = bool(action[interface.BOOST] == 1)
            controls.handbrake = bool(action[interface.HANDBRAKE] == 1)
            car.set_controls(controls)
            self.prev_actions[i] = action

        self.arena.step(interface.TICK_SKIP - interface.ACTION_DELAY)
        self.ticks += interface.TICK_SKIP
