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
# A demolished car is out for the rest of the episode. The simulator would bring it back at a
# spawn point it picks at random, which cannot be seeded, so no two runs would be the same.
RESPAWN_DELAY = 1e9

_initialized = False
_pristine_arenas = {}


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
        try:
            rs.init(_collision_mesh_folder())
        except RuntimeError as e:
            if "inited" not in str(e).lower():
                raise   # Something else, like a missing folder
            # Another part of the program, such as an RLGym engine, loaded the arena already
        _initialized = True


def _vec(v):
    return np.array([v.x, v.y, v.z], dtype=np.float32)


def _rs_vec(values):
    return rs.Vec(float(values[0]), float(values[1]), float(values[2]))


def rot_mat_from_yaw(yaw: float) -> rs.RotMat:
    """The orientation of a car standing on its wheels and facing `yaw` radians from the +X axis."""
    return rs.Angle(float(yaw), 0.0, 0.0).as_rot_mat()


_ball_only_arena = None


def ball_path(pos, vel, seconds: float, ang_vel=(0, 0, 0)):
    """Where an untouched ball goes: its position at every tick, as an array of shape (ticks, 3)."""
    global _ball_only_arena
    init()
    if _ball_only_arena is None:
        _ball_only_arena = rs.Arena(rs.GameMode.SOCCAR)   # Making an arena is slow, so one is kept
    arena = _ball_only_arena

    state = rs.BallState()
    state.pos = _rs_vec(pos)
    state.vel = _rs_vec(vel)
    state.ang_vel = _rs_vec(ang_vel)
    arena.ball.set_state(state)

    path = np.empty((int(seconds * interface.TICK_RATE), 3), dtype=np.float32)
    for tick in range(len(path)):
        arena.step(1)
        path[tick] = _vec(arena.ball.get_state().pos)
    return path


class Game:
    """One arena with a blue car and, optionally, an orange car."""

    def __init__(self, with_opponent: bool = True):
        init()
        self._with_opponent = with_opponent

        # An arena that is never played in. Every episode is played in a fresh copy of it, so
        # nothing an episode leaves behind, down to the state of the suspension, can affect the
        # next one. Resetting an arena in place is faster but does not clear everything.
        if with_opponent not in _pristine_arenas:
            _pristine_arenas[with_opponent] = self._new_arena(with_opponent)
        self._pristine = _pristine_arenas[with_opponent]

        self.arena = None
        self._goal_line = None
        self.reset()

    @staticmethod
    def _apply_rules(arena):
        mutators = arena.get_mutator_config()
        mutators.boost_used_per_second = BOOST_USED_PER_SECOND
        mutators.car_spawn_boost_amount = CAR_SPAWN_BOOST
        mutators.respawn_delay = RESPAWN_DELAY
        arena.set_mutator_config(mutators)
        return mutators

    @staticmethod
    def _new_arena(with_opponent):
        arena = rs.Arena(rs.GameMode.SOCCAR)
        Game._apply_rules(arena)

        arena.add_car(rs.Team.BLUE, CAR_BODY)
        if with_opponent:
            arena.add_car(rs.Team.ORANGE, CAR_BODY)
        return arena

    # ---- Setting up a situation ----

    def reset(self):
        """Starts from a clean arena. The ball and cars must then be placed."""
        self.arena = self._pristine.clone()
        # A copy comes with the simulator's default settings, not the original's
        mutators = self._apply_rules(self.arena)
        self._goal_line = mutators.goal_base_threshold_y + mutators.ball_radius

        # Blue first, then orange
        self.cars = sorted(self.arena.get_cars(), key=lambda car: int(car.team))
        assert [int(car.team) for car in self.cars] == [BLUE, ORANGE][: len(self.cars)]

        self.prev_actions = np.zeros((len(self.cars), 8), dtype=np.float32)
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

    def step(self, action_indices, scripted=None):
        """Advances one decision step.

        `action_indices` holds one index into the action table per car. `scripted` maps a car's
        index to simulator controls, for cars driven by a fixed program instead of a model;
        their entry in `action_indices` is ignored. Scripted cars get the same action delay.
        """
        if len(action_indices) != len(self.cars):
            raise ValueError("Expected one action per car")
        scripted = scripted or {}

        # The cars keep doing what they were doing until the new decision takes effect
        self.arena.step(interface.ACTION_DELAY)

        for i, car in enumerate(self.cars):
            if i in scripted:
                car.set_controls(scripted[i])
                continue

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
