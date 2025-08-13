from enum import Enum, auto
from typing import Tuple

import numpy as np

from pgeon.discretizer import Discretizer, Predicate


class Position(Enum):
    LEFT = auto()
    MIDDLE = auto()
    RIGHT = auto()


class Velocity(Enum):
    LEFT = auto()
    RIGHT = auto()


class Angle(Enum):
    STANDING = auto()
    STUCK_LEFT = auto()
    STUCK_RIGHT = auto()
    FALLING_LEFT = auto()
    FALLING_RIGHT = auto()
    STABILIZING_LEFT = auto()
    STABILIZING_RIGHT = auto()


class Action(Enum):
    LEFT = 0
    RIGHT = 1


class CartpoleDiscretizer(Discretizer):
    def __init__(self):
        super(CartpoleDiscretizer, self).__init__()

    def discretize(self, state: np.ndarray) -> Tuple[Predicate, Predicate, Predicate]:
        position, velocity, angle, ang_velocity = state

        if -2 < position < 2:
            pos_predicate = Position.MIDDLE
        elif position < 0:
            pos_predicate = Position.LEFT
        else:
            pos_predicate = Position.RIGHT

        if velocity < 0:
            mov_predicate = Velocity.LEFT
        else:
            mov_predicate = Velocity.RIGHT

        stuck_velocity_thr = 0.1
        standing_angle_thr = 0.0005
        pole_predicate = ""
        if -standing_angle_thr < angle < standing_angle_thr:
            pole_predicate = Angle.STANDING
        elif angle < 0 and -stuck_velocity_thr < ang_velocity < stuck_velocity_thr:
            pole_predicate = Angle.STUCK_LEFT
        elif angle > 0 and -stuck_velocity_thr < ang_velocity < stuck_velocity_thr:
            pole_predicate = Angle.STUCK_RIGHT
        elif angle < 0 and ang_velocity < 0:
            pole_predicate = Angle.FALLING_LEFT
        elif angle < 0 and ang_velocity > 0:
            pole_predicate = Angle.STABILIZING_RIGHT
        elif angle > 0 and ang_velocity > 0:
            pole_predicate = Angle.FALLING_RIGHT
        elif angle > 0 and ang_velocity < 0:
            pole_predicate = Angle.STABILIZING_LEFT

        return (
            Predicate(Position, [pos_predicate]),
            Predicate(Velocity, [mov_predicate]),
            Predicate(Angle, [pole_predicate]),
        )

    def state_to_str(self, state: Tuple[Predicate, Predicate, Predicate]) -> str:
        return "&".join(str(pred) for pred in state)

    def str_to_state(self, state: str):
        pos, vel, angle = state.split("&")
        pos_predicate = Position[pos[:-1].split("(")[1]]
        mov_predicate = Velocity[vel[:-1].split("(")[1]]
        pole_predicate = Angle[angle[:-1].split("(")[1]]

        return (
            Predicate(Position, [pos_predicate]),
            Predicate(Velocity, [mov_predicate]),
            Predicate(Angle, [pole_predicate]),
        )

    def nearest_state(self, state):
        og_position, og_velocity, og_angle = state

        for e in Position:
            if [e] != og_position.value:
                yield Predicate(Position, [e]), og_velocity, og_angle
        for e in Velocity:
            if [e] != og_velocity.value:
                yield og_position, Predicate(Velocity, [e]), og_angle
        for e in Angle:
            if [e] != og_angle.value:
                yield og_position, og_velocity, Predicate(Angle, [e])

        for e in Position:
            for f in Velocity:
                for g in Angle:
                    amount_of_equals_to_og = (
                        int([e] == og_position.value)
                        + int([f] == og_velocity.value)
                        + int([g] == og_angle.value)
                    )
                    if amount_of_equals_to_og <= 1:
                        yield (
                            Predicate(Position, [e]),
                            Predicate(Velocity, [f]),
                            Predicate(Angle, [g]),
                        )

    def all_actions(self):
        return [Action.LEFT, Action.RIGHT]

    def get_predicate_space(self):
        all_tuples = []
        for p in Position:
            for v in Velocity:
                for a in Angle:
                    all_tuples.append((p, v, a))
        return all_tuples
