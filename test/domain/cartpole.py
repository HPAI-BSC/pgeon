from enum import Enum, auto
from typing import Tuple

import numpy as np

from pgeon.discretizer import Predicate, PredicateBasedState, PredicateDiscretizer


class PredicateName(Enum):
    POSITION = auto()
    VELOCITY = auto()
    ANGLE = auto()


P = PredicateName


class Direction(Enum):
    LEFT = auto()
    MIDDLE = auto()
    RIGHT = auto()


class Velocity(Enum):
    LEFT = auto()
    RIGHT = auto()


class Stability(Enum):
    STANDING = auto()
    STUCK = auto()
    FALLING = auto()
    STABILIZING = auto()


class Action(Enum):
    LEFT = 0
    RIGHT = 1


class CartpoleDiscretizer(PredicateDiscretizer):
    def __init__(self):
        super(CartpoleDiscretizer, self).__init__()

    def discretize(self, np_state: np.ndarray) -> PredicateBasedState:
        position, velocity, angle, ang_velocity = np_state

        if -2 < position < 2:
            pos_predicate = PredicateName.MIDDLE
        elif position < 0:
            pos_predicate = Predicate(P.POSITION, [Direction.LEFT])
        else:
            pos_predicate = Predicate(P.POSITION, [Direction.RIGHT])

        if velocity < 0:
            mov_predicate = Predicate(P.VELOCITY, [Velocity.LEFT])
        else:
            mov_predicate = Predicate(P.VELOCITY, [Velocity.RIGHT])

        stuck_velocity_thr = 0.1
        standing_angle_thr = 0.0005
        pole_predicate = ""
        if -standing_angle_thr < angle < standing_angle_thr:
            pole_predicate = Predicate(P.ANGLE, [Stability.STANDING])
        elif angle < 0 and -stuck_velocity_thr < ang_velocity < stuck_velocity_thr:
            pole_predicate = Predicate(P.ANGLE, [Stability.STUCK, Direction.LEFT])
        elif angle > 0 and -stuck_velocity_thr < ang_velocity < stuck_velocity_thr:
            pole_predicate = Predicate(P.ANGLE, [Stability.STUCK, Direction.RIGHT])
        elif angle < 0 and ang_velocity < 0:
            pole_predicate = Predicate(P.ANGLE, [Stability.FALLING, Direction.LEFT])
        elif angle < 0 and ang_velocity > 0:
            pole_predicate = Predicate(
                P.ANGLE, [Stability.STABILIZING, Direction.RIGHT]
            )
        elif angle > 0 and ang_velocity > 0:
            pole_predicate = Predicate(P.ANGLE, [Stability.FALLING, Direction.RIGHT])
        elif angle > 0 and ang_velocity < 0:
            pole_predicate = Predicate(P.ANGLE, [Stability.STABILIZING, Direction.LEFT])

        return (
            Predicate(pos_predicate),
            Predicate(mov_predicate),
            Predicate(pole_predicate),
        )

    def state_to_str(self, state: Tuple[Predicate, Predicate, Predicate]) -> str:
        return "&".join(str(pred) for pred in state)

    def str_to_predicate(self, name_and_arguments: dict[str, list[str]]):
        for name, arguments in name_and_arguments.items():
            if name == P.POSITION.name:
                pos_predicate = Predicate(P.POSITION, [Direction[arguments[0]]])
            elif name == P.VELOCITY.name:
                mov_predicate = Predicate(P.VELOCITY, [Velocity[arguments[0]]])
            elif name == P.ANGLE.name:
                pole_predicate = Predicate(P.ANGLE, [Stability[arguments[0]]])

        return (
            Predicate(pos_predicate),
            Predicate(mov_predicate),
            Predicate(pole_predicate),
        )

    def nearest_state(self, state):
        og_position, og_velocity, og_angle = state

        for e in Position:
            if [e] != og_position.value:
                yield Predicate(e), og_velocity, og_angle
        for e in Velocity:
            if [e] != og_velocity.value:
                yield og_position, Predicate(e), og_angle
        for e in Angle:
            if [e] != og_angle.value:
                yield og_position, og_velocity, Predicate(e)

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
                            Predicate(e),
                            Predicate(f),
                            Predicate(g),
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
