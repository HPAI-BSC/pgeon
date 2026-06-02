from enum import Enum, auto
from typing import Tuple

import numpy as np

from pgeon import Discretizer, Predicate


# ---
# Classes by variable
# ---

class XPos(Enum):
    LEFT_POS = auto()
    CENTER_POS = auto()
    RIGHT_POS = auto()


class YPos(Enum):
    LOW_ALTITUDE = auto()
    DESCENT_ZONE = auto()
    HIGH_ALTITUDE = auto()


class XVel(Enum):
    MOVE_LEFT = auto()
    NO_HORIZONTAL_MOVEMENT = auto()
    MOVE_RIGHT = auto()


class YVel(Enum):
    DESCENDING = auto()
    HOVERING = auto()
    ASCENDING = auto()


class Angle(Enum):
    TILTED_LEFT = auto()
    UPRIGHT = auto()
    TILTED_RIGHT = auto()


class AngularVel(Enum):
    ROTATING_LEFT = auto()
    NO_ROTATION = auto()
    ROTATING_RIGHT = auto()


class Contact(Enum):
    NO_CONTACT = auto()
    LEFT_LEG = auto()
    RIGHT_LEG = auto()
    BOTH_LEGS = auto()


class LunarLanderSemanticSortedDiscretizer(Discretizer):
    PREDICATE_TYPES = (
        XPos,
        YPos,
        XVel,
        YVel,
        Angle,
        AngularVel,
        Contact,
    )

    def __init__(self):
        super().__init__()

        self.low = np.array([-2.5, -2.5, -10, -10, -2 * np.pi, -10])
        self.high = np.array([2.5, 2.5, 10, 10, 2 * np.pi, 10])

        self.n_states = [3, 3, 3, 3, 3, 3]

        self.edges = [
            np.linspace(self.low[i], self.high[i], self.n_states[i] + 1)
            for i in range(6)
        ]

        self.maps = [
            list(XPos),
            list(YPos),
            list(XVel),
            list(YVel),
            list(Angle),
            list(AngularVel),
        ]

    def discretize(
        self, obs: np.ndarray
    ) -> Tuple[
        Predicate,
        Predicate,
        Predicate,
        Predicate,
        Predicate,
        Predicate,
        Predicate,
    ]:
        x, y, x_vel, y_vel, angle, ang_vel, left_leg, right_leg = obs

        contact = self.map_contact(left_leg, right_leg)

        return (
            Predicate(
                self._map_value(
                    x,
                    self.edges[0],
                    self.maps[0],
                    self.low[0],
                    self.high[0],
                )
            ),
            Predicate(
                self._map_value(
                    y,
                    self.edges[1],
                    self.maps[1],
                    self.low[1],
                    self.high[1],
                )
            ),
            Predicate(
                self._map_value(
                    x_vel,
                    self.edges[2],
                    self.maps[2],
                    self.low[2],
                    self.high[2],
                )
            ),
            Predicate(
                self._map_value(
                    y_vel,
                    self.edges[3],
                    self.maps[3],
                    self.low[3],
                    self.high[3],
                )
            ),
            Predicate(
                self._map_value(
                    angle,
                    self.edges[4],
                    self.maps[4],
                    self.low[4],
                    self.high[4],
                )
            ),
            Predicate(
                self._map_value(
                    ang_vel,
                    self.edges[5],
                    self.maps[5],
                    self.low[5],
                    self.high[5],
                )
            ),
            Predicate(contact),
        )

    def state_to_str(self, state) -> str:
        return super().state_to_str(state)

    def str_to_state(self, state_str):
        return super().str_to_state(state_str)

    def nearest_state(self, state):
        """
        Return the nearest symbolic states.

        For now, this discretizer only returns the same state. If the library
        expects an iterator, yielding is safer than returning the state directly.
        """
        yield state

    def all_actions(self):
        return [0, 1, 2, 3]

    def get_predicate_space(self):
        states = []

        for xp in XPos:
            for yp in YPos:
                for xv in XVel:
                    for yv in YVel:
                        for a in Angle:
                            for av in AngularVel:
                                for c in Contact:
                                    states.append(
                                        (
                                            Predicate(xp),
                                            Predicate(yp),
                                            Predicate(xv),
                                            Predicate(yv),
                                            Predicate(a),
                                            Predicate(av),
                                            Predicate(c),
                                        )
                                    )

        return states

    def _map_value(self, value, edges, mapping, low, high):
        value = np.clip(value, low, high)
        idx = np.digitize(value, edges[1:-1])
        return mapping[idx]

    def map_contact(self, left_leg, right_leg):
        left = int(left_leg) == 1
        right = int(right_leg) == 1

        if left and right:
            return Contact.BOTH_LEGS
        elif left:
            return Contact.LEFT_LEG
        elif right:
            return Contact.RIGHT_LEG
        else:
            return Contact.NO_CONTACT