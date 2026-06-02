from enum import Enum, auto
from typing import Tuple
import numpy as np

from pgeon import Discretizer, Predicate


# ---
# Clases por Variable
# ---

# - X -
class XPos(Enum):
    LEFT_POS = auto()
    CENTER_POS = auto()
    RIGHT_POS = auto()


# - Y -
class YPos(Enum):
    LOW_ALTITUDE = auto()
    DESCENT_ZONE = auto()
    HIGH_ALTITUDE = auto()


# - X Velocity -
class XVel(Enum):
    MOVE_LEFT = auto()
    NO_HORIZONTAL_MOVEMENT = auto()
    MOVE_RIGHT = auto()


# - Y Velocity -
class YVel(Enum):
    DESCENDING = auto()
    HOVERING = auto()
    ASCENDING = auto()


# - Angle -
class Angle(Enum):
    TILTED_LEFT = auto()
    UPRIGHT = auto()
    TILTED_RIGHT = auto()


# - Angular Velocity -
class AngularVel(Enum):
    ROTATING_LEFT = auto()
    NO_ROTATION = auto()
    ROTATING_RIGHT = auto()


# --- CONTACTOS ---
class Contact(Enum):
    NO_CONTACT = auto()
    LEFT_LEG = auto()
    RIGHT_LEG = auto()
    BOTH_LEGS = auto()


# ---
# Discretizador
# ---

class LunarLanderSemanticDiscretizer(Discretizer):
    def __init__(self):
        super().__init__()

        self.low = np.array([-2.5, -2.5, -10, -10, -2*np.pi, -10])
        self.high = np.array([ 2.5,  2.5,  10,  10,  2*np.pi,  10])

        # nº de estados por variable (todas = 3)
        self.n_states = [3, 3, 3, 3, 3, 3]

        # crear bordes automáticamente
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

    # --- Métodos obligatorios para la clase abstracta ---
    def nearest_state(self, state):
        """
        Devuelve el estado más cercano en el espacio de predicados.
        Para este caso, simplemente devolvemos el mismo estado.
        """
        return state

    def str_to_state(self, state_str):
        """
        Convierte un string de estado de vuelta a la tupla de Predicates.
        Asume que state_to_str usa '&' como separador.
        """
        return tuple(state_str.split("&"))

    def _map_value(self, value, edges, mapping, low, high):
        value = np.clip(value, low, high)
        idx = np.digitize(value, edges[1:-1])  # devuelve 0,1,2
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

    def discretize(
        self, obs: np.ndarray
    ) -> Tuple[Predicate, Predicate, Predicate, Predicate, Predicate, Predicate, Predicate, Predicate]:

        x, y, x_vel, y_vel, angle, ang_vel, left_leg, right_leg = obs

        contact = self.map_contact(left_leg, right_leg)

        return (
            Predicate(self._map_value(x,       self.edges[0], self.maps[0], self.low[0], self.high[0])),
            Predicate(self._map_value(y,       self.edges[1], self.maps[1], self.low[1], self.high[1])),
            Predicate(self._map_value(x_vel,   self.edges[2], self.maps[2], self.low[2], self.high[2])),
            Predicate(self._map_value(y_vel,   self.edges[3], self.maps[3], self.low[3], self.high[3])),
            Predicate(self._map_value(angle,   self.edges[4], self.maps[4], self.low[4], self.high[4])),
            Predicate(self._map_value(ang_vel, self.edges[5], self.maps[5], self.low[5], self.high[5])),
            Predicate(contact),
        )

    def state_to_str(self, state):
        return "&".join(str(pred) for pred in state)

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
                                    states.append((xp, yp, xv, yv, a, av, c))
        return states