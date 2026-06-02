from enum import Enum, auto
from typing import Tuple

import numpy as np

from pgeon import Discretizer, Predicate


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
    PREDICATE_TYPES = (Position, Velocity, Angle)

    PREDICATE_ORDER = {
        Position: 0,
        Velocity: 1,
        Angle: 2,
    }

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
        else:
            raise ValueError(
                f"Could not discretize pole state: angle={angle}, "
                f"ang_velocity={ang_velocity}"
            )

        return (
            Predicate(pos_predicate),
            Predicate(mov_predicate),
            Predicate(pole_predicate),
        )

    def state_to_str(self, state: Tuple[Predicate, Predicate, Predicate]) -> str:
        """
        Serialize states using a deterministic and unambiguous predicate order.

        The default Predicate.__str__ representation may produce strings such as
        RIGHT(), which are ambiguous because both Position and Velocity contain
        RIGHT. Therefore, this method explicitly stores the enum type name.
        """
        ordered_state = sorted(
            state,
            key=lambda predicate: self._predicate_sort_key(predicate),
        )

        return "&".join(
            self._predicate_to_str(predicate)
            for predicate in ordered_state
        )

    def str_to_state(self, state: str) -> Tuple[Predicate, Predicate, Predicate]:
        """
        Deserialize a state string independently of predicate order.

        This method does not assume that the string is saved as:
            Position & Velocity & Angle

        Instead, each serialized predicate is identified by its enum type.
        """
        parsed_predicates = {}

        for raw_predicate in state.split("&"):
            predicate = self._str_to_predicate(raw_predicate)

            enum_value = self._predicate_enum_value(predicate)
            enum_type = type(enum_value)

            if enum_type in parsed_predicates:
                raise ValueError(
                    f"Duplicated predicate type {enum_type.__name__} "
                    f"in serialized state: {state}"
                )

            parsed_predicates[enum_type] = predicate

        missing_types = [
            enum_type.__name__
            for enum_type in self.PREDICATE_TYPES
            if enum_type not in parsed_predicates
        ]

        if missing_types:
            raise ValueError(
                f"Serialized state is missing predicates of type "
                f"{missing_types}: {state}"
            )

        return (
            parsed_predicates[Position],
            parsed_predicates[Velocity],
            parsed_predicates[Angle],
        )

    def _predicate_to_str(self, predicate: Predicate) -> str:
        enum_value = self._predicate_enum_value(predicate)
        enum_type = type(enum_value)

        return f"Predicate({enum_type.__name__}.{enum_value.name})"

    def _predicate_sort_key(self, predicate: Predicate) -> int:
        enum_value = self._predicate_enum_value(predicate)
        enum_type = type(enum_value)

        if enum_type not in self.PREDICATE_ORDER:
            raise ValueError(f"Unknown predicate type: {enum_type}")

        return self.PREDICATE_ORDER[enum_type]

    def _predicate_enum_value(self, predicate: Predicate) -> Enum:
        """
        Extract the enum value stored inside a Predicate.

        In the current pgeon Predicate implementation, the enum value is stored
        in predicate.name, not in predicate.value.
        """
        value = predicate.name

        if not isinstance(value, Enum):
            raise ValueError(f"Predicate does not contain an enum name: {predicate}")

        return value

    def _str_to_predicate(self, raw_predicate: str) -> Predicate:
        """
        Parse one serialized predicate.

        The parser identifies predicates by enum type and enum member name, so it
        does not depend on the order in which predicates appear in the state
        string.
        """
        raw_predicate = raw_predicate.strip()

        for enum_type in self.PREDICATE_TYPES:
            for enum_value in enum_type:
                enum_type_name = enum_type.__name__
                enum_member_name = enum_value.name

                valid_patterns = [
                    f"Predicate({enum_type_name}.{enum_member_name})",
                    f"{enum_type_name}.{enum_member_name}",
                    f"{enum_type_name}({enum_member_name})",
                    f"{enum_type_name}: {enum_member_name}",
                ]

                if raw_predicate in valid_patterns:
                    return Predicate(enum_value)

        raise ValueError(f"Could not parse predicate from string: {raw_predicate}")

    def nearest_state(self, state):
        og_position, og_velocity, og_angle = state

        for e in Position:
            if Predicate(e) != og_position:
                yield Predicate(e), og_velocity, og_angle

        for e in Velocity:
            if Predicate(e) != og_velocity:
                yield og_position, Predicate(e), og_angle

        for e in Angle:
            if Predicate(e) != og_angle:
                yield og_position, og_velocity, Predicate(e)

        for e in Position:
            for f in Velocity:
                for g in Angle:
                    amount_of_equals_to_og = (
                        int(Predicate(e) == og_position)
                        + int(Predicate(f) == og_velocity)
                        + int(Predicate(g) == og_angle)
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