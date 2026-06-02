import unittest
from enum import Enum, auto

from pgeon import Predicate

from example.cartpole.discretizer import (
    CartpoleDiscretizer,
    Position,
    Velocity,
    Angle,
)


class TestCartpoleDiscretizer(unittest.TestCase):

    # -------------------------
    # HAPPY PATHS
    # -------------------------

    def test_str_to_state_handles_predicates_in_expected_order(self):
        discretizer = CartpoleDiscretizer()

        state_str = (
            "Predicate(Position.MIDDLE)&"
            "Predicate(Velocity.RIGHT)&"
            "Predicate(Angle.STANDING)"
        )

        state = discretizer.str_to_state(state_str)

        self.assertEqual(
            state,
            (
                Predicate(Position.MIDDLE),
                Predicate(Velocity.RIGHT),
                Predicate(Angle.STANDING),
            ),
        )

    def test_str_to_state_handles_unordered_predicates(self):
        discretizer = CartpoleDiscretizer()

        state_str = (
            "Predicate(Angle.STANDING)&"
            "Predicate(Position.MIDDLE)&"
            "Predicate(Velocity.RIGHT)"
        )

        state = discretizer.str_to_state(state_str)

        self.assertEqual(
            state,
            (
                Predicate(Position.MIDDLE),
                Predicate(Velocity.RIGHT),
                Predicate(Angle.STANDING),
            ),
        )

    def test_state_to_str_uses_consistent_order(self):
        discretizer = CartpoleDiscretizer()

        state = (
            Predicate(Angle.STANDING),
            Predicate(Position.MIDDLE),
            Predicate(Velocity.RIGHT),
        )

        state_str = discretizer.state_to_str(state)

        self.assertLess(state_str.index("Position"), state_str.index("Velocity"))
        self.assertLess(state_str.index("Velocity"), state_str.index("Angle"))

    def test_state_serialization_round_trip_with_unordered_state(self):
        discretizer = CartpoleDiscretizer()

        original_state = (
            Predicate(Angle.FALLING_LEFT),
            Predicate(Velocity.LEFT),
            Predicate(Position.RIGHT),
        )

        serialized = discretizer.state_to_str(original_state)
        loaded_state = discretizer.str_to_state(serialized)

        self.assertEqual(
            loaded_state,
            (
                Predicate(Position.RIGHT),
                Predicate(Velocity.LEFT),
                Predicate(Angle.FALLING_LEFT),
            ),
        )

    # -------------------------
    # UNHAPPY PATHS
    # -------------------------

    def test_str_to_state_raises_error_when_position_is_missing(self):
        discretizer = CartpoleDiscretizer()

        state_str = (
            "Predicate(Velocity.RIGHT)&"
            "Predicate(Angle.STANDING)"
        )

        with self.assertRaises(ValueError):
            discretizer.str_to_state(state_str)

    def test_str_to_state_raises_error_when_velocity_is_missing(self):
        discretizer = CartpoleDiscretizer()

        state_str = (
            "Predicate(Position.MIDDLE)&"
            "Predicate(Angle.STANDING)"
        )

        with self.assertRaises(ValueError):
            discretizer.str_to_state(state_str)

    def test_str_to_state_raises_error_when_angle_is_missing(self):
        discretizer = CartpoleDiscretizer()

        state_str = (
            "Predicate(Position.MIDDLE)&"
            "Predicate(Velocity.RIGHT)"
        )

        with self.assertRaises(ValueError):
            discretizer.str_to_state(state_str)

    def test_str_to_state_raises_error_when_predicate_type_is_duplicated(self):
        discretizer = CartpoleDiscretizer()

        state_str = (
            "Predicate(Position.LEFT)&"
            "Predicate(Position.RIGHT)&"
            "Predicate(Velocity.RIGHT)&"
            "Predicate(Angle.STANDING)"
        )

        with self.assertRaises(ValueError):
            discretizer.str_to_state(state_str)

    def test_str_to_state_raises_error_for_unknown_predicate_type(self):
        discretizer = CartpoleDiscretizer()

        state_str = (
            "Predicate(Position.MIDDLE)&"
            "Predicate(Velocity.RIGHT)&"
            "Predicate(Unknown.VALUE)"
        )

        with self.assertRaises(ValueError):
            discretizer.str_to_state(state_str)

    def test_state_to_str_raises_error_for_unknown_predicate_type(self):
        discretizer = CartpoleDiscretizer()

        class UnknownPredicateType(Enum):
            UNKNOWN = auto()

        state = (
            Predicate(Position.MIDDLE),
            Predicate(Velocity.RIGHT),
            Predicate(UnknownPredicateType.UNKNOWN),
        )

        with self.assertRaises(ValueError):
            discretizer.state_to_str(state)


if __name__ == "__main__":
    unittest.main()