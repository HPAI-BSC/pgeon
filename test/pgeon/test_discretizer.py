import unittest
from enum import Enum, auto

from pydantic import ValidationError

from pgeon.discretizer import (
    Predicate,
    PredicateBasedState,
    State,
    Transition,
)


class PredicateName(Enum):
    IS_DAYTIME = auto()
    HAS_COLOR = auto()
    HAS_SHAPE = auto()
    IMPLIES = auto()


P = PredicateName


class Object(Enum):
    BALL = auto()
    PYRAMID = auto()


# Define some enums for testing
class Color(Enum):
    RED = auto()
    GREEN = auto()


class Shape(Enum):
    ANY_SHAPE = auto()
    SQUARE = auto()
    CIRCLE = auto()


class TestPredicate(unittest.TestCase):
    def test_init_without_arguments(self):
        p = Predicate(P.IS_DAYTIME)
        self.assertEqual(p.name, P.IS_DAYTIME)
        self.assertEqual(p.arguments, ())

    def test_init_with_arguments(self):
        p = Predicate(
            P.HAS_COLOR,
            (
                Object.BALL,
                Color.RED,
            ),
        )
        self.assertEqual(p.name, P.HAS_COLOR)
        self.assertEqual(p.arguments, (Object.BALL, Color.RED))

    def test_eq(self):
        p1 = Predicate(P.HAS_COLOR, (Object.BALL, Color.RED))
        p2 = Predicate(P.HAS_COLOR, (Object.BALL, Color.RED))
        p3 = Predicate(P.HAS_COLOR, (Object.BALL, Color.GREEN))
        p4 = Predicate(P.IMPLIES, (Object.BALL, Color.RED))

        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
        self.assertNotEqual(p1, p4)
        self.assertNotEqual(p1, "not a predicate")

    def test_str_without_arguments(self):
        p = Predicate(P.IS_DAYTIME)
        self.assertEqual(str(p), "IS_DAYTIME()")

    def test_str_with_arguments(self):
        p = Predicate(P.HAS_COLOR, (Object.BALL, Color.RED))
        self.assertEqual(str(p), "HAS_COLOR(BALL;RED)")

    def test_repr(self):
        p = Predicate(P.HAS_COLOR, (Object.BALL, Color.RED))
        self.assertEqual(repr(p), "HAS_COLOR(BALL;RED)")

    def test_hash(self):
        p1 = Predicate(P.HAS_COLOR, (Object.BALL, Color.RED))
        p2 = Predicate(P.HAS_COLOR, (Object.BALL, Color.RED))
        p3 = Predicate(P.HAS_COLOR, (Object.BALL, Color.GREEN))
        s = {p1}

        self.assertEqual(hash(p1), hash(p2))
        self.assertNotEqual(hash(p1), hash(p3))
        self.assertIn(p1, s)
        self.assertIn(p2, s)
        self.assertNotIn(p3, s)

    def test_lt(self):
        # alphabetical order
        p1 = Predicate(P.HAS_COLOR, (Object.BALL, Color.GREEN))
        p2 = Predicate(P.HAS_COLOR, (Object.BALL, Color.RED))
        p3 = Predicate(Shape.SQUARE)

        self.assertTrue(p1 < p2)
        self.assertTrue(p1 < p3)
        self.assertTrue(p2 < p3)

        with self.assertRaises(ValueError):
            p1 < "not a predicate"


class TestTransition(unittest.TestCase):
    def test_transition_creation(self):
        """Test that a Transition object can be created with default values."""
        transition = Transition(action=0)
        self.assertEqual(transition.action, 0)
        self.assertEqual(transition.probability, 0.0)
        self.assertEqual(transition.frequency, 0)

    def test_transition_creation_with_values(self):
        """Test that a Transition object can be created with specified values."""
        transition = Transition(action=1, probability=0.5, frequency=10)
        self.assertEqual(transition.action, 1)
        self.assertEqual(transition.probability, 0.5)
        self.assertEqual(transition.frequency, 10)


class TestPredicateBasedStateRepresentation(unittest.TestCase):
    def setUp(self):
        self.ball_has_color_red = Predicate(P.HAS_COLOR, (Object.BALL, Color.RED))
        self.ball_has_shape_square = Predicate(P.HAS_COLOR, (Object.BALL, Shape.SQUARE))
        self.ball_has_shape_circle = Predicate(P.HAS_SHAPE, (Object.BALL, Shape.CIRCLE))
        self.is_daytime = Predicate(P.IS_DAYTIME)

    def test_init(self):
        s = PredicateBasedState((self.ball_has_color_red, self.ball_has_shape_square))
        self.assertEqual(
            s.predicates,
            frozenset((self.ball_has_color_red, self.ball_has_shape_square)),
        )

    def test_eq_with_same_type(self):
        s1 = PredicateBasedState((self.ball_has_color_red, self.ball_has_shape_square))
        s2 = PredicateBasedState((self.ball_has_color_red, self.ball_has_shape_square))
        s3 = PredicateBasedState((self.ball_has_shape_square, self.ball_has_color_red))
        s4 = PredicateBasedState((self.ball_has_color_red,))

        self.assertEqual(s1, s2)
        self.assertEqual(s1, s3)
        self.assertNotEqual(s1, s4)
        self.assertNotEqual(s1, "not a state representation")

        # Test against a different StateRepresentation implementation
        class OtherStateRep(State):
            def __init__(self, predicates):
                self.predicates = predicates

            def __eq__(self, other):
                return False

            def __hash__(self):
                return 0

        other_rep = OtherStateRep((self.ball_has_color_red, self.ball_has_shape_square))
        self.assertNotEqual(s1, other_rep)
        self.assertNotEqual(other_rep, s1)

    def test_eq_with_tuple(self):
        s1 = PredicateBasedState(
            predicates=(self.ball_has_color_red, self.ball_has_shape_square)
        )

        self.assertEqual(s1, (self.ball_has_color_red, self.ball_has_shape_square))
        self.assertEqual(s1, (self.ball_has_shape_square, self.ball_has_color_red))
        self.assertNotEqual(s1, (self.ball_has_color_red,))
        self.assertNotEqual(
            s1,
            (
                self.ball_has_color_red,
                self.ball_has_shape_square,
                self.ball_has_shape_circle,
            ),
        )

    def test_hash(self):
        s1 = PredicateBasedState(
            predicates=[self.ball_has_color_red, self.ball_has_shape_square]
        )
        s2 = PredicateBasedState(
            predicates=(self.ball_has_color_red, self.ball_has_shape_square)
        )
        s3 = PredicateBasedState((self.ball_has_shape_square, self.ball_has_color_red))
        s4 = PredicateBasedState((self.ball_has_color_red,))

        state_set = {s1}
        self.assertEqual(hash(s1), hash(s2))
        self.assertEqual(hash(s1), hash(s3))
        self.assertNotEqual(hash(s1), hash(s4))
        self.assertIn(s2, state_set)
        self.assertIn(s3, state_set)
        self.assertNotIn(s4, state_set)

    def test_frozen(self):
        s = PredicateBasedState(predicates=[self.ball_has_color_red])
        with self.assertRaises(ValidationError):
            s.predicates = frozenset(
                [self.ball_has_color_red, self.ball_has_shape_square]
            )


if __name__ == "__main__":
    unittest.main()
