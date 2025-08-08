import unittest
from enum import Enum

from pgeon.discretizer import (
    Predicate,
    PredicateBasedStateRepresentation,
    StateRepresentation,
    Transition,
)


# Define some enums for testing
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class Shape(Enum):
    SQUARE = 1
    CIRCLE = 2
    TRIANGLE = 3


class TestPredicate(unittest.TestCase):
    def test_init(self):
        p = Predicate(Color.RED)
        self.assertEqual(p.predicate, Color)
        self.assertEqual(p.value, Color.RED)

    def test_eq(self):
        p1 = Predicate(Color.RED)
        p2 = Predicate(Color.RED)
        p3 = Predicate(Color.GREEN)
        p4 = Predicate(Shape.SQUARE)

        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
        self.assertNotEqual(p1, p4)
        self.assertNotEqual(p1, "not a predicate")

    def test_str(self):
        p = Predicate(Color.RED)
        self.assertEqual(str(p), "Color(RED)")

    def test_repr(self):
        p = Predicate(Color.RED)
        self.assertEqual(repr(p), "Color(RED)")

    def test_hash(self):
        p1 = Predicate(Color.RED)
        p2 = Predicate(Color.RED)
        p3 = Predicate(Color.GREEN)
        s = {p1}

        self.assertEqual(hash(p1), hash(p2))
        self.assertNotEqual(hash(p1), hash(p3))
        self.assertIn(p2, s)
        self.assertNotIn(p3, s)

    def test_lt(self):
        p1 = Predicate(Color.RED)
        p2 = Predicate(Shape.SQUARE)

        # The result of this comparison depends on the hash values,
        # which can change between python sessions.
        # So I will just check that it does not raise an error, and one is less than the other or vice-versa.
        self.assertNotEqual(p1 < p2, p2 < p1)

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
        self.p1 = Predicate(Color.RED)
        self.p2 = Predicate(Shape.SQUARE)
        self.predicates = (self.p1, self.p2)

    def test_init(self):
        s = PredicateBasedStateRepresentation(predicates=self.predicates)
        self.assertEqual(list(s.predicates), list(self.predicates))

    def test_eq_with_same_type(self):
        s1 = PredicateBasedStateRepresentation(predicates=self.predicates)
        s2 = PredicateBasedStateRepresentation(predicates=self.predicates)
        s3 = PredicateBasedStateRepresentation(predicates=(self.p2, self.p1))
        s4 = PredicateBasedStateRepresentation(predicates=(self.p1,))

        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, s3)
        self.assertNotEqual(s1, s4)
        self.assertNotEqual(s1, "not a state representation")

        # Test against a different StateRepresentation implementation
        class OtherStateRep(StateRepresentation):
            def __init__(self, predicates):
                self.predicates = predicates

            def __eq__(self, other):
                return False

            def __hash__(self):
                return 0

        other_rep = OtherStateRep(predicates=self.predicates)
        self.assertNotEqual(s1, other_rep)

    def test_eq_with_tuple(self):
        s1 = PredicateBasedStateRepresentation(predicates=self.predicates)

        self.assertEqual(s1, self.predicates)
        self.assertNotEqual(s1, (self.p2, self.p1))
        self.assertNotEqual(s1, (self.p1,))
        self.assertNotEqual(s1, (self.p1, self.p2, self.p1))

    def test_hash(self):
        s1 = PredicateBasedStateRepresentation(predicates=[self.p1, self.p2])
        s2 = PredicateBasedStateRepresentation(predicates=(self.p1, self.p2))
        s3 = PredicateBasedStateRepresentation(predicates=(self.p2, self.p1))

        state_set = {s1}
        self.assertEqual(hash(s1), hash(s2))
        self.assertNotEqual(hash(s1), hash(s3))
        self.assertIn(s2, state_set)
        self.assertNotIn(s3, state_set)

    def test_frozen(self):
        s = PredicateBasedStateRepresentation(predicates=[self.p1])
        with self.assertRaises(AttributeError):
            s.predicates = []


if __name__ == "__main__":
    unittest.main()
