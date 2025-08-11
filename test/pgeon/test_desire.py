import unittest
from test.domain.test_env import DummyState

from pgeon.desire import Desire, Goal
from pgeon.discretizer import Action, Predicate, PredicateBasedState


class TestDesire(unittest.TestCase):
    def setUp(self):
        self.predicate1 = Predicate(DummyState.ZERO)
        self.predicate2 = Predicate(DummyState.ONE)
        self.goal = Goal("test_goal", {self.predicate1, self.predicate2})
        self.action: Action = 0
        self.state = PredicateBasedState([self.predicate1])
        self.desire = Desire("test_desire", self.action, self.state)

    def test_goal_initialization(self):
        self.assertEqual(self.goal.name, "test_goal")
        self.assertEqual(self.goal.clause, {self.predicate1, self.predicate2})

    def test_desire_initialization(self):
        self.assertEqual(self.desire.name, "test_desire")
        self.assertEqual(self.desire.action, self.action)
        self.assertEqual(self.desire.clause, self.state)
        self.assertEqual(self.desire.type, "achievement")

    def test_desire_immutability(self):
        with self.assertRaises(Exception):
            self.desire.name = "new_name"


if __name__ == "__main__":
    unittest.main()
