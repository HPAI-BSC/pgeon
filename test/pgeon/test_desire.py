import unittest
from enum import Enum
from test.domain.test_env import DummyState

from pgeon.desire import Desire, Goal, IntentionalStateMetadata
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


class TestIntentionalStateMetadata(unittest.TestCase):
    class TestPredicateEnum(Enum):
        HELD_PLAYER_SOUP = "HELD_PLAYER(SOUP)"
        ACTION2NEAREST_SERVICE_INTERACT = "ACTION2NEAREST(SERVICE;INTERACT)"

    def test_default_initialization(self):
        metadata = IntentionalStateMetadata()
        self.assertEqual(metadata.intention, {})
        self.assertEqual(metadata.frequency, 0)
        self.assertEqual(metadata.probability, 0)

    def test_initialization_with_values(self):
        metadata = IntentionalStateMetadata(
            frequency=1,
            probability=0.5,
            intention={
                Desire(
                    "desire_to_service",
                    "5",
                    PredicateBasedState(
                        [
                            Predicate(self.TestPredicateEnum.HELD_PLAYER_SOUP),
                            Predicate(
                                self.TestPredicateEnum.ACTION2NEAREST_SERVICE_INTERACT
                            ),
                        ]
                    ),
                ): 0.5
            },
        )
        self.assertEqual(metadata.frequency, 1)
        self.assertEqual(metadata.probability, 0.5)
        self.assertEqual(
            metadata.intention,
            {
                Desire(
                    "desire_to_service",
                    "5",
                    PredicateBasedState(
                        [
                            Predicate(self.TestPredicateEnum.HELD_PLAYER_SOUP),
                            Predicate(
                                self.TestPredicateEnum.ACTION2NEAREST_SERVICE_INTERACT
                            ),
                        ]
                    ),
                ): 0.5
            },
        )

    def test_model_dump(self):
        # Create a simple desire with an empty PredicateBasedState
        desire = Desire("desire_to_service", 5, PredicateBasedState([]))
        metadata = IntentionalStateMetadata(
            frequency=1,
            probability=0.5,
            intention={desire: 0.5},
        )
        self.assertEqual(
            metadata.model_dump(),
            {
                "frequency": 1,
                "probability": 0.5,
                "intention": {desire: 0.5},
            },
        )

    def test_model_validate(self):
        desire = Desire(
            "desire_to_service",
            "5",
            PredicateBasedState(
                [
                    Predicate(self.TestPredicateEnum.HELD_PLAYER_SOUP),
                    Predicate(self.TestPredicateEnum.ACTION2NEAREST_SERVICE_INTERACT),
                ]
            ),
        )
        metadata = IntentionalStateMetadata.model_validate(
            {
                "frequency": 1,
                "probability": 0.5,
                "intention": {desire: 0.5},
            }
        )
        self.assertEqual(metadata.frequency, 1)
        self.assertEqual(metadata.probability, 0.5)
        self.assertEqual(
            metadata.intention,
            {desire: 0.5},
        )


if __name__ == "__main__":
    unittest.main()
