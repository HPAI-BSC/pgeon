import unittest
from enum import Enum
from test.domain.test_env import (
    DummyState,
    TestingAgent,
    TestingDiscretizer,
    TestingEnv,
)
from typing import List

from pgeon.desire import Desire, IntentionalStateMetadata
from pgeon.discretizer import Action, Predicate, PredicateBasedState
from pgeon.intention_aware_policy_approximator import (
    IntentionAwarePolicyApproximator,
    ProbQuery,
)
from pgeon.policy_representation import GraphRepresentation


class TestProbQuery(unittest.TestCase):
    def test_valid_queries(self):
        s = PredicateBasedState((Predicate(DummyState.ZERO),))
        a = 0
        s_prima = PredicateBasedState((Predicate(DummyState.ONE),))

        # Valid queries that should not raise an error
        ProbQuery(s=s)
        ProbQuery(a=a, given_s=s)
        ProbQuery(s_prima=s_prima, a=a, given_s=s)
        ProbQuery(s_prima=s_prima, given_a=a, given_s=s)
        ProbQuery(s_prima=s_prima, given_do_a=a, given_s=s)
        ProbQuery(s_prima=s_prima, given_s=s)

    def test_invalid_queries(self):
        s = PredicateBasedState((Predicate(DummyState.ZERO),))
        a = 0

        # s and given_s
        with self.assertRaises(AssertionError):
            ProbQuery(s=s, given_s=s)

        # Multiple action types
        with self.assertRaises(AssertionError):
            ProbQuery(a=a, given_a=a)
        with self.assertRaises(AssertionError):
            ProbQuery(a=a, given_do_a=a)
        with self.assertRaises(AssertionError):
            ProbQuery(given_a=a, given_do_a=a)
        with self.assertRaises(AssertionError):
            ProbQuery(a=a, given_a=a, given_do_a=a)

        # s with other parameters
        with self.assertRaises(AssertionError):
            ProbQuery(s=s, a=a)

        # given_a without s_prima
        with self.assertRaises(AssertionError):
            ProbQuery(given_a=a, given_s=s)

        # given_do_a without s_prima
        with self.assertRaises(AssertionError):
            ProbQuery(given_do_a=a, given_s=s)

        # No parameters
        with self.assertRaises(AssertionError):
            ProbQuery()


class TestIntentionAwarePolicyApproximator(unittest.TestCase):
    """Tests for the IntentionAwarePolicyApproximator class."""

    class TestEnum(Enum):
        DUMMY = 0

    def get_desires(self, only_one_pot=False) -> List[Desire]:
        action_name_to_idx = {"Interact": "5"}
        clause = {
            Predicate(DummyState.ZERO),
            Predicate(DummyState.ONE),
        }
        action = action_name_to_idx["Interact"]
        desire_to_service = Desire(
            "desire_to_service", action, PredicateBasedState(clause)
        )

        clause = {
            Predicate(DummyState.ZERO),
            Predicate(DummyState.ONE),
            Predicate(DummyState.TWO),
        }
        action = action_name_to_idx["Interact"]
        desire_to_cook0 = Desire("desire_to_cook0", action, PredicateBasedState(clause))

        clause = {
            Predicate(DummyState.ZERO),
            Predicate(DummyState.ONE),
            Predicate(DummyState.TWO),
        }
        action = action_name_to_idx["Interact"]
        desire_to_start_cooking0 = Desire(
            "desire_to_start_cooking0", action, PredicateBasedState(clause)
        )

        to_return = [desire_to_service, desire_to_cook0, desire_to_start_cooking0]

        if not only_one_pot:
            clause = {
                Predicate(DummyState.ZERO),
                Predicate(DummyState.ONE),
                Predicate(DummyState.TWO),
            }
            action = action_name_to_idx["Interact"]
            desire_to_cook1 = Desire(
                "desire_to_cook1", action, PredicateBasedState(clause)
            )

            clause = {
                Predicate(DummyState.ZERO),
                Predicate(DummyState.ONE),
                Predicate(DummyState.TWO),
            }
            action = action_name_to_idx["Interact"]
            desire_to_start_cooking1 = Desire(
                "desire_to_start_cooking1", action, PredicateBasedState(clause)
            )

            to_return.append(desire_to_cook1)
            to_return.append(desire_to_start_cooking1)

        return to_return

    def setUp(self):
        """Set up test data before each test."""
        self.env = TestingEnv()
        self.discretizer = TestingDiscretizer()
        self.representation = GraphRepresentation()
        self.agent = TestingAgent()
        self.ipg = IntentionAwarePolicyApproximator(
            self.discretizer,
            GraphRepresentation(state_metadata_class=IntentionalStateMetadata),
            self.env,
            self.agent,
        )
        self.ipg.fit(n_episodes=1)

        self.state0 = PredicateBasedState((Predicate(DummyState.ZERO),))
        self.state1 = PredicateBasedState((Predicate(DummyState.ONE),))
        self.action0: Action = 0

        self.desire_north = Desire(
            "north", self.action0, PredicateBasedState([Predicate(DummyState.ONE)])
        )
        self.desires = self.get_desires()

    def test_initialization(self):
        """Test that the graph is initialized correctly."""
        self.assertEqual(self.ipg.c_threshold, 0.5)
        self.assertFalse(self.ipg.verbose)
        self.assertIsInstance(self.ipg.policy_representation, GraphRepresentation)
        self.assertTrue(len(self.ipg.policy_representation.get_all_states()) > 0)

    def test_desire_registration(self):
        """Test that desires are registered correctly."""
        self.ipg.register_desire(self.desire_north)
        self.assertIn(self.desire_north, self.ipg.registered_desires)

        self.ipg.register_all_desires(self.desires)
        for desire in self.desires:
            self.assertIn(desire, self.ipg.registered_desires)

    def test_intention_propagation(self):
        """Test that intentions are propagated correctly."""
        self.ipg.register_desire(self.desire_north)
        intentions = self.ipg.get_intentions(self.state0)
        self.assertIn(self.desire_north, intentions)
        self.assertGreater(intentions[self.desire_north], 0)

    def test_get_intention(self):
        """Test that get_intention returns the correct value."""
        self.ipg.register_desire(self.desire_north)
        intention_value = self.ipg.get_intention(self.state0, self.desire_north)
        self.assertGreater(intention_value, 0)

        # Test for a desire that is not present
        desire_not_present = Desire(
            "not_present", 1, PredicateBasedState([Predicate(self.TestEnum.DUMMY)])
        )
        intention_value = self.ipg.get_intention(self.state0, desire_not_present)
        self.assertEqual(intention_value, 0)

    def test_check_desire(self):
        """Test that check_desire works correctly."""
        # This state satisfies the desire
        self.assertGreater(self.ipg.check_desire(self.state1, self.desire_north), 0)

        # This state does not satisfy the desire
        self.assertEqual(self.ipg.check_desire(self.state0, self.desire_north), 0)

    def test_answer_what(self):
        # No desires registered yet
        intentions = self.ipg.answer_what(self.state1)
        self.assertEqual(len(intentions), 0)

        self.ipg.register_desire(self.desire_north)
        intentions = self.ipg.answer_what(self.state1)
        self.assertEqual(len(intentions), 1)
        self.assertEqual(intentions[0][0], self.desire_north)

    def test_answer_how(self):
        # Desire is already fulfilled
        self.ipg.register_desire(self.desire_north)
        how_trace = self.ipg.answer_how(self.state1, [self.desire_north])
        self.assertIn(self.desire_north, how_trace)
        self.assertEqual(len(how_trace[self.desire_north]), 1)

        # Multi-step path
        how_trace = self.ipg.answer_how(self.state0, [self.desire_north])
        self.assertIn(self.desire_north, how_trace)
        self.assertGreaterEqual(len(how_trace[self.desire_north]), 1)

    def test_answer_why(self):
        self.ipg.register_desire(self.desire_north)
        why_trace = self.ipg.answer_why(self.state0, self.action0)
        self.assertEqual(len(why_trace), 0)

        # Test with minimum_probability_of_increase
        why_trace = self.ipg.answer_why(
            self.state0, self.action0, minimum_probability_of_increase=1.0
        )
        self.assertEqual(len(why_trace), 0)


if __name__ == "__main__":
    unittest.main()
