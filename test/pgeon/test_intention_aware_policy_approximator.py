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
        # Use empty PredicateBasedState for test purposes
        clause = PredicateBasedState([])
        action = action_name_to_idx["Interact"]
        desire_to_service = Desire("desire_to_service", action, clause)

        clause = PredicateBasedState([])
        action = action_name_to_idx["Interact"]
        desire_to_cook0 = Desire("desire_to_cook0", action, clause)

        clause = PredicateBasedState([])
        action = action_name_to_idx["Interact"]
        desire_to_start_cooking0 = Desire("desire_to_start_cooking0", action, clause)

        to_return = [desire_to_service, desire_to_cook0, desire_to_start_cooking0]

        if not only_one_pot:
            clause = PredicateBasedState([])
            action = action_name_to_idx["Interact"]
            desire_to_cook1 = Desire("desire_to_cook1", action, clause)

            clause = PredicateBasedState([])
            action = action_name_to_idx["Interact"]
            desire_to_start_cooking1 = Desire(
                "desire_to_start_cooking1", action, clause
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
        self.assertTrue(len(list(self.ipg.policy_representation.states)) > 0)

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
        self.assertTrue(self.ipg.check_desire(self.state1, self.desire_north))

        # This state does not satisfy the desire
        self.assertFalse(self.ipg.check_desire(self.state0, self.desire_north))

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
        self.assertEqual(len(why_trace), 1)

        # Test with minimum_probability_of_increase
        why_trace = self.ipg.answer_why(
            self.state0, self.action0, minimum_probability_of_increase=0.5
        )
        self.assertEqual(len(why_trace), 1)

    def test_compute_desire_and_commitment_stats(self):
        # Register multiple desires
        for d in self.desires:
            self.ipg.register_desire(d)
        # Ensure some stats can be computed without error
        for d in self.desires:
            action_probs, nodes = self.ipg.compute_desire_statistics(d)
            self.assertIsInstance(action_probs, list)
            self.assertIsInstance(nodes, list)
            self.assertEqual(len(action_probs), len(nodes))
            # Probabilities are in [0,1]
            for p in action_probs:
                self.assertGreaterEqual(p, 0)
                self.assertLessEqual(p, 1)

            intentions, nodes_with_intent = self.ipg.compute_commitment_stats(
                d.name, commitment_threshold=0.0
            )
            self.assertEqual(len(intentions), len(nodes_with_intent))
            for i in intentions:
                self.assertGreaterEqual(i, 0)

    def test_compute_intention_metrics(self):
        for d in self.desires:
            self.ipg.register_desire(d)
        attrib_probs, expected = self.ipg.compute_intention_metrics(c_threshold=0.0)
        # Contains entries for each desire and 'Any'
        for d in self.desires:
            self.assertIn(d.name, attrib_probs)
            self.assertIn(d.name, expected)
            self.assertGreaterEqual(attrib_probs[d.name], 0)
            self.assertGreaterEqual(expected[d.name], 0)
        self.assertIn("Any", attrib_probs)
        self.assertIn("Any", expected)
        self.assertGreaterEqual(attrib_probs["Any"], 0)
        self.assertGreaterEqual(expected["Any"], 0)

    def test_answer_how_stochastic(self):
        for d in self.desires:
            self.ipg.register_desire(d)
        res = self.ipg.answer_how_stochastic(
            self.state0, self.desires, min_branch_probability=0.0
        )
        # Should provide an entry per desire
        self.assertEqual(set(res.keys()), set(self.desires))
        for d, branches in res.items():
            # Should be a list of tuples (a, s', I, prob)
            self.assertIsInstance(branches, list)
            for item in branches:
                self.assertEqual(len(item), 4)
                a, s_prime, i, p = item
                self.assertIsInstance(p, float)
                self.assertGreaterEqual(p, 0)


if __name__ == "__main__":
    unittest.main()
