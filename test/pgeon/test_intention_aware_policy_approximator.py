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
from pgeon.discretizer import Action, Predicate, PredicateBasedState, Transition
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


class TestIntentionPropagation(unittest.TestCase):
    def setUp(self):
        self.env = TestingEnv()
        self.discretizer = TestingDiscretizer()
        self.agent = TestingAgent()
        # States
        self.s0 = PredicateBasedState((Predicate(DummyState.ZERO),))
        self.s1 = PredicateBasedState((Predicate(DummyState.ONE),))
        self.s2 = PredicateBasedState((Predicate(DummyState.TWO),))

        # Graph representation with explicit probabilities
        self.rep = GraphRepresentation(state_metadata_class=IntentionalStateMetadata)
        self.rep.states[self.s0] = IntentionalStateMetadata(probability=0.2)
        self.rep.states[self.s1] = IntentionalStateMetadata(probability=0.3)
        self.rep.states[self.s2] = IntentionalStateMetadata(probability=0.5)
        # Transitions (action 0)
        self.rep.transitions[self.s0][self.s1] = Transition(action=0, probability=0.7)
        self.rep.transitions[self.s0][self.s2] = Transition(action=0, probability=0.3)
        # Self-loop at s1 (fulfilling action)
        self.rep.transitions[self.s1][self.s1] = Transition(action=0, probability=1.0)

        self.ipg = IntentionAwarePolicyApproximator(
            self.discretizer,
            self.rep,
            self.env,
            self.agent,
        )
        self.desire = Desire(
            "reach_one", 0, PredicateBasedState([Predicate(DummyState.ONE)])
        )
        self.ipg.register_desire(self.desire, stop_criterion=1e-6)

    def test_propagation_values(self):
        # Intention seeded at s1 with P(a|s1)=1 due to self-loop; propagation to s0
        # multiplies by P(s1|s0)=0.7. s2 is not on any path to fulfillment.
        self.assertAlmostEqual(self.ipg.get_intention(self.s1, self.desire), 1.0)
        self.assertAlmostEqual(self.ipg.get_intention(self.s0, self.desire), 0.7)
        self.assertAlmostEqual(self.ipg.get_intention(self.s2, self.desire), 0.0)

    def test_desire_statistics(self):
        # Only s1 satisfies the desire clause and its fulfilling action has prob 1.0
        action_probs, nodes = self.ipg.compute_desire_statistics(self.desire)
        self.assertEqual(nodes, [self.s1])
        self.assertEqual(action_probs, [1.0])

    def test_commitment_and_intention_metrics(self):
        # With threshold 0, both s1 (1.0) and s0 (0.7) are counted as committed.
        intentions, nodes_with_intent = self.ipg.compute_commitment_stats(
            self.desire.name, commitment_threshold=0.0
        )
        mapping = {n: i for n, i in zip(nodes_with_intent, intentions)}
        self.assertAlmostEqual(mapping.get(self.s1, 0.0), 1.0)
        self.assertAlmostEqual(mapping.get(self.s0, 0.0), 0.7)

        attrib_probs, expected = self.ipg.compute_intention_metrics(c_threshold=0.5)
        # At c_threshold=0.5, both s1 (1.0) and s0 (0.7) still qualify;
        # attributed probability is P(s1)+P(s0)=0.3+0.2=0.5 and
        # expected intention is (1*0.3 + 0.7*0.2)/0.5 = 0.88. "Any" matches here.
        self.assertIn("reach_one", attrib_probs)
        self.assertIn("reach_one", expected)
        self.assertAlmostEqual(attrib_probs["reach_one"], 0.5)
        self.assertAlmostEqual(expected["reach_one"], 0.88)
        self.assertAlmostEqual(attrib_probs["Any"], 0.5)
        self.assertAlmostEqual(expected["Any"], 0.88)

    def test_how_stochastic_branches(self):
        # From s0 there are two branches for action 0: to s1 with 0.7 and to s2 with 0.3.
        # We expect those exact branch probabilities in descending order.
        how_stoch = self.ipg.answer_how_stochastic(
            self.s0, [self.desire], min_branch_probability=0.0
        )
        branches = how_stoch[self.desire]
        probs = sorted(
            [round(p, 6) for _, sp, _, p in branches if sp in (self.s1, self.s2)],
            reverse=True,
        )
        self.assertEqual(probs, [0.7, 0.3])


if __name__ == "__main__":
    unittest.main()
