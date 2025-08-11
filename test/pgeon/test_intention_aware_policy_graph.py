import unittest
from enum import Enum
from test.domain.test_env import State, TestingAgent, TestingDiscretizer, TestingEnv
from typing import List

from pgeon.desire import Desire
from pgeon.discretizer import Predicate, PredicateBasedStateRepresentation
from pgeon.intention_aware_policy_graph import IntentionAwarePolicyGraph
from pgeon.policy_representation import Action, GraphRepresentation

# action_idx_to_name = {'0': 'UP', '1': 'DOWN', '2': 'RIGHT', '3': 'LEFT', '4': 'STAY', '5': 'Interact'}
action_name_to_idx = {"Interact": "5"}


class TestIntentionAwarePolicyGraph(unittest.TestCase):
    """Tests for the IntentionAwarePolicyGraph class."""

    class TestEnum(Enum):
        DUMMY = 0

    def get_desires(self, only_one_pot=False) -> List[Desire]:
        action_name_to_idx = {"Interact": "5"}
        clause = {"HELD_PLAYER(SOUP)", "ACTION2NEAREST(SERVICE;INTERACT)"}
        action = action_name_to_idx["Interact"]
        desire_to_service = Desire("desire_to_service", action, clause)

        clause = {
            "HELD_PLAYER(ONION)",
            "ACTION2NEAREST(POT0;INTERACT)",
            "POT_STATE(POT0;PREPARING)",
        }
        action = action_name_to_idx["Interact"]
        desire_to_cook0 = Desire("desire_to_cook0", action, clause)

        clause = {
            "HELD_PLAYER(ONION)",
            "ACTION2NEAREST(POT0;INTERACT)",
            "POT_STATE(POT0;NOT_STARTED)",
        }
        action = action_name_to_idx["Interact"]
        desire_to_start_cooking0 = Desire("desire_to_start_cooking0", action, clause)

        to_return = [desire_to_service, desire_to_cook0, desire_to_start_cooking0]

        if not only_one_pot:
            clause = {
                "HELD_PLAYER(ONION)",
                "ACTION2NEAREST(POT1;INTERACT)",
                "POT_STATE(POT1;PREPARING)",
            }
            action = action_name_to_idx["Interact"]
            desire_to_cook1 = Desire("desire_to_cook1", action, clause)

            clause = {
                "HELD_PLAYER(ONION)",
                "ACTION2NEAREST(POT1;INTERACT)",
                "POT_STATE(POT1;NOT_STARTED)",
            }
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
        self.ipg = IntentionAwarePolicyGraph(
            self.discretizer, self.representation, self.env, self.agent
        )
        self.ipg.fit(n_episodes=1)

        self.state0 = PredicateBasedStateRepresentation((Predicate(State.ZERO),))
        self.state1 = PredicateBasedStateRepresentation((Predicate(State.ONE),))
        self.action0: Action = 0

        self.desire_north = Desire("north", self.action0, {Predicate(State.ONE)})
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
        desire_not_present = Desire("not_present", 1, {Predicate(self.TestEnum.DUMMY)})
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
            self.state0, self.action0, minimum_probability_of_increase=1.0
        )
        self.assertEqual(len(why_trace), 1)


if __name__ == "__main__":
    unittest.main()
