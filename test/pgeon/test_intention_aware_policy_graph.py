import unittest
from test.domain.test_env import State, TestingAgent, TestingDiscretizer, TestingEnv
from typing import List

from pgeon import Predicate
from pgeon.desire import Desire
from pgeon.discretizer import PredicateBasedStateRepresentation
from pgeon.intention_aware_policy_graph import IntentionAwarePolicyGraph
from pgeon.policy_representation import Action, GraphRepresentation

# action_idx_to_name = {'0': 'UP', '1': 'DOWN', '2': 'RIGHT', '3': 'LEFT', '4': 'STAY', '5': 'Interact'}
action_name_to_idx = {"Interact": "5"}
Action = "str"


def get_desires(only_one_pot=False) -> List[Desire]:
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
        desire_to_start_cooking1 = Desire("desire_to_start_cooking1", action, clause)

        to_return.append(desire_to_cook1)
        to_return.append(desire_to_start_cooking1)

    return to_return


class TestIntentionAwarePolicyGraph(unittest.TestCase):
    """Tests for the IntentionAwarePolicyGraph class."""

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
        self.state2 = PredicateBasedStateRepresentation((Predicate(State.TWO),))
        self.state3 = PredicateBasedStateRepresentation((Predicate(State.THREE),))
        self.action0: Action = 0
        self.action1: Action = 1

        self.desire_north = Desire("north", self.action0, {Predicate(State.ONE)})
        self.desire_south = Desire("south", self.action1, {Predicate(State.TWO)})

    def test_desire_registration(self):
        self.ipg.register_desire(self.desire_north)
        self.assertIn(self.desire_north, self.ipg.registered_desires)


if __name__ == "__main__":
    unittest.main()
