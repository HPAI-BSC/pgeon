import unittest
from test.domain.test_env import State, TestingAgent, TestingDiscretizer, TestingEnv
from typing import List

import networkx as nx

from pgeon import GraphRepresentation, Predicate
from pgeon.desire import Desire
from pgeon.discretizer import PredicateBasedStateRepresentation
from pgeon.intention_aware_policy_graph import IPG
from pgeon.policy_representation import Action

# action_idx_to_name = {'0': 'UP', '1': 'DOWN', '2': 'RIGHT', '3': 'LEFT', '4': 'STAY', '5': 'Interact'}
action_name_to_idx = {"Interact": "5"}


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
        self.discretizer = TestingDiscretizer()

        # Create states and actions for testing
        self.state0 = PredicateBasedStateRepresentation(
            (Predicate(State, [State.ZERO]),)
        )
        self.state1 = PredicateBasedStateRepresentation(
            (Predicate(State, [State.ONE]),)
        )
        self.state2 = PredicateBasedStateRepresentation(
            (Predicate(State, [State.TWO]),)
        )
        self.state3 = PredicateBasedStateRepresentation(
            (Predicate(State, [State.THREE]),)
        )

        self.action0: Action = 0
        self.representation = GraphRepresentation()
        self.graph = IPG(
            self.discretizer,
            self.representation,
            TestingEnv(),
            TestingAgent(),
        )

    def test_initialization(self):
        """Test initialization of policy representation."""
        # self.assertEqual(self.graph, self.representation.graph.nx_graph)
        self.assertIsInstance(self.graph.graph, nx.MultiDiGraph)
        self.assertEqual(len(self.graph.graph.nodes), 0)
        self.assertEqual(len(self.graph.graph.edges), 0)

    def test_propogate_intentions(self):
        # load graph from edges and nodes files
        desires = [
            Desire(
                "test_desire",
                self.action0,
                {Predicate(State, [State.ONE])},
            )
        ]
        # this automatically calls propogate_intentions() on the nodes
        self.graph.register_all_desires(desires)

        # validate the intentions of several nodes have the expected values
        self.assertEqual(self.graph.graph.nodes[self.state0].intention, 0)
        self.assertEqual(self.graph.graph.nodes[self.state1].intention, 1)


if __name__ == "__main__":
    unittest.main()
