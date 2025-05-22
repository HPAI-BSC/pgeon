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
        self.states = [self.state0, self.state1, self.state2, self.state3]

        self.action0: Action = "0"
        self.action1: Action = "1"
        self.representation = GraphRepresentation()

        self.representation.graph.add_nodes_from(self.states)

        # adding 3 edges: idea is state2 is terminal, state0 goes to 1 or 2 depending on actions & probs
        # Desire is to use action0 in state1, bringing you univocally to state2
        self.edges = [
            # - 4-tuples (u, v, k, d) for an edge with data and key k; as per policy_rep key is action
            (
                self.states[2],
                self.states[2],
                self.action0,
                {"action": self.action0, "prob": 0.75},
            ),
            (
                self.states[2],
                self.states[2],
                self.action1,
                {"action": self.action1, "prob": 0.25},
            ),
            (
                self.states[0],
                self.states[2],
                self.action0,
                {"action": self.action0, "prob": 0.5},
            ),
            (
                self.states[0],
                self.states[1],
                self.action1,
                {"action": self.action1, "prob": 0.4},
            ),
            (
                self.states[0],
                self.states[2],
                self.action1,
                {"action": self.action1, "prob": 0.1},
            ),
            (
                self.states[1],
                self.states[2],
                self.action0,
                {"action": self.action0, "prob": 0.5},
            ),
            (
                self.states[1],
                self.states[0],
                self.action1,
                {"action": self.action1, "prob": 0.1},
            ),
            (
                self.states[1],
                self.states[2],
                self.action1,
                {"action": self.action1, "prob": 0.4},
            ),
        ]
        # Expected intention return in 0 = sum_i (0.4*0.1)^i[going to 1 and back] *(0.4*0.5) [final transition]
        #  = 0.208333 according to wolframalpha, may be smaller depending on stop_criterion
        self.representation.graph.add_edges_from(self.edges)

        self.ipg = IPG(
            self.discretizer,
            self.representation,
            TestingEnv(),
            TestingAgent(),
        )
        self.ipg.graph = (
            self.representation.graph.nx_graph
        )  # TODO: Temporary line, to remove once the PG class works

    def test_initialization(self):
        """Test initialization of policy representation."""
        # self.assertEqual(self.graph, self.representation.graph.nx_graph)
        self.assertIsInstance(self.ipg.graph, nx.MultiDiGraph)
        self.assertEqual(len(self.ipg.graph.nodes), 4)
        self.assertEqual(len(self.ipg.graph.edges), 8)

    def test_propagate_intentions(self):
        # load graph from edges and nodes files
        desires = [
            Desire(
                "test_desire",
                self.action0,
                {Predicate(State, [State.ONE])},
            )
        ]
        # this automatically calls propogate_intentions() on the nodes
        num_places = 4
        self.ipg.register_all_desires(desires, stop_criterion=10 ^ -num_places)

        # validate the intentions of several nodes have the expected values
        # Id(s0) = 0.208333 according to wolframalpha, may be smaller depending on stop_criterion
        int_s0 = self.ipg.graph.nodes[self.state0]["intention"][desires[0]]
        self.assertAlmostEqual(0.208333, int_s0, places=num_places)


if __name__ == "__main__":
    unittest.main()
