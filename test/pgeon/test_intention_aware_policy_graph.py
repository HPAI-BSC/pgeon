import unittest
from test.domain.test_env import State, TestingAgent, TestingDiscretizer, TestingEnv

import networkx as nx

from pgeon import GraphRepresentation, Predicate
from pgeon.discretizer import PredicateBasedStateRepresentation
from pgeon.intention_aware_policy_graph import IPG
from pgeon.policy_representation import Action


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
        pass


if __name__ == "__main__":
    unittest.main()
