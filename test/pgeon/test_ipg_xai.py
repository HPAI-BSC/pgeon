import unittest
from test.domain.test_env import State, TestingAgent, TestingDiscretizer, TestingEnv

from pgeon.desire import Desire
from pgeon.discretizer import PredicateBasedStateRepresentation
from pgeon.intention_aware_policy_graph import IntentionAwarePolicyGraph
from pgeon.ipg_xai import IPG_XAI_analyser
from pgeon.policy_representation import Action, GraphRepresentation


class TestIPGXAI(unittest.TestCase):
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

        self.state0 = PredicateBasedStateRepresentation(
            (Predicate(State, [State.ZERO]),)
        )
        self.state1 = PredicateBasedStateRepresentation(
            (Predicate(State, [State.ONE]),)
        )
        self.action0: Action = 0

        self.desire_north = Desire(
            "north", self.action0, {Predicate(State, [State.ONE])}
        )

        self.analyser = IPG_XAI_analyser(self.ipg)


if __name__ == "__main__":
    unittest.main()
