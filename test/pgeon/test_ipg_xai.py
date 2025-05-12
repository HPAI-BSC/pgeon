import unittest
from test.domain.test_env import State, TestingAgent, TestingDiscretizer, TestingEnv

from pgeon import GraphRepresentation, Predicate
from pgeon.desire import Desire
from pgeon.discretizer import PredicateBasedStateRepresentation
from pgeon.intention_aware_policy_graph import IPG
from pgeon.ipg_xai import IPG_XAI_analyser

Action = str


class MyTestCase(unittest.TestCase):
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

        self.action0: Action = "a0"
        self.action1: Action = "a1"
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

        self.ipg_xai = IPG_XAI_analyser(self.ipg)
        self.desire0 = Desire(
            "test_desire",
            self.action0,
            {Predicate(State, [State.ONE])},
        )
        num_places = 4
        self.ipg.register_all_desires([self.desire0], stop_criterion=10 ^ -num_places)

    def test_how(self):
        trace = self.ipg_xai.answer_how(self.state0, [self.desire0], c_threshold=0.1)
        expected_trace = {
            self.desire0: [
                [self.action1, self.state1, 0.521],
                [self.action0, None, None],
            ]
        }

        self.assertEqual(expected_trace.keys(), trace.keys())
        for ed_trace, d_trace in zip(expected_trace.values(), trace.values()):
            for ev, v in zip(ed_trace, d_trace):
                e_act, e_st, e_int = ev
                act, st, intention = v
                self.assertEqual(e_act, act)
                self.assertEqual(e_st, st)
                self.assertAlmostEqual(e_int, intention, places=3)


if __name__ == "__main__":
    unittest.main()
