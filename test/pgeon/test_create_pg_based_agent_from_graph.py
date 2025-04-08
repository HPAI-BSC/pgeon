import unittest
from test.domain.cartpole import CartpoleDiscretizer

import gymnasium
import numpy as np

from pgeon import (
    PGBasedPolicy,
    PGBasedPolicyMode,
    PGBasedPolicyNodeNotFoundMode,
    PolicyGraph,
)


class TestCreatePGBasedAgentFromGraph(unittest.TestCase):
    env = gymnasium.make("CartPole-v1")
    discretizer = CartpoleDiscretizer()
    pg_cartpole = PolicyGraph.from_nodes_and_edges(
        "./test/data/cartpole_nodes_small.csv",
        "./test/data/cartpole_edges_small.csv",
        env,
        discretizer,
    )

    def test_initialize(self):
        policy = PGBasedPolicy(self.pg_cartpole, PGBasedPolicyMode.GREEDY)

        self.assertEqual(policy.pg, self.pg_cartpole)
        self.assertEqual(policy.mode, PGBasedPolicyMode.GREEDY)
        self.assertEqual(
            policy.node_not_found_mode, PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM
        )
        self.assertEqual(policy.all_possible_actions, {0, 1})

    def test_act(self):
        policy = PGBasedPolicy(self.pg_cartpole, PGBasedPolicyMode.GREEDY)
        # Translates to MIDDLE, LEFT, STABILIZING_RIGHT
        action = policy.act(np.array([0.3, -0.5, -0.1, 0.7]))

        self.assertEqual(action, 1)

    def test_act_greedy_node_not_found_random_uniform(self):
        policy = PGBasedPolicy(self.pg_cartpole, PGBasedPolicyMode.GREEDY)
        # Translates to LEFT, LEFT, STABILIZING_RIGHT
        action = policy.act(np.array([-2.3, -0.5, -0.1, 0.7]))

        self.assertEqual(action, 0)

    def test_act_greedy_node_not_found_nearest_state(self):
        policy = PGBasedPolicy(
            self.pg_cartpole,
            PGBasedPolicyMode.GREEDY,
            PGBasedPolicyNodeNotFoundMode.FIND_SIMILAR_NODES,
        )
        # Translates to LEFT, LEFT, STABILIZING_RIGHT
        action = policy.act(np.array([-2.3, -0.5, -0.1, 0.7]))

        self.assertEqual(action, 1)

    # TODO Find an acceptable way to test stochastic mode policies (which are stochastic in nature)
