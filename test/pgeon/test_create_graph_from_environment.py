import unittest

from pgeon import PolicyGraph, Predicate, GraphRepresentation
from test.domain.test_env import TestingEnv, TestingDiscretizer, TestingAgent, State

class TestCreateGraphFromEnvironment(unittest.TestCase):
    env = TestingEnv()
    discretizer = TestingDiscretizer()
    policy_representation = GraphRepresentation()
    agent = TestingAgent()

    def test_initialize_pg(self):
        pg: PolicyGraph = PolicyGraph(self.discretizer, self.policy_representation, self.env, self.agent)

        self.assertEqual(pg.discretizer, self.discretizer)
        self.assertEqual(pg.environment, self.env)
        self.assertEqual(pg._is_fit, False)
        self.assertEqual(pg._trajectories_of_last_fit, [])

    def test_fit_from_agent_and_env(self):
        pg: PolicyGraph = PolicyGraph(self.discretizer, self.policy_representation, self.env, self.agent)
        pg.fit(self.agent, num_episodes=1)

        self.assertEqual(pg._is_fit, True)
        self.assertEqual(len(pg.graph.nodes), 4)
        self.assertEqual(len(pg.graph.edges), 4)

        p = ((Predicate(State, [State.ZERO ]), ),
             (Predicate(State, [State.ONE  ]), ),
             (Predicate(State, [State.TWO  ]), ),
             (Predicate(State, [State.THREE]), ),
             )

        self.assertIn(p[0], pg.graph.nodes)
        self.assertIn(p[1], pg.graph.nodes)
        self.assertIn(p[2], pg.graph.nodes)
        self.assertIn(p[3], pg.graph.nodes)

        self.assertIn((p[0], p[1], 0), pg.graph.edges)
        self.assertIn((p[1], p[2], 0), pg.graph.edges)
        self.assertIn((p[2], p[3], 0), pg.graph.edges)
        self.assertIn((p[3], p[0], 0), pg.graph.edges)

        self.assertEqual(pg.graph.nodes[p[0]]['frequency'], 8)
        self.assertEqual(pg.graph.nodes[p[1]]['frequency'], 8)
        self.assertEqual(pg.graph.nodes[p[2]]['frequency'], 8)
        self.assertEqual(pg.graph.nodes[p[3]]['frequency'], 7)
        self.assertAlmostEqual(pg.graph.nodes[p[0]]['probability'], 8/31, delta=0.001)
        self.assertAlmostEqual(pg.graph.nodes[p[1]]['probability'], 8/31, delta=0.001)
        self.assertAlmostEqual(pg.graph.nodes[p[2]]['probability'], 8/31, delta=0.001)
        self.assertAlmostEqual(pg.graph.nodes[p[3]]['probability'], 7/31, delta=0.001)

        self.assertEqual(pg.graph.edges[(p[0], p[1], 0)]['frequency'], 8)
        self.assertEqual(pg.graph.edges[(p[1], p[2], 0)]['frequency'], 8)
        self.assertEqual(pg.graph.edges[(p[2], p[3], 0)]['frequency'], 7)
        self.assertEqual(pg.graph.edges[(p[3], p[0], 0)]['frequency'], 7)
        self.assertAlmostEqual(pg.graph.edges[(p[0], p[1], 0)]['probability'], 1, delta=0.001)
        self.assertAlmostEqual(pg.graph.edges[(p[1], p[2], 0)]['probability'], 1, delta=0.001)
        self.assertAlmostEqual(pg.graph.edges[(p[2], p[3], 0)]['probability'], 1, delta=0.001)
        self.assertAlmostEqual(pg.graph.edges[(p[3], p[0], 0)]['probability'], 1, delta=0.001)
