import unittest

from pgeon import PolicyGraph, Predicate
from test.domain.test_env import TestingEnv, TestingDiscretizer, TestingAgent, State


class TestCreateGraphFromEnvironment(unittest.TestCase):
    env = TestingEnv()
    discretizer = TestingDiscretizer()
    agent = TestingAgent()

    def test_initialize_pg(self):
        pg: PolicyGraph = PolicyGraph(self.env, self.discretizer)

        self.assertEqual(pg.discretizer, self.discretizer)
        self.assertEqual(pg.environment, self.env)
        self.assertEqual(pg._is_fit, False)
        self.assertEqual(pg._trajectories_of_last_fit, [])

    def test_fit_from_agent_and_env(self):
        pg: PolicyGraph = PolicyGraph(self.env, self.discretizer)
        pg.fit(self.agent, num_episodes=1)

        self.assertEqual(pg._is_fit, True)
        self.assertEqual(len(pg.nodes), 4)
        self.assertEqual(len(pg.edges), 4)

        p = ((Predicate(State, [State.ZERO ]), ),
             (Predicate(State, [State.ONE  ]), ),
             (Predicate(State, [State.TWO  ]), ),
             (Predicate(State, [State.THREE]), ),
             )

        print(type(list(pg.nodes)[0]))
        print(type(list(pg.nodes)[0][0]))
        print(pg.nodes)

        self.assertIn(p[0], pg.nodes)
        self.assertIn(p[1], pg.nodes)
        self.assertIn(p[2], pg.nodes)
        self.assertIn(p[3], pg.nodes)

        self.assertIn((p[0], p[1], 0), pg.edges)
        self.assertIn((p[1], p[2], 0), pg.edges)
        self.assertIn((p[2], p[3], 0), pg.edges)
        self.assertIn((p[3], p[0], 0), pg.edges)

        self.assertEqual(pg.nodes[p[0]]['frequency'], 8)
        self.assertEqual(pg.nodes[p[1]]['frequency'], 8)
        self.assertEqual(pg.nodes[p[2]]['frequency'], 8)
        self.assertEqual(pg.nodes[p[3]]['frequency'], 7)
        self.assertAlmostEqual(pg.nodes[p[0]]['probability'], 8/31, delta=0.001)
        self.assertAlmostEqual(pg.nodes[p[1]]['probability'], 8/31, delta=0.001)
        self.assertAlmostEqual(pg.nodes[p[2]]['probability'], 8/31, delta=0.001)
        self.assertAlmostEqual(pg.nodes[p[3]]['probability'], 7/31, delta=0.001)

        self.assertEqual(pg.edges[(p[0], p[1], 0)]['frequency'], 8)
        self.assertEqual(pg.edges[(p[1], p[2], 0)]['frequency'], 8)
        self.assertEqual(pg.edges[(p[2], p[3], 0)]['frequency'], 7)
        self.assertEqual(pg.edges[(p[3], p[0], 0)]['frequency'], 7)
        self.assertAlmostEqual(pg.edges[(p[0], p[1], 0)]['probability'], 1, delta=0.001)
        self.assertAlmostEqual(pg.edges[(p[1], p[2], 0)]['probability'], 1, delta=0.001)
        self.assertAlmostEqual(pg.edges[(p[2], p[3], 0)]['probability'], 1, delta=0.001)
        self.assertAlmostEqual(pg.edges[(p[3], p[0], 0)]['probability'], 1, delta=0.001)
