import unittest

import gymnasium

from pgeon import PolicyGraph, Predicate
from test.domain.cartpole import CartpoleDiscretizer, Position, Velocity, Angle


class TestLoadGraphFromFile(unittest.TestCase):
    env = gymnasium.make('CartPole-v1')
    discretizer = CartpoleDiscretizer()

    # TODO: Update the test to save to a pickle file first, then load from it
    pg_pickle = './test/data/cartpole_small.pickle'
    pg_csv_edges = './test/data/cartpole_edges_small.csv'
    pg_csv_nodes = './test/data/cartpole_nodes_small.csv'
    pg_csv_trajectories = './test/data/cartpole_trajectories_small.csv'

    def test_load_pg_from_nodes_and_edges(self):
        pg = PolicyGraph.from_nodes_and_edges(self.pg_csv_nodes,
                                              self.pg_csv_edges,
                                              self.env,
                                              self.discretizer)

        self.assertEqual(pg._is_fit, True)
        self.assertEqual(len(pg.graph.nodes), 4)
        self.assertEqual(len(pg.graph.edges), 18)
        self.assertEqual(pg._trajectories_of_last_fit, [])

        p = ((Predicate(Position, [Position.MIDDLE]), Predicate(Velocity,  [Velocity.LEFT]), Predicate(Angle, [Angle.STUCK_LEFT])),
             (Predicate(Position, [Position.MIDDLE]), Predicate(Velocity,  [Velocity.LEFT]), Predicate(Angle, [Angle.STABILIZING_RIGHT])),
             (Predicate(Position, [Position.MIDDLE]), Predicate(Velocity, [Velocity.RIGHT]), Predicate(Angle, [Angle.FALLING_LEFT])),
             (Predicate(Position, [Position.MIDDLE]), Predicate(Velocity,  [Velocity.LEFT]), Predicate(Angle, [Angle.FALLING_LEFT]))
             )

        self.assertIn(p[0], pg.graph.nodes)
        self.assertIn(p[1], pg.graph.nodes)
        self.assertIn(p[2], pg.graph.nodes)
        self.assertIn(p[3], pg.graph.nodes)

        self.assertAlmostEqual(pg.graph.nodes[p[2]]['probability'], 0.51863, delta=0.001)
        self.assertEqual(pg.graph.nodes[p[2]]['frequency'], 2464)

        self.assertIn((p[1], p[2], 1), pg.graph.edges)
        self.assertIn((p[2], p[1], 0), pg.graph.edges)
        self.assertIn((p[3], p[3], 1), pg.graph.edges)
        self.assertIn((p[2], p[1], 0), pg.graph.edges)
        self.assertIn((p[2], p[3], 0), pg.graph.edges)
        self.assertIn((p[1], p[0], 1), pg.graph.edges)

        self.assertAlmostEqual(pg.graph.edges[(p[1], p[3], 1)]['probability'], 0.06126, delta=0.001)
        self.assertEqual(pg.graph.edges[(p[1], p[3], 1)]['frequency'], 74)
        self.assertEqual(pg.graph.edges[(p[1], p[3], 1)]['action'], 1)
        self.assertAlmostEqual(pg.graph.edges[(p[1], p[1], 1)]['probability'], 0.26738, delta=0.001)
        self.assertEqual(pg.graph.edges[(p[1], p[1], 1)]['frequency'], 323)
        self.assertEqual(pg.graph.edges[(p[1], p[1], 1)]['action'], 1)
        self.assertAlmostEqual(pg.graph.edges[(p[3], p[0], 0)]['probability'], 0.21829, delta=0.001)
        self.assertEqual(pg.graph.edges[(p[3], p[0], 0)]['frequency'], 74)
        self.assertEqual(pg.graph.edges[(p[3], p[0], 0)]['action'], 0)

    def test_load_pg_from_nodes_and_trajectories(self):
        pg = PolicyGraph.from_nodes_and_trajectories(self.pg_csv_nodes,
                                                     self.pg_csv_trajectories,
                                                     self.env,
                                                     self.discretizer)

        self.assertEqual(pg._is_fit, True)
        self.assertEqual(len(pg.graph.nodes), 4)
        self.assertEqual(len(pg.graph.edges), 18)
        self.assertEqual(len(pg._trajectories_of_last_fit), 4)

        p = ((Predicate(Position, [Position.MIDDLE]), Predicate(Velocity,  [Velocity.LEFT]), Predicate(Angle, [Angle.STUCK_LEFT])),
             (Predicate(Position, [Position.MIDDLE]), Predicate(Velocity,  [Velocity.LEFT]), Predicate(Angle, [Angle.STABILIZING_RIGHT])),
             (Predicate(Position, [Position.MIDDLE]), Predicate(Velocity, [Velocity.RIGHT]), Predicate(Angle, [Angle.FALLING_LEFT])),
             (Predicate(Position, [Position.MIDDLE]), Predicate(Velocity,  [Velocity.LEFT]), Predicate(Angle, [Angle.FALLING_LEFT]))
             )

        self.assertIn(p[0], pg.graph.nodes)
        self.assertIn(p[1], pg.graph.nodes)
        self.assertIn(p[2], pg.graph.nodes)
        self.assertIn(p[3], pg.graph.nodes)

        self.assertEqual(pg.graph.nodes[p[0]]['frequency'], 33)
        self.assertEqual(pg.graph.nodes[p[1]]['frequency'], 52)
        self.assertEqual(pg.graph.nodes[p[2]]['frequency'], 45)
        self.assertEqual(pg.graph.nodes[p[3]]['frequency'], 23)

        # TODO Bug: We forgot to add `normalize()` at the end of `from_nodes_and_trajectories()`
        # self.assertAlmostEqual(pg.nodes[p[0]]['probability'], 0.21569, delta=0.001)
        # self.assertAlmostEqual(pg.nodes[p[1]]['probability'], 0.33987, delta=0.001)
        # self.assertAlmostEqual(pg.nodes[p[2]]['probability'], 0.29412, delta=0.001)
        # self.assertAlmostEqual(pg.nodes[p[3]]['probability'], 0.15033, delta=0.001)

        self.assertIn((p[0], p[2], 1), pg.graph.edges)
        self.assertIn((p[0], p[1], 0), pg.graph.edges)
        self.assertIn((p[0], p[3], 1), pg.graph.edges)
        self.assertIn((p[1], p[1], 0), pg.graph.edges)
        self.assertIn((p[1], p[1], 1), pg.graph.edges)
        self.assertIn((p[1], p[0], 1), pg.graph.edges)
        self.assertIn((p[1], p[3], 1), pg.graph.edges)
        self.assertIn((p[1], p[2], 1), pg.graph.edges)
        self.assertIn((p[2], p[2], 1), pg.graph.edges)
        self.assertIn((p[2], p[2], 0), pg.graph.edges)
        self.assertIn((p[2], p[3], 0), pg.graph.edges)
        self.assertIn((p[2], p[0], 0), pg.graph.edges)
        self.assertIn((p[2], p[1], 0), pg.graph.edges)
        self.assertIn((p[3], p[1], 0), pg.graph.edges)
        self.assertIn((p[3], p[2], 1), pg.graph.edges)
        self.assertIn((p[3], p[0], 0), pg.graph.edges)
        self.assertIn((p[3], p[3], 1), pg.graph.edges)
        self.assertIn((p[3], p[3], 0), pg.graph.edges)
        self.assertEqual(pg.graph.edges[(p[0], p[2], 1)]['frequency'], 10)
        self.assertEqual(pg.graph.edges[(p[0], p[1], 0)]['frequency'], 19)
        self.assertEqual(pg.graph.edges[(p[0], p[3], 1)]['frequency'], 3)
        self.assertEqual(pg.graph.edges[(p[1], p[1], 0)]['frequency'], 15)
        self.assertEqual(pg.graph.edges[(p[1], p[1], 1)]['frequency'], 12)
        self.assertEqual(pg.graph.edges[(p[1], p[0], 1)]['frequency'], 17)
        self.assertEqual(pg.graph.edges[(p[1], p[3], 1)]['frequency'], 4)
        self.assertEqual(pg.graph.edges[(p[1], p[2], 1)]['frequency'], 2)
        self.assertEqual(pg.graph.edges[(p[2], p[2], 1)]['frequency'], 13)
        self.assertEqual(pg.graph.edges[(p[2], p[2], 0)]['frequency'], 15)
        self.assertEqual(pg.graph.edges[(p[2], p[3], 0)]['frequency'], 7)
        self.assertEqual(pg.graph.edges[(p[2], p[0], 0)]['frequency'], 8)
        self.assertEqual(pg.graph.edges[(p[2], p[1], 0)]['frequency'], 1)
        self.assertEqual(pg.graph.edges[(p[3], p[1], 0)]['frequency'], 5)
        self.assertEqual(pg.graph.edges[(p[3], p[2], 1)]['frequency'], 5)
        self.assertEqual(pg.graph.edges[(p[3], p[0], 0)]['frequency'], 4)
        self.assertEqual(pg.graph.edges[(p[3], p[3], 1)]['frequency'], 4)
        self.assertEqual(pg.graph.edges[(p[3], p[3], 0)]['frequency'], 5)

        # We forgot to add `normalize()` at the end of `from_nodes_and_trajectories()`
        # self.assertAlmostEqual(pg.edges[(p[0], p[2], 1)]['probability'], 0.06711, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[0], p[1], 0)]['probability'], 0.12752, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[0], p[3], 1)]['probability'], 0.02013, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[1], p[1], 0)]['probability'], 0.10067, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[1], p[1], 1)]['probability'], 0.08054, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[1], p[0], 1)]['probability'], 0.11409, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[1], p[3], 1)]['probability'], 0.02685, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[1], p[2], 1)]['probability'], 0.01342, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[2], p[2], 1)]['probability'], 0.08725, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[2], p[2], 0)]['probability'], 0.10067, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[2], p[3], 0)]['probability'], 0.04698, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[2], p[0], 0)]['probability'], 0.05369, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[2], p[1], 0)]['probability'], 0.00671, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[3], p[1], 0)]['probability'], 0.03356, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[3], p[2], 1)]['probability'], 0.03356, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[3], p[0], 0)]['probability'], 0.02685, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[3], p[3], 1)]['probability'], 0.02685, delta=0.001)
        # self.assertAlmostEqual(pg.edges[(p[3], p[3], 0)]['probability'], 0.03356, delta=0.001)

    @unittest.skip("Skipping pickle test until we update the test to save to a pickle file first")
    def test_load_pg_from_pickle(self):
        pg = PolicyGraph.from_pickle(self.pg_pickle)

        self.assertEqual(pg._is_fit, True)
        self.assertEqual(len(pg.graph.nodes), 4)
        self.assertEqual(len(pg.graph.edges), 18)
        self.assertEqual(len(pg._trajectories_of_last_fit), 0)

        p = ((Predicate(Position, [Position.MIDDLE]), Predicate(Velocity,  [Velocity.LEFT]), Predicate(Angle, [Angle.STUCK_LEFT])),
             (Predicate(Position, [Position.MIDDLE]), Predicate(Velocity,  [Velocity.LEFT]), Predicate(Angle, [Angle.STABILIZING_RIGHT])),
             (Predicate(Position, [Position.MIDDLE]), Predicate(Velocity, [Velocity.RIGHT]), Predicate(Angle, [Angle.FALLING_LEFT])),
             (Predicate(Position, [Position.MIDDLE]), Predicate(Velocity,  [Velocity.LEFT]), Predicate(Angle, [Angle.FALLING_LEFT]))
             )

        self.assertIn(p[0], pg.graph.nodes)
        self.assertIn(p[1], pg.graph.nodes)
        self.assertIn(p[2], pg.graph.nodes)
        self.assertIn(p[3], pg.graph.nodes)

        self.assertAlmostEqual(pg.graph.nodes[p[2]]['probability'], 0.51863, delta=0.001)
        self.assertEqual(pg.graph.nodes[p[2]]['frequency'], 2464)

        self.assertIn((p[1], p[2], 1), pg.graph.edges)
        self.assertIn((p[2], p[1], 0), pg.graph.edges)
        self.assertIn((p[3], p[3], 1), pg.graph.edges)
        self.assertIn((p[2], p[1], 0), pg.graph.edges)
        self.assertIn((p[2], p[3], 0), pg.graph.edges)
        self.assertIn((p[1], p[0], 1), pg.graph.edges)

        self.assertAlmostEqual(pg.graph.edges[(p[1], p[3], 1)]['probability'], 0.06126, delta=0.001)
        self.assertEqual(pg.graph.edges[(p[1], p[3], 1)]['frequency'], 74)
        self.assertEqual(pg.graph.edges[(p[1], p[3], 1)]['action'], 1)
        self.assertAlmostEqual(pg.edges[(p[1], p[1], 1)]['probability'], 0.26738, delta=0.001)
        self.assertEqual(pg.graph.edges[(p[1], p[1], 1)]['frequency'], 323)
        self.assertEqual(pg.graph.edges[(p[1], p[1], 1)]['action'], 1)
        self.assertAlmostEqual(pg.graph.edges[(p[3], p[0], 0)]['probability'], 0.21829, delta=0.001)
        self.assertEqual(pg.graph.edges[(p[3], p[0], 0)]['frequency'], 74)
        self.assertEqual(pg.graph.edges[(p[3], p[0], 0)]['action'], 0)
