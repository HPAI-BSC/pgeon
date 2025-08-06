import unittest
from test.domain.cartpole import Angle, CartpoleDiscretizer, Position, Velocity

import gymnasium

from pgeon import Predicate
from pgeon.policy_approximator import OfflinePolicyApproximator


class TestLoadGraphFromFile(unittest.TestCase):
    env = gymnasium.make("CartPole-v1")
    discretizer = CartpoleDiscretizer()

    # TODO: Update the test to save to a pickle file first, then load from it
    pg_pickle = "./test/data/cartpole_small.pickle"
    pg_csv_edges = "./test/data/cartpole_edges_small.csv"
    pg_csv_nodes = "./test/data/cartpole_nodes_small.csv"
    pg_csv_trajectories = "./test/data/cartpole_trajectories_small.csv"

    def test_load_pg_from_nodes_and_edges(self):
        pg = OfflinePolicyApproximator.from_nodes_and_edges(
            self.pg_csv_nodes, self.pg_csv_edges, self.discretizer
        )

        self.assertEqual(pg._is_fit, True)
        self.assertEqual(len(pg.policy_representation.get_all_states()), 4)
        self.assertEqual(len(pg.policy_representation.get_all_transitions()), 18)
        self.assertEqual(pg._trajectories_of_last_fit, [])

        p = (
            (
                Predicate(Position, [Position.MIDDLE]),
                Predicate(Velocity, [Velocity.LEFT]),
                Predicate(Angle, [Angle.STUCK_LEFT]),
            ),
            (
                Predicate(Position, [Position.MIDDLE]),
                Predicate(Velocity, [Velocity.LEFT]),
                Predicate(Angle, [Angle.STABILIZING_RIGHT]),
            ),
            (
                Predicate(Position, [Position.MIDDLE]),
                Predicate(Velocity, [Velocity.RIGHT]),
                Predicate(Angle, [Angle.FALLING_LEFT]),
            ),
            (
                Predicate(Position, [Position.MIDDLE]),
                Predicate(Velocity, [Velocity.LEFT]),
                Predicate(Angle, [Angle.FALLING_LEFT]),
            ),
        )

        self.assertIn(p[0], pg.policy_representation.get_all_states())
        self.assertIn(p[1], pg.policy_representation.get_all_states())
        self.assertIn(p[2], pg.policy_representation.get_all_states())
        self.assertIn(p[3], pg.policy_representation.get_all_states())

        self.assertAlmostEqual(
            pg.policy_representation.get_state_attributes("probability")[p[2]],
            0.51863,
            delta=0.001,
        )
        self.assertEqual(
            pg.policy_representation.get_state_attributes("frequency")[p[2]], 2464
        )

        self.assertTrue(pg.policy_representation.has_transition(p[1], p[2], 1))
        self.assertTrue(pg.policy_representation.has_transition(p[2], p[1], 0))
        self.assertTrue(pg.policy_representation.has_transition(p[3], p[3], 1))
        self.assertTrue(pg.policy_representation.has_transition(p[2], p[1], 0))
        self.assertTrue(pg.policy_representation.has_transition(p[2], p[3], 0))
        self.assertTrue(pg.policy_representation.has_transition(p[1], p[0], 1))

        self.assertAlmostEqual(
            pg.policy_representation.get_transition_data(p[1], p[3], 1)["probability"],
            0.06126,
            delta=0.001,
        )
        self.assertEqual(
            pg.policy_representation.get_transition_data(p[1], p[3], 1)["frequency"],
            74,
        )
        self.assertEqual(
            pg.policy_representation.get_transition_data(p[1], p[3], 1)["action"], 1
        )
        self.assertAlmostEqual(
            pg.policy_representation.get_transition_data(p[1], p[1], 1)["probability"],
            0.26738,
            delta=0.001,
        )
        self.assertEqual(
            pg.policy_representation.get_transition_data(p[1], p[1], 1)["frequency"],
            323,
        )
        self.assertEqual(
            pg.policy_representation.get_transition_data(p[1], p[1], 1)["action"], 1
        )
        self.assertAlmostEqual(
            pg.policy_representation.get_transition_data(p[3], p[0], 0)["probability"],
            0.21829,
            delta=0.001,
        )
        self.assertEqual(
            pg.policy_representation.get_transition_data(p[3], p[0], 0)["frequency"],
            74,
        )
        self.assertEqual(
            pg.policy_representation.get_transition_data(p[3], p[0], 0)["action"], 0
        )

    def test_load_pg_from_nodes_and_trajectories(self):
        pg = OfflinePolicyApproximator.from_nodes_and_trajectories(
            self.pg_csv_nodes, self.pg_csv_trajectories, self.discretizer
        )

        self.assertEqual(pg._is_fit, True)
        self.assertEqual(len(pg.policy_representation.get_all_states()), 4)
        self.assertEqual(len(pg.policy_representation.get_all_transitions()), 18)
        self.assertEqual(len(pg._trajectories_of_last_fit), 4)

        p = (
            (
                Predicate(Position, [Position.MIDDLE]),
                Predicate(Velocity, [Velocity.LEFT]),
                Predicate(Angle, [Angle.STUCK_LEFT]),
            ),
            (
                Predicate(Position, [Position.MIDDLE]),
                Predicate(Velocity, [Velocity.LEFT]),
                Predicate(Angle, [Angle.STABILIZING_RIGHT]),
            ),
            (
                Predicate(Position, [Position.MIDDLE]),
                Predicate(Velocity, [Velocity.RIGHT]),
                Predicate(Angle, [Angle.FALLING_LEFT]),
            ),
            (
                Predicate(Position, [Position.MIDDLE]),
                Predicate(Velocity, [Velocity.LEFT]),
                Predicate(Angle, [Angle.FALLING_LEFT]),
            ),
        )

        self.assertIn(p[0], pg.policy_representation.get_all_states())
        self.assertIn(p[1], pg.policy_representation.get_all_states())
        self.assertIn(p[2], pg.policy_representation.get_all_states())
        self.assertIn(p[3], pg.policy_representation.get_all_states())

        self.assertEqual(
            pg.policy_representation.get_state_attributes("frequency")[p[0]], 33
        )
        self.assertEqual(
            pg.policy_representation.get_state_attributes("frequency")[p[1]], 52
        )
        self.assertEqual(
            pg.policy_representation.get_state_attributes("frequency")[p[2]], 45
        )
        self.assertEqual(
            pg.policy_representation.get_state_attributes("frequency")[p[3]], 23
        )

    @unittest.skip(
        "Skipping pickle test until we update the test to save to a pickle file first"
    )
    def test_load_pg_from_pickle(self):
        pg = OfflinePolicyApproximator.from_pickle(self.pg_pickle)

        self.assertEqual(pg._is_fit, True)
        self.assertEqual(len(pg.policy_representation.get_all_states()), 4)
        self.assertEqual(len(pg.policy_representation.get_all_transitions()), 18)
        self.assertEqual(len(pg._trajectories_of_last_fit), 0)

        p = (
            (
                Predicate(Position, [Position.MIDDLE]),
                Predicate(Velocity, [Velocity.LEFT]),
                Predicate(Angle, [Angle.STUCK_LEFT]),
            ),
            (
                Predicate(Position, [Position.MIDDLE]),
                Predicate(Velocity, [Velocity.LEFT]),
                Predicate(Angle, [Angle.STABILIZING_RIGHT]),
            ),
            (
                Predicate(Position, [Position.MIDDLE]),
                Predicate(Velocity, [Velocity.RIGHT]),
                Predicate(Angle, [Angle.FALLING_LEFT]),
            ),
            (
                Predicate(Position, [Position.MIDDLE]),
                Predicate(Velocity, [Velocity.LEFT]),
                Predicate(Angle, [Angle.FALLING_LEFT]),
            ),
        )

        self.assertIn(p[0], pg.policy_representation.get_all_states())
        self.assertIn(p[1], pg.policy_representation.get_all_states())
        self.assertIn(p[2], pg.policy_representation.get_all_states())
        self.assertIn(p[3], pg.policy_representation.get_all_states())

        self.assertAlmostEqual(
            pg.policy_representation.get_state_attributes("probability")[p[2]],
            0.51863,
            delta=0.001,
        )
        self.assertEqual(
            pg.policy_representation.get_state_attributes("frequency")[p[2]], 2464
        )

        self.assertTrue(pg.policy_representation.has_transition(p[1], p[2], 1))
        self.assertTrue(pg.policy_representation.has_transition(p[2], p[1], 0))
        self.assertTrue(pg.policy_representation.has_transition(p[3], p[3], 1))
        self.assertTrue(pg.policy_representation.has_transition(p[2], p[1], 0))
        self.assertTrue(pg.policy_representation.has_transition(p[2], p[3], 0))
        self.assertTrue(pg.policy_representation.has_transition(p[1], p[0], 1))

        self.assertAlmostEqual(
            pg.policy_representation.get_transition_data(p[1], p[3], 1)["probability"],
            0.06126,
            delta=0.001,
        )
        self.assertEqual(
            pg.policy_representation.get_transition_data(p[1], p[3], 1)["frequency"],
            74,
        )
        self.assertEqual(
            pg.policy_representation.get_transition_data(p[1], p[3], 1)["action"], 1
        )
        self.assertAlmostEqual(
            pg.policy_representation.get_transition_data(p[1], p[1], 1)["probability"],
            0.26738,
            delta=0.001,
        )
        self.assertEqual(
            pg.policy_representation.get_transition_data(p[1], p[1], 1)["frequency"],
            323,
        )
        self.assertEqual(
            pg.policy_representation.get_transition_data(p[1], p[1], 1)["action"], 1
        )
        self.assertAlmostEqual(
            pg.policy_representation.get_transition_data(p[3], p[0], 0)["probability"],
            0.21829,
            delta=0.001,
        )
        self.assertEqual(
            pg.policy_representation.get_transition_data(p[3], p[0], 0)["frequency"],
            74,
        )
        self.assertEqual(
            pg.policy_representation.get_transition_data(p[3], p[0], 0)["action"], 0
        )
