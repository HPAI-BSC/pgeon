import unittest

import gymnasium

from pgeon import PolicyGraph, Predicate
from test.domain.cartpole import CartpoleDiscretizer, Position, Velocity, Angle, Action


class TestAskQuestionsToGraph(unittest.TestCase):
    env = gymnasium.make('CartPole-v1')
    discretizer = CartpoleDiscretizer()
    pg_cartpole = PolicyGraph.from_nodes_and_edges('./test/data/cartpole_nodes_small.csv',
                                                   './test/data/cartpole_edges_small.csv',
                                                   env, discretizer
                                                   )

    def test_question_1(self):
        result = self.pg_cartpole.question1(
            (Predicate(Position, [Position.MIDDLE]), Predicate(Velocity, [Velocity.LEFT]), Predicate(Angle, [Angle.STABILIZING_RIGHT]))
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], Action.RIGHT)
        self.assertEqual(result[1][0], Action.LEFT)
        self.assertAlmostEqual(result[0][1], 0.53228, delta=0.001)
        self.assertAlmostEqual(result[1][1], 0.46772, delta=0.001)

    def test_question_1_nearest_predicate(self):
        ...
        # TODO Bug: As of now, the PG picks a random predicate from the entire predicate space (all predicates)
        #           instead of calling the discretizer to find the nearest node
        #      Uncomment when code is fixed
        # result = self.pg_cartpole.question1(
        #     (Predicate(Position, [Position.LEFT]), Predicate(Velocity, [Velocity.LEFT]), Predicate(Angle, [Angle.STABILIZING_RIGHT]))
        # )
        # print(result)
        #
        # self.assertEqual(len(result), 2)
        # self.assertEqual(result[0][0], Action.RIGHT)
        # self.assertEqual(result[1][0], Action.LEFT)
        # self.assertAlmostEqual(result[0][1], 0.53228, delta=0.001)
        # self.assertAlmostEqual(result[1][1], 0.46772, delta=0.001)

    def test_question_1_no_nearest_predicate(self):
        result = self.pg_cartpole.question1(
            (Predicate(Position, [Position.RIGHT]), Predicate(Velocity, [Velocity.RIGHT]), Predicate(Angle, [Angle.STUCK_LEFT]))
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], Action.LEFT)
        self.assertEqual(result[1][0], Action.RIGHT)
        self.assertAlmostEqual(result[0][1], 0.5, delta=0.001)
        self.assertAlmostEqual(result[1][1], 0.5, delta=0.001)

    def test_question_2(self):
        result = self.pg_cartpole.question2(0)

        # TODO Bug: Current PGs return the action of the single edge with most probability (i.e. action of argmax p(s',a|s),
        #           instead of accumulating the probabilities across all edges (i.e. argmax p(a))
        #      Uncomment when code is fixed
        # self.assertIn((Predicate(Position, [Position.MIDDLE]), Predicate(Velocity, [Velocity.LEFT]), Predicate(Angle, [Angle.STUCK_LEFT])),
        #               result)
        # self.assertIn((Predicate(Position, [Position.MIDDLE]), Predicate(Velocity, [Velocity.RIGHT]), Predicate(Angle, [Angle.FALLING_LEFT])),
        #               result)
        # self.assertIn((Predicate(Position, [Position.MIDDLE]), Predicate(Velocity, [Velocity.LEFT]), Predicate(Angle, [Angle.FALLING_LEFT])),
        #               result)
        # self.assertEqual(len(result), 3)

    def test_question_3(self):
        # TODO Find a good discretized state to test
        predicate = tuple()
        result = self.pg_cartpole.question3(predicate, 1)

        # TODO Asserts