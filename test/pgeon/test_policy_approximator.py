import unittest
from test.domain.cartpole import CartpoleDiscretizer
from test.domain.test_env import State, TestingAgent, TestingDiscretizer, TestingEnv

from pgeon import GraphRepresentation, Predicate
from pgeon.discretizer import PredicateBasedStateRepresentation
from pgeon.policy_approximator import (
    OfflinePolicyApproximator,
    PolicyApproximatorFromBasicObservation,
)
from pgeon.policy_representation import Action


class TestPolicyApproximator(unittest.TestCase):
    """
    Tests for the PolicyApproximatorFromBasicObservation class using GraphRepresentation.
    """

    def setUp(self):
        self.env = TestingEnv()
        self.discretizer = TestingDiscretizer()
        self.representation = GraphRepresentation()
        self.representation.clear()
        self.agent = TestingAgent()

        self.approximator = PolicyApproximatorFromBasicObservation(
            self.discretizer, self.representation, self.env, self.agent
        )

        self.state0 = PredicateBasedStateRepresentation((Predicate(State.ZERO),))
        self.state1 = PredicateBasedStateRepresentation((Predicate(State.ONE),))
        self.state2 = PredicateBasedStateRepresentation((Predicate(State.TWO),))
        self.state3 = PredicateBasedStateRepresentation((Predicate(State.THREE),))

        self.action0: Action = 0
        self.action1: Action = 1

        self.original_get_predicate_space = self.discretizer.get_predicate_space
        self.discretizer.get_predicate_space = lambda: [
            (Predicate(State.ZERO),),
            (Predicate(State.ONE),),
            (Predicate(State.TWO),),
            (Predicate(State.THREE),),
        ]

    def tearDown(self):
        self.discretizer.get_predicate_space = self.original_get_predicate_space

    def test_initialization(self):
        self.assertEqual(self.approximator.discretizer, self.discretizer)
        self.assertEqual(self.approximator.policy_representation, self.representation)
        self.assertEqual(self.approximator.environment, self.env)
        self.assertEqual(self.approximator.agent, self.agent)
        self.assertFalse(self.approximator._is_fit)
        self.assertEqual(self.approximator._trajectories_of_last_fit, [])

    def test_fit(self):
        self.approximator.fit(n_episodes=1)

        self.assertTrue(self.approximator._is_fit)

        states = self.representation.get_all_states()
        self.assertEqual(len(states), 4)
        self.assertIn(self.state0, states)
        self.assertIn(self.state1, states)
        self.assertIn(self.state2, states)
        self.assertIn(self.state3, states)

        self.assertTrue(
            self.representation.has_transition(self.state0, self.state1, self.action0)
        )
        self.assertTrue(
            self.representation.has_transition(self.state1, self.state2, self.action0)
        )
        self.assertTrue(
            self.representation.has_transition(self.state2, self.state3, self.action0)
        )
        self.assertTrue(
            self.representation.has_transition(self.state3, self.state0, self.action0)
        )

        transitions = self.representation.get_all_transitions(include_data=True)
        for transition in transitions:
            from_state, to_state, data = transition
            if isinstance(data, dict) and "frequency" in data:
                self.assertGreater(data["frequency"], 0)
                self.assertIn("probability", data)
                self.assertGreater(data["probability"], 0)
                self.assertLessEqual(data["probability"], 1.0)
                if from_state == self.state0:
                    self.assertEqual(to_state, self.state1)
                    self.assertEqual(data["action"], self.action0)
                    self.assertEqual(data["probability"], 1.0)
                elif from_state == self.state1:
                    self.assertEqual(to_state, self.state2)
                    self.assertEqual(data["action"], self.action0)
                    self.assertEqual(data["probability"], 1.0)
                elif from_state == self.state2:
                    self.assertEqual(to_state, self.state3)
                    self.assertEqual(data["action"], self.action0)
                    self.assertEqual(data["probability"], 1.0)
                elif from_state == self.state3:
                    self.assertEqual(to_state, self.state0)
                    self.assertEqual(data["action"], self.action0)
                    self.assertEqual(data["probability"], 1.0)

    def test_get_nearest_state(self):
        self.approximator.fit(n_episodes=1)

        nearest = self.approximator.get_nearest_state(self.state0)
        self.assertEqual(nearest, self.state0)

        nonexistent_state = PredicateBasedStateRepresentation(
            (Predicate(State.ZERO), Predicate(State.ONE))
        )
        nearest = self.approximator.get_nearest_state(nonexistent_state)
        self.assertIn(nearest, [self.state0, self.state1, self.state2, self.state3])

        # Test with multiple nearest predicates
        self.representation.clear()
        self.representation.add_states_from([self.state0, self.state1])
        nonexistent_state = PredicateBasedStateRepresentation((Predicate(State.TWO),))
        nearest = self.approximator.get_nearest_state(nonexistent_state)
        self.assertIn(nearest, [self.state0, self.state1])

    def test_question1(self):
        self.approximator.fit(n_episodes=1)

        result = self.approximator.question1(self.state0)
        self.assertEqual(len(result), 1)
        action, prob = result[0]
        self.assertEqual(action, 0)
        self.assertEqual(prob, 1.0)

        # Test with multiple possible actions
        self.representation.clear()
        self.representation.add_states_from([self.state0, self.state1, self.state2])
        self.representation.add_transition(
            self.state0, self.state1, self.action0, probability=0.5, frequency=1
        )
        self.representation.add_transition(
            self.state0, self.state2, self.action1, probability=0.5, frequency=1
        )
        result = self.approximator.question1(self.state0)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 0)
        self.assertEqual(result[0][1], 0.5)
        self.assertEqual(result[1][0], 1)
        self.assertEqual(result[1][1], 0.5)

        # Test with no possible actions
        self.representation.clear()
        self.representation.add_state(self.state0)
        result = self.approximator.question1(self.state0)
        self.assertEqual(len(result), 0)

    def test_question2(self):
        self.approximator.fit(n_episodes=1)

        best_nodes = self.approximator.question2(self.action0)
        self.assertEqual(len(best_nodes), 4)
        self.assertIn(self.state0, best_nodes)
        self.assertIn(self.state1, best_nodes)
        self.assertIn(self.state2, best_nodes)
        self.assertIn(self.state3, best_nodes)

        # Test with action being best in some states
        self.representation.clear()
        self.representation.add_states_from([self.state0, self.state1, self.state2])
        self.representation.add_transition(
            self.state0, self.state1, self.action0, probability=1.0, frequency=1
        )
        self.representation.add_transition(
            self.state1, self.state2, 1, probability=1.0, frequency=1
        )
        self.representation.add_transition(
            self.state2, self.state0, self.action0, probability=1.0, frequency=1
        )
        best_nodes = self.approximator.question2(self.action0)
        self.assertEqual(len(best_nodes), 2)
        self.assertIn(self.state0, best_nodes)
        self.assertIn(self.state2, best_nodes)

    def test_question3(self):
        self.approximator.fit(n_episodes=1)

        explanations = self.approximator.question3(self.state0, self.action0)
        self.assertEqual(len(explanations), 0)

        # Test with explanations
        self.representation.clear()
        self.representation.add_states_from([self.state0, self.state1, self.state2])
        self.representation.add_transition(
            self.state0, self.state1, self.action0, probability=1.0, frequency=1
        )
        self.representation.add_transition(
            self.state1, self.state2, 1, probability=1.0, frequency=1
        )
        self.representation.add_transition(
            self.state2, self.state0, self.action0, probability=1.0, frequency=1
        )
        explanations = self.approximator.question3(self.state1, self.action0)
        self.assertEqual(len(explanations), 0)

    def test_creating_graph_representation_programmatically(self):
        self.representation.clear()

        self.representation.add_states_from(
            [self.state0, self.state1, self.state2, self.state3],
            frequency=1,
            probability=0.25,
        )

        transitions = [
            (self.state0, self.state1, self.action0),
            (self.state1, self.state2, self.action0),
            (self.state2, self.state3, self.action0),
            (self.state3, self.state0, self.action0),
        ]
        self.representation.add_transitions_from(
            transitions, frequency=1, probability=1.0
        )

        self.assertEqual(len(self.representation.get_all_states()), 4)
        self.assertEqual(len(self.representation.get_all_transitions()), 4)

        # Set up a new approximator with the graph representation
        approx = PolicyApproximatorFromBasicObservation(
            self.discretizer, self.representation, self.env, self.agent
        )

        result = approx.question1(self.state0)
        self.assertEqual(len(result), 1)
        action, prob = result[0]
        self.assertEqual(action, 0)
        self.assertEqual(prob, 1.0)

    def test_fit_update(self):
        self.approximator.fit(n_episodes=1)
        self.assertEqual(len(self.approximator._trajectories_of_last_fit), 1)

        self.approximator.fit(n_episodes=1, update=True)
        self.assertEqual(len(self.approximator._trajectories_of_last_fit), 2)

    def test_save(self):
        self.approximator.fit(n_episodes=1)

        with self.assertRaises(NotImplementedError):
            self.approximator.save(format="unsupported", path="")

        with self.assertRaises(Exception):
            self.approximator.save(format="csv", path="path")

        with self.assertRaises(Exception):
            self.approximator.save(format="gram", path=["path"])

        with self.assertRaises(Exception):
            self.approximator.save(format="pickle", path=["path"])


class TestOfflinePolicyApproximator(unittest.TestCase):
    def setUp(self):
        self.discretizer = CartpoleDiscretizer()

    def test_from_nodes_and_edges(self):
        approximator = OfflinePolicyApproximator.from_nodes_and_edges(
            "test/data/cartpole_nodes_small.csv",
            "test/data/cartpole_edges_small.csv",
            self.discretizer,
        )

        self.assertTrue(approximator._is_fit)
        self.assertEqual(len(approximator.policy_representation.get_all_states()), 4)
        self.assertEqual(
            len(approximator.policy_representation.get_all_transitions()), 18
        )

    def test_from_nodes_and_trajectories(self):
        approximator = OfflinePolicyApproximator.from_nodes_and_trajectories(
            "test/data/cartpole_nodes_small.csv",
            "test/data/cartpole_trajectories_small.csv",
            self.discretizer,
        )
        self.assertTrue(approximator._is_fit)
        self.assertEqual(len(approximator.policy_representation.get_all_states()), 4)
        self.assertEqual(
            len(approximator.policy_representation.get_all_transitions()), 4
        )

    def test_from_pickle(self):
        # First, create a pickle file to test with
        approximator = OfflinePolicyApproximator.from_nodes_and_edges(
            "test/data/cartpole_nodes_small.csv",
            "test/data/cartpole_edges_small.csv",
            self.discretizer,
        )
        approximator.save("pickle", "test/data/cartpole_small.pickle")

        # Now, load from the pickle file and test
        loaded_approximator = OfflinePolicyApproximator.from_pickle(
            "test/data/cartpole_small.pickle"
        )
        self.assertTrue(loaded_approximator._is_fit)
        self.assertEqual(
            len(loaded_approximator.policy_representation.get_all_states()), 4
        )
        self.assertEqual(
            len(loaded_approximator.policy_representation.get_all_transitions()), 18
        )


if __name__ == "__main__":
    unittest.main()
