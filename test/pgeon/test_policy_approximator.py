import unittest

from pgeon import GraphRepresentation, Predicate
from pgeon.policy_approximator import PolicyApproximatorFromBasicObservation
from test.domain.test_env import State, TestingDiscretizer, TestingEnv, TestingAgent
from pgeon.discretizer import PredicateBasedStateRepresentation
from pgeon.policy_representation import Action


class TestPolicyApproximator(unittest.TestCase):
    """
    Tests for the PolicyApproximatorFromBasicObservation class using GraphRepresentation.
    """

    def setUp(self):
        self.env = TestingEnv()
        self.discretizer = TestingDiscretizer()
        self.representation = GraphRepresentation()
        self.agent = TestingAgent()

        self.approximator = PolicyApproximatorFromBasicObservation(
            self.discretizer, self.representation, self.env, self.agent
        )

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

        self.original_get_predicate_space = self.discretizer.get_predicate_space
        self.discretizer.get_predicate_space = lambda: [
            (Predicate(State, [State.ZERO]),),
            (Predicate(State, [State.ONE]),),
            (Predicate(State, [State.TWO]),),
            (Predicate(State, [State.THREE]),),
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

    def test_get_possible_actions(self):
        self.approximator.fit(n_episodes=1)

        possible_actions = self.approximator.get_possible_actions(self.state0)
        self.assertEqual(len(possible_actions), 1)

        action, prob = possible_actions[0]
        self.assertEqual(action, 0)
        self.assertEqual(prob, 1.0)

        nonexistent_state = PredicateBasedStateRepresentation(
            (Predicate(State, [State.ZERO]), Predicate(State, [State.ONE]))
        )
        possible_actions = self.approximator.get_possible_actions(nonexistent_state)
        self.assertEqual(len(possible_actions), 1)
        self.assertEqual(possible_actions[0][0], 0)
        self.assertEqual(possible_actions[0][1], 1.0)

    def test_get_nearest_predicate(self):
        self.approximator.fit(n_episodes=1)

        nearest = self.approximator.get_nearest_predicate(self.state0)
        self.assertEqual(nearest, self.state0)

        nonexistent_state = PredicateBasedStateRepresentation(
            (Predicate(State, [State.ZERO]), Predicate(State, [State.ONE]))
        )
        nearest = self.approximator.get_nearest_predicate(nonexistent_state)
        self.assertIn(nearest, [self.state0, self.state1, self.state2, self.state3])

    def test_question1(self):
        self.approximator.fit(n_episodes=1)

        result = self.approximator.question1(self.state0)
        self.assertEqual(len(result), 1)
        action, prob = result[0]
        self.assertEqual(action, 0)
        self.assertEqual(prob, 1.0)

    def test_question2(self):
        self.approximator.fit(n_episodes=1)

        best_nodes = self.approximator.question2(self.action0)
        self.assertEqual(len(best_nodes), 4)
        self.assertIn(self.state0, best_nodes)
        self.assertIn(self.state1, best_nodes)
        self.assertIn(self.state2, best_nodes)
        self.assertIn(self.state3, best_nodes)

    def test_question3(self):
        self.approximator.fit(n_episodes=1)

        explanations = self.approximator.question3(self.state0, self.action0)
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


if __name__ == "__main__":
    unittest.main()
