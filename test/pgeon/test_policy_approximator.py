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
        """Set up test environment, discretizer, and policy representation."""
        self.env = TestingEnv()
        self.discretizer = TestingDiscretizer()
        self.representation = GraphRepresentation()
        self.agent = TestingAgent()

        # Initialize the policy approximator
        self.approximator = PolicyApproximatorFromBasicObservation(
            self.discretizer, self.representation, self.env, self.agent
        )

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

        # TestingEnv only supports action 0
        self.action0: Action = 0  # type: ignore

        # Patch the get_predicate_space method in the discretizer for testing
        self.original_get_predicate_space = self.discretizer.get_predicate_space
        self.discretizer.get_predicate_space = lambda: [
            (Predicate(State, [State.ZERO]),),
            (Predicate(State, [State.ONE]),),
            (Predicate(State, [State.TWO]),),
            (Predicate(State, [State.THREE]),),
        ]

    def tearDown(self):
        """Restore the original get_predicate_space method."""
        self.discretizer.get_predicate_space = self.original_get_predicate_space

    def test_initialization(self):
        """Test initialization of policy approximator."""
        self.assertEqual(self.approximator.discretizer, self.discretizer)
        self.assertEqual(self.approximator.policy_representation, self.representation)
        self.assertEqual(self.approximator.environment, self.env)
        self.assertEqual(self.approximator.agent, self.agent)
        self.assertFalse(self.approximator._is_fit)
        self.assertEqual(self.approximator._trajectories_of_last_fit, [])

    def test_fit(self):
        """Test fitting the policy approximator."""
        # Fit the approximator with 1 episode
        self.approximator.fit(n_episodes=1)

        # Verify the approximator is fit
        self.assertTrue(self.approximator._is_fit)

        # Check that states were added to the representation
        states = self.representation.get_all_states()
        self.assertEqual(len(states), 4)
        self.assertIn(self.state0, states)
        self.assertIn(self.state1, states)
        self.assertIn(self.state2, states)
        self.assertIn(self.state3, states)

        # Check that transitions were added
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

        # Check transition counts
        transitions = self.representation.get_all_transitions(include_data=True)
        for transition in transitions:
            from_state, to_state, data = transition
            if isinstance(data, dict) and "frequency" in data:
                # Each transition should have been observed at least once
                self.assertGreater(data["frequency"], 0)

                # Probability should be set
                self.assertIn("probability", data)
                self.assertGreater(data["probability"], 0)
                self.assertLessEqual(data["probability"], 1.0)

    def test_get_possible_actions(self):
        """Test getting possible actions from a state."""
        # Fit the approximator
        self.approximator.fit(n_episodes=1)

        # Test with a state that has outgoing transitions
        possible_actions = self.approximator.get_possible_actions(self.state0)
        self.assertEqual(len(possible_actions), 1)  # Only one action (0) in TestingEnv

        # The first item should be a tuple (action, probability)
        action, prob = possible_actions[0]
        self.assertEqual(action, 0)  # The action from TestingEnv.all_actions()
        self.assertEqual(
            prob, 1.0
        )  # Probability should be 1.0 since there's only one possible action

        # Test with a state that doesn't exist in the representation
        nonexistent_state = PredicateBasedStateRepresentation(
            (Predicate(State, [State.ZERO]), Predicate(State, [State.ONE]))
        )
        possible_actions = self.approximator.get_possible_actions(nonexistent_state)
        self.assertEqual(
            len(possible_actions), 1
        )  # Should return one action with equal probability
        self.assertEqual(possible_actions[0][0], 0)  # Should be action 0
        self.assertEqual(possible_actions[0][1], 1.0)  # Probability should be 1.0

    def test_get_nearest_predicate(self):
        """Test getting the nearest predicate."""
        # Fit the approximator
        self.approximator.fit(n_episodes=1)

        # Test with a state that exists
        nearest = self.approximator.get_nearest_predicate(self.state0)
        self.assertEqual(nearest, self.state0)

        # Test with a state that doesn't exist
        nonexistent_state = PredicateBasedStateRepresentation(
            (Predicate(State, [State.ZERO]), Predicate(State, [State.ONE]))
        )
        nearest = self.approximator.get_nearest_predicate(nonexistent_state)
        # The result should be a state in the representation
        self.assertIn(nearest, [self.state0, self.state1, self.state2, self.state3])

    def test_question1(self):
        """Test the question1 method (what actions would you take in state X?)."""
        # Fit the approximator
        self.approximator.fit(n_episodes=1)

        # Test with a state that has outgoing transitions
        result = self.approximator.question1(self.state0)
        self.assertEqual(len(result), 1)
        action, prob = result[0]
        self.assertEqual(action, 0)
        self.assertEqual(prob, 1.0)

    def test_question2(self):
        """Test the question2 method (when do you perform action X?)."""
        # Fit the approximator
        self.approximator.fit(n_episodes=1)

        # Test with action 0
        best_nodes = self.approximator.question2(self.action0)
        # All states should be best for action 0 since it's the only action
        self.assertEqual(len(best_nodes), 4)
        self.assertIn(self.state0, best_nodes)
        self.assertIn(self.state1, best_nodes)
        self.assertIn(self.state2, best_nodes)
        self.assertIn(self.state3, best_nodes)

    def test_question3(self):
        """Test the question3 method (why do you perform action X in state Y?)."""
        # Fit the approximator
        self.approximator.fit(n_episodes=1)

        # Test with state0 and action0
        explanations = self.approximator.question3(self.state0, self.action0)
        # In our simple environment, there's only one action, so this should be empty
        # as there are no alternative actions to explain
        self.assertEqual(len(explanations), 0)

    def test_creating_graph_representation_programmatically(self):
        """Test creating a graph representation programmatically."""
        # Clear any existing data
        self.representation.clear()

        # Add states
        self.representation.add_states_from(
            [self.state0, self.state1, self.state2, self.state3],
            frequency=1,
            probability=0.25,
        )

        # Add transitions
        transitions = [
            (self.state0, self.state1, self.action0),
            (self.state1, self.state2, self.action0),
            (self.state2, self.state3, self.action0),
            (self.state3, self.state0, self.action0),
        ]
        self.representation.add_transitions_from(
            transitions, frequency=1, probability=1.0
        )

        # Verify the structure
        self.assertEqual(len(self.representation.get_all_states()), 4)
        self.assertEqual(len(self.representation.get_all_transitions()), 4)

        # Create a new approximator with this representation
        approx = PolicyApproximatorFromBasicObservation(
            self.discretizer, self.representation, self.env, self.agent
        )

        # Test that we can query the approximator with our manually created representation
        result = approx.question1(self.state0)
        self.assertEqual(len(result), 1)
        action, prob = result[0]
        self.assertEqual(action, 0)
        self.assertEqual(prob, 1.0)


if __name__ == "__main__":
    unittest.main()
