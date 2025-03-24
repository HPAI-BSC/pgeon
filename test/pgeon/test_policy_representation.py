import unittest
from typing import Dict, Any, Tuple, cast, Optional, List

import networkx as nx
import numpy as np

from pgeon import GraphRepresentation, Predicate
from test.domain.test_env import State, TestingDiscretizer, TestingEnv
from pgeon.discretizer import StateRepresentation
from pgeon.policy_representation import Action


class TestPolicyRepresentation(unittest.TestCase):
    """
    Tests for the PolicyRepresentation and GraphRepresentation classes.
    """

    def setUp(self):
        """Set up test data before each test."""
        self.discretizer = TestingDiscretizer()

        # Create states and actions for testing
        self.state0 = StateRepresentation((Predicate(State, [State.ZERO]),))
        self.state1 = StateRepresentation((Predicate(State, [State.ONE]),))
        self.state2 = StateRepresentation((Predicate(State, [State.TWO]),))
        self.state3 = StateRepresentation((Predicate(State, [State.THREE]),))

        # TestingEnv only supports action 0
        self.action0: Action = 0

        # Initialize a GraphRepresentation for testing
        self.representation = GraphRepresentation()

    def test_initialization(self):
        """Test initialization of policy representation."""
        self.assertIsInstance(self.representation.graph.nx_graph, nx.MultiDiGraph)
        self.assertEqual(len(self.representation.get_all_states()), 0)
        self.assertEqual(len(self.representation.get_all_transitions()), 0)

    def test_add_state(self):
        """Test adding states to the representation."""
        # Add a single state
        self.representation.add_state(self.state0, frequency=1, probability=0.25)

        # Verify state was added correctly
        self.assertTrue(self.representation.has_state(self.state0))
        self.assertEqual(len(self.representation.get_all_states()), 1)

        # Test with attributes
        state_attrs = self.representation.get_state_attributes("frequency")
        self.assertEqual(state_attrs[self.state0], 1)

        # Add multiple states
        self.representation.add_states_from(
            [self.state1, self.state2, self.state3], frequency=2, probability=0.25
        )

        # Verify states were added correctly
        self.assertTrue(self.representation.has_state(self.state1))
        self.assertTrue(self.representation.has_state(self.state2))
        self.assertTrue(self.representation.has_state(self.state3))
        self.assertEqual(len(self.representation.get_all_states()), 4)

        # Test with attributes
        state_attrs = self.representation.get_state_attributes("frequency")
        self.assertEqual(state_attrs[self.state1], 2)
        self.assertEqual(state_attrs[self.state2], 2)
        self.assertEqual(state_attrs[self.state3], 2)

    def test_add_transition(self):
        """Test adding transitions to the representation."""
        # Add states first
        self.representation.add_states_from(
            [self.state0, self.state1, self.state2, self.state3], frequency=1
        )

        # Add a single transition
        self.representation.add_transition(
            self.state0, self.state1, self.action0, frequency=5, probability=1.0
        )

        # Verify transition was added correctly
        self.assertTrue(self.representation.has_transition(self.state0, self.state1))
        self.assertTrue(
            self.representation.has_transition(self.state0, self.state1, self.action0)
        )
        self.assertEqual(len(self.representation.get_all_transitions()), 1)

        # Test with attributes
        transition_data = self.representation.get_transition_data(
            self.state0, self.state1, self.action0
        )
        self.assertEqual(transition_data["frequency"], 5)
        self.assertEqual(transition_data["probability"], 1.0)
        self.assertEqual(transition_data["action"], self.action0)

        # Add multiple transitions
        transitions: List[Tuple[StateRepresentation, StateRepresentation, Action]] = [
            (self.state1, self.state2, self.action0),
            (self.state2, self.state3, self.action0),
            (self.state3, self.state0, self.action0),
        ]
        self.representation.add_transitions_from(
            transitions, frequency=3, probability=0.75
        )

        # Verify transitions were added correctly
        self.assertTrue(
            self.representation.has_transition(self.state1, self.state2, self.action0)
        )
        self.assertTrue(
            self.representation.has_transition(self.state2, self.state3, self.action0)
        )
        self.assertTrue(
            self.representation.has_transition(self.state3, self.state0, self.action0)
        )
        self.assertEqual(len(self.representation.get_all_transitions()), 4)

        # Test with attributes
        transition_data = self.representation.get_transition_data(
            self.state1, self.state2, self.action0
        )
        self.assertEqual(transition_data["frequency"], 3)
        self.assertEqual(transition_data["probability"], 0.75)

    def test_get_possible_actions(self):
        """Test getting possible actions from a state."""
        # Setup states and transitions
        self.setup_test_graph()

        # Test with a state that has outgoing transitions
        actions = self.representation.get_possible_actions(self.state0)
        self.assertEqual(len(actions), 1)
        self.assertIn(self.action0, actions)

        # Test with a state that has no outgoing transitions - not applicable for our test env
        # All states have outgoing transitions in our cycle

        # Test with a state that doesn't exist
        nonexistent_state = StateRepresentation((Predicate(State, [State.ZERO]), Predicate(State, [State.ONE])))
        actions = self.representation.get_possible_actions(nonexistent_state)
        self.assertEqual(len(actions), 0)

    def test_get_possible_next_states(self):
        """Test getting possible next states from a state."""
        # Setup states and transitions
        self.setup_test_graph()

        # Test with a state and specific action
        next_states = self.representation.get_possible_next_states(
            self.state0, self.action0
        )
        self.assertEqual(len(next_states), 1)
        self.assertIn(self.state1, next_states)

        # Test with a state and no action specified (all actions)
        next_states = self.representation.get_possible_next_states(self.state0)
        self.assertEqual(len(next_states), 1)
        self.assertIn(self.state1, next_states)

        # Test with a state that doesn't exist
        nonexistent_state = StateRepresentation((Predicate(State, [State.ZERO]), Predicate(State, [State.ONE])))
        next_states = self.representation.get_possible_next_states(nonexistent_state)
        self.assertEqual(len(next_states), 0)

    def test_get_transitions_from_state(self):
        """Test getting transitions from a state, grouped by action."""
        # Setup states and transitions
        self.setup_test_graph()

        # Test with a state that has outgoing transitions
        transitions = self.representation.get_transitions_from_state(self.state0)
        self.assertEqual(len(transitions), 1)
        self.assertIn(self.action0, transitions)
        self.assertIn(self.state1, transitions[self.action0])

        # Test with a state that doesn't exist
        nonexistent_state = StateRepresentation((Predicate(State, [State.ZERO]), Predicate(State, [State.ONE])))
        transitions = self.representation.get_transitions_from_state(nonexistent_state)
        self.assertEqual(len(transitions), 0)

    def test_get_state_attributes(self):
        """Test getting and setting state attributes."""
        # Setup states with attributes
        self.representation.add_states_from(
            [self.state0, self.state1, self.state2, self.state3],
            frequency=1,
            probability=0.25,
        )

        # Test getting an attribute
        attrs = self.representation.get_state_attributes("frequency")
        self.assertEqual(len(attrs), 4)
        self.assertEqual(attrs[self.state0], 1)
        self.assertEqual(attrs[self.state1], 1)

        # Test setting attributes
        new_attrs = {self.state0: 10, self.state1: 20}
        self.representation.set_state_attributes(new_attrs, "frequency")

        # Verify attributes were updated
        attrs = self.representation.get_state_attributes("frequency")
        self.assertEqual(attrs[self.state0], 10)
        self.assertEqual(attrs[self.state1], 20)
        self.assertEqual(attrs[self.state2], 1)  # Unchanged

    def test_get_outgoing_transitions(self):
        """Test getting outgoing transitions from a state."""
        # Setup states and transitions
        self.setup_test_graph()

        # Test without data
        transitions = self.representation.get_outgoing_transitions(
            self.state0, include_data=False
        )
        self.assertEqual(len(transitions), 1)

        # Test with data
        transitions = self.representation.get_outgoing_transitions(
            self.state0, include_data=True
        )
        self.assertEqual(len(transitions), 1)
        for _, _, data in transitions:
            self.assertIn("action", data)
            self.assertIn("frequency", data)
            self.assertIn("probability", data)

    def test_clear(self):
        """Test clearing the representation."""
        # Setup states and transitions
        self.setup_test_graph()

        # Verify setup
        self.assertEqual(len(self.representation.get_all_states()), 4)
        self.assertEqual(len(self.representation.get_all_transitions()), 4)

        # Clear the representation
        self.representation.clear()

        # Verify clearing worked
        self.assertEqual(len(self.representation.get_all_states()), 0)
        self.assertEqual(len(self.representation.get_all_transitions()), 0)

    def test_backward_compatibility(self):
        """Test that the backward compatibility methods work correctly."""
        # Setup states and transitions
        self.setup_test_graph()

        # Test old methods against new methods
        self.assertEqual(
            len(self.representation.nodes()), len(self.representation.get_all_states())
        )
        self.assertEqual(
            len(self.representation.edges()),
            len(self.representation.get_all_transitions()),
        )
        self.assertEqual(
            len(self.representation.out_edges(self.state0)),
            len(self.representation.get_outgoing_transitions(self.state0)),
        )

        # Test node attributes
        old_attrs = self.representation.get_node_attributes("frequency")
        new_attrs = self.representation.get_state_attributes("frequency")
        self.assertEqual(old_attrs, new_attrs)

        # Test edge data
        old_data = self.representation.get_edge_data(
            self.state0, self.state1, self.action0
        )
        new_data = self.representation.get_transition_data(
            self.state0, self.state1, self.action0
        )
        self.assertEqual(old_data, new_data)

    def test_simulation_with_environment(self):
        """Test simulating a policy with the TestingEnv environment."""
        # Create a fresh environment
        env = TestingEnv()

        # Create a representation and populate it
        self.representation.clear()

        # Add states
        self.representation.add_states_from(
            [self.state0, self.state1, self.state2, self.state3],
            frequency=1,
            probability=0.25,
        )

        # Add transitions that match the environment behavior
        transitions: List[Tuple[StateRepresentation, StateRepresentation, Action]] = [
            (self.state0, self.state1, self.action0),
            (self.state1, self.state2, self.action0),
            (self.state2, self.state3, self.action0),
            (self.state3, self.state0, self.action0),
        ]
        self.representation.add_transitions_from(
            transitions, frequency=1, probability=1.0
        )

        # Test simulation
        obs, _ = env.reset()

        # Check that the initial observation matches our discretization
        initial_state = StateRepresentation(self.discretizer.discretize(obs))
        self.assertEqual(initial_state, self.state0)

        # Make one step and verify transition
        state = initial_state
        next_obs, _, _, _, _ = env.step(self.action0)
        next_state = StateRepresentation(self.discretizer.discretize(next_obs))

        # Check that the state transition is as expected
        self.assertTrue(
            self.representation.has_transition(state, next_state, self.action0)
        )

        # Get possible next states from representation
        possible_next_states = self.representation.get_possible_next_states(
            state, self.action0
        )
        self.assertIn(next_state, possible_next_states)

    def setup_test_graph(self):
        """Helper method to set up a test graph for multiple tests."""
        # Clear any existing data
        self.representation.clear()

        # Add states
        self.representation.add_states_from(
            [self.state0, self.state1, self.state2, self.state3],
            frequency=1,
            probability=0.25,
        )

        # Add transitions - in TestingEnv, we can only do action 0 and it cycles through states
        transitions: List[Tuple[StateRepresentation, StateRepresentation, Action]] = [
            (self.state0, self.state1, self.action0),
            (self.state1, self.state2, self.action0),
            (self.state2, self.state3, self.action0),
            (self.state3, self.state0, self.action0),
        ]
        self.representation.add_transitions_from(
            transitions, frequency=1, probability=1.0
        )


if __name__ == "__main__":
    unittest.main()
