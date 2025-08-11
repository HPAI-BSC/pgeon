import unittest
from pathlib import Path
from test.domain.test_env import DummyState, TestingDiscretizer, TestingEnv
from typing import Dict, List, Tuple

from pgeon import GraphRepresentation, Predicate
from pgeon.discretizer import (
    PredicateBasedState,
    State,
    Transition,
)
from pgeon.policy_representation import Action


class TestPolicyRepresentation(unittest.TestCase):
    """Tests for the PolicyRepresentation and GraphRepresentation classes."""

    def setUp(self):
        """Set up test data before each test."""
        self.discretizer = TestingDiscretizer()

        # Create states and actions for testing
        self.state0 = PredicateBasedState((Predicate(DummyState.ZERO),))
        self.state1 = PredicateBasedState((Predicate(DummyState.ONE),))
        self.state2 = PredicateBasedState((Predicate(DummyState.TWO),))
        self.state3 = PredicateBasedState((Predicate(DummyState.THREE),))

        self.action0: Action = 0
        self.representation = GraphRepresentation()

        self.tmp_dir = Path(".tmp")

    def tearDown(self):
        """Tear down test data after each test."""
        if self.tmp_dir.exists():
            for file in self.tmp_dir.iterdir():
                file.unlink()
            self.tmp_dir.rmdir()

    def test_initialization(self):
        """Test initialization of policy representation."""
        self.assertEqual(len(self.representation.get_all_states()), 0)
        self.assertEqual(len(self.representation.get_all_transitions()), 0)

    def test_add_state(self):
        """Test adding states to the representation."""
        self.representation.add_state(self.state0, frequency=1, probability=0.25)
        self.assertTrue(self.representation.has_state(self.state0))
        self.assertFalse(self.representation.has_state(self.state1))
        self.assertFalse(self.representation.has_state(self.state2))

    def test_add_states_from(self):
        """Test adding multiple states to the representation."""
        self.representation.add_states_from(
            [self.state1, self.state2, self.state3], frequency=2, probability=0.25
        )

        self.assertTrue(self.representation.has_state(self.state1))
        self.assertTrue(self.representation.has_state(self.state2))
        self.assertTrue(self.representation.has_state(self.state3))
        self.assertEqual(len(self.representation.get_all_states()), 3)

        state_attrs = self.representation.get_state_attributes("frequency")
        self.assertEqual(state_attrs[self.state1], 2)
        self.assertEqual(state_attrs[self.state2], 2)
        self.assertEqual(state_attrs[self.state3], 2)

    def test_add_transition(self):
        """Test adding transitions to the representation."""
        self.representation.add_states_from(
            [self.state0, self.state1, self.state2, self.state3], frequency=1
        )

        self.representation.add_transition(
            self.state0,
            self.state1,
            Transition(action=self.action0, frequency=5, probability=1.0),
        )

        self.assertTrue(self.representation.has_transition(self.state0, self.state1))
        self.assertTrue(
            self.representation.has_transition(self.state0, self.state1, self.action0)
        )
        self.assertEqual(len(self.representation.get_all_transitions()), 1)

        transition_data = self.representation.get_transition_data(
            self.state0, self.state1, self.action0
        )
        self.assertEqual(transition_data.frequency, 5)
        self.assertEqual(transition_data.probability, 1.0)
        self.assertEqual(transition_data.action, self.action0)

        # Add multiple transitions
        transitions = [
            (self.state1, self.state2, self.action0),
            (self.state2, self.state3, self.action0),
            (self.state3, self.state0, self.action0),
        ]
        self.representation.add_transitions_from(
            transitions, frequency=3, probability=0.75
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
        self.assertEqual(len(self.representation.get_all_transitions()), 4)

        transition_data = self.representation.get_transition_data(
            self.state1, self.state2, self.action0
        )
        self.assertEqual(transition_data.frequency, 3)
        self.assertEqual(transition_data.probability, 0.75)

        # Test updating an existing transition
        self.representation.add_transition(
            self.state0,
            self.state1,
            Transition(action=self.action0, frequency=10, probability=0.9),
        )
        transition_data = self.representation.get_transition_data(
            self.state0, self.state1, self.action0
        )
        self.assertEqual(transition_data.frequency, 10)
        self.assertEqual(transition_data.probability, 0.9)

    def test_save_and_load_csv(self):
        """Test saving and loading a policy representation from CSV files."""
        nodes_path = self.tmp_dir / "test_nodes.csv"
        edges_path = self.tmp_dir / "test_edges.csv"
        self.tmp_dir.mkdir(exist_ok=True)
        if nodes_path.exists():
            nodes_path.unlink()
        if edges_path.exists():
            edges_path.unlink()
        self.maxDiff = None
        self.representation.add_states_from(
            [self.state0, self.state1, self.state2, self.state3],
            frequency=1,
            probability=0.25,
        )
        self.representation.add_transition(
            self.state0,
            self.state1,
            Transition(action=self.action0, frequency=5, probability=1.0),
        )

        self.representation.save_csv(self.discretizer, nodes_path, edges_path)
        loaded_representation = GraphRepresentation.load_csv(
            "networkx", self.discretizer, nodes_path, edges_path
        )

        self.assertEqual(len(loaded_representation.get_all_states()), 4)
        self.assertEqual(len(loaded_representation.get_all_transitions()), 1)
        self.assertEqual(
            loaded_representation.get_state_attributes("frequency"),
            self.representation.get_state_attributes("frequency"),
        )
        self.assertEqual(
            loaded_representation.get_state_attributes("probability"),
            self.representation.get_state_attributes("probability"),
        )
        self.assertEqual(
            loaded_representation.get_transition_data(
                self.state0, self.state1, self.action0
            ),
            self.representation.get_transition_data(
                self.state0, self.state1, self.action0
            ),
        )

        # Test with multiple transitions
        self.representation.clear()
        self.setup_test_graph()
        self.representation.save_csv(self.discretizer, nodes_path, edges_path)
        loaded_representation = GraphRepresentation.load_csv(
            "networkx", self.discretizer, nodes_path, edges_path
        )
        self.assertEqual(len(loaded_representation.get_all_states()), 4)
        self.assertEqual(len(loaded_representation.get_all_transitions()), 4)

    def test_save_and_load_gram(self):
        """Test saving and loading a policy representation from gram files."""
        gram_path = self.tmp_dir / "test.gram"
        self.tmp_dir.mkdir(exist_ok=True)
        if gram_path.exists():
            gram_path.unlink()
        self.maxDiff = None
        self.representation.add_states_from(
            [self.state0, self.state1, self.state2, self.state3],
            frequency=1,
            probability=0.25,
        )
        self.representation.add_transition(
            self.state0,
            self.state1,
            Transition(action=self.action0, frequency=5, probability=1.0),
        )

        self.representation.save_gram(self.discretizer, gram_path)
        loaded_representation = GraphRepresentation.load_gram(
            "networkx", self.discretizer, gram_path
        )

        self.assertEqual(len(loaded_representation.get_all_states()), 4)
        self.assertEqual(len(loaded_representation.get_all_transitions()), 1)
        self.assertEqual(
            loaded_representation.get_state_attributes("frequency"),
            self.representation.get_state_attributes("frequency"),
        )
        self.assertEqual(
            loaded_representation.get_state_attributes("probability"),
            self.representation.get_state_attributes("probability"),
        )
        self.assertEqual(
            loaded_representation.get_transition_data(
                self.state0, self.state1, self.action0
            ),
            self.representation.get_transition_data(
                self.state0, self.state1, self.action0
            ),
        )

        # Test with multiple transitions
        self.representation.clear()
        self.setup_test_graph()
        self.representation.save_gram(self.discretizer, gram_path)
        loaded_representation = GraphRepresentation.load_gram(
            "networkx", self.discretizer, gram_path
        )
        self.assertEqual(len(loaded_representation.get_all_states()), 4)
        self.assertEqual(len(loaded_representation.get_all_transitions()), 4)

    def test_get_possible_transitions(self):
        """Test getting possible actions from a state."""
        self.setup_test_graph()

        actions = self.representation.get_possible_transitions(self.state0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, self.action0)
        self.assertEqual(actions[0].probability, 1.0)

        nonexistent_state = PredicateBasedState(
            (Predicate(DummyState.ZERO), Predicate(DummyState.ONE))
        )
        actions = self.representation.get_possible_transitions(nonexistent_state)
        self.assertEqual(len(actions), 0)

        # Test with multiple actions
        self.representation.add_transition(
            self.state0, self.state2, Transition(action=1, probability=0.5)
        )
        self.representation.add_transition(
            self.state0,
            self.state3,
            Transition(action=self.action0, probability=0.5),
        )
        transitions = self.representation.get_possible_transitions(self.state0)
        self.assertEqual(len(transitions), 3)

    def test_get_possible_next_states(self):
        """Test getting possible next states from a state."""
        self.setup_test_graph()

        next_states = self.representation.get_possible_next_states(
            self.state0, self.action0
        )
        self.assertEqual(len(next_states), 1)
        self.assertIn(self.state1, next_states)

        next_states = self.representation.get_possible_next_states(self.state0)
        self.assertEqual(len(next_states), 1)
        self.assertIn(self.state1, next_states)

        nonexistent_state = PredicateBasedState(
            (Predicate(DummyState.ZERO), Predicate(DummyState.ONE))
        )
        next_states = self.representation.get_possible_next_states(nonexistent_state)
        self.assertEqual(len(next_states), 0)

        # Test with multiple next states
        self.representation.add_transition(
            self.state0, self.state2, Transition(action=self.action0)
        )
        next_states = self.representation.get_possible_next_states(
            self.state0, self.action0
        )
        self.assertEqual(len(next_states), 2)
        self.assertIn(self.state1, next_states)
        self.assertIn(self.state2, next_states)

    def test_get_transitions_from_state(self):
        """Test getting transitions from a state, grouped by action."""
        self.setup_test_graph()

        transitions = self.representation.get_transitions_from_state(self.state0)
        self.assertEqual(len(transitions), 1)
        self.assertIn(self.action0, transitions)
        self.assertIn(self.state1, transitions[self.action0])

        nonexistent_state = PredicateBasedState(
            (Predicate(DummyState.ZERO), Predicate(DummyState.ONE))
        )
        transitions = self.representation.get_transitions_from_state(nonexistent_state)
        self.assertEqual(len(transitions), 0)

        # Test with multiple transitions
        self.representation.add_transition(
            self.state0, self.state2, Transition(action=1)
        )
        transitions = self.representation.get_transitions_from_state(self.state0)
        self.assertEqual(len(transitions), 2)
        self.assertIn(self.action0, transitions)
        self.assertIn(1, transitions)
        self.assertIn(self.state1, transitions[self.action0])
        self.assertIn(self.state2, transitions[1])

    def test_get_state_attributes(self):
        """Test getting and setting state attributes."""
        self.representation.add_states_from(
            [self.state0, self.state1, self.state2, self.state3],
            frequency=1,
            probability=0.25,
        )

        attrs = self.representation.get_state_attributes("frequency")
        self.assertEqual(len(attrs), 4)
        self.assertEqual(attrs[self.state0], 1)
        self.assertEqual(attrs[self.state1], 1)

        new_attrs: Dict[State, int] = {self.state0: 10, self.state1: 20}
        self.representation.set_state_attributes(new_attrs, "frequency")

        attrs = self.representation.get_state_attributes("frequency")
        self.assertEqual(attrs[self.state0], 10)
        self.assertEqual(attrs[self.state1], 20)
        self.assertEqual(attrs[self.state2], 1)  # Unchanged

    def test_get_outgoing_transitions(self):
        """Test getting outgoing transitions from a state."""
        self.setup_test_graph()
        self.representation.add_transition(
            self.state0,
            self.state2,
            Transition(action=self.action0, frequency=1, probability=1.0),
        )
        transitions = self.representation.get_outgoing_transitions(self.state0)
        self.assertEqual(len(transitions), 2)

    def test_clear(self):
        """Test clearing the representation."""
        self.setup_test_graph()

        self.assertEqual(len(self.representation.get_all_states()), 4)
        self.assertEqual(len(self.representation.get_all_transitions()), 4)

        self.representation.clear()

        self.assertEqual(len(self.representation.get_all_states()), 0)
        self.assertEqual(len(self.representation.get_all_transitions()), 0)

    def test_simulation_with_environment(self):
        """Test simulating a policy with the TestingEnv environment."""
        env = TestingEnv()
        self.representation.clear()

        self.representation.add_states_from(
            [self.state0, self.state1, self.state2, self.state3],
            frequency=1,
            probability=0.25,
        )

        transitions: List[Tuple[State, State, Action]] = [
            (self.state0, self.state1, self.action0),
            (self.state1, self.state2, self.action0),
            (self.state2, self.state3, self.action0),
            (self.state3, self.state0, self.action0),
        ]
        self.representation.add_transitions_from(
            transitions, frequency=1, probability=1.0
        )

        obs, _ = env.reset()
        initial_state = PredicateBasedState(self.discretizer.discretize(obs))
        self.assertEqual(initial_state, self.state0)

        state = initial_state
        next_obs, _, _, _, _ = env.step(self.action0)
        next_state = PredicateBasedState(self.discretizer.discretize(next_obs))

        self.assertTrue(
            self.representation.has_transition(state, next_state, self.action0)
        )

        possible_next_states = self.representation.get_possible_next_states(
            state, self.action0
        )
        self.assertIn(next_state, possible_next_states)

    def test_add_trajectory(self):
        """Test adding a trajectory to the representation."""
        self.representation.clear()

        trajectory = [
            (self.state0, self.action0, self.state1),
            (self.state1, self.action0, self.state2),
            (self.state2, self.action0, self.state3),
            (self.state3, self.action0, self.state0),
        ]

        self.representation.add_trajectory(trajectory)

        self.assertEqual(len(self.representation.get_all_states()), 4)
        self.assertEqual(len(self.representation.get_all_transitions()), 4)

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

        transition_data = self.representation.get_transition_data(
            self.state0, self.state1, self.action0
        )
        self.assertEqual(transition_data.frequency, 1)

        # Test with a trajectory of integers
        self.representation.clear()
        trajectory_int = [0, 0, 1, 0, 2, 0, 3, 0, 0]
        self.representation.add_trajectory(trajectory_int)
        self.assertEqual(len(self.representation.get_all_states()), 4)
        self.assertEqual(len(self.representation.get_all_transitions()), 4)
        self.assertTrue(self.representation.has_transition(0, 1, 0))
        self.assertTrue(self.representation.has_transition(1, 2, 0))
        self.assertTrue(self.representation.has_transition(2, 3, 0))
        self.assertTrue(self.representation.has_transition(3, 0, 0))
        transition_data = self.representation.get_transition_data(0, 1, 0)
        self.assertEqual(transition_data.frequency, 1)

    def setup_test_graph(self):
        """Helper method to set up a test graph for multiple tests."""
        self.representation.clear()

        self.representation.add_states_from(
            [self.state0, self.state1, self.state2, self.state3],
            frequency=1,
            probability=0.25,
        )

        transitions: List[Tuple[State, State, Action]] = [
            (self.state0, self.state1, self.action0),
            (self.state1, self.state2, self.action0),
            (self.state2, self.state3, self.action0),
            (self.state3, self.state0, self.action0),
        ]
        self.representation.add_transitions_from(
            transitions, frequency=1, probability=1.0
        )

    def test_get_possible_transitions_single_transition(self):
        """Test getting possible actions from a state with a single transition."""
        self.representation.add_states_from([self.state0, self.state1])
        self.representation.add_transition(
            self.state0,
            self.state1,
            Transition(action=self.action0, frequency=1, probability=1.0),
        )
        actions = self.representation.get_possible_transitions(self.state0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, self.action0)
        self.assertEqual(actions[0].probability, 1.0)


if __name__ == "__main__":
    unittest.main()
