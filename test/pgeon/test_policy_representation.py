import unittest
from pathlib import Path
from test.domain.test_env import DummyState, TestingDiscretizer, TestingEnv

from pgeon import GraphRepresentation, Predicate
from pgeon.discretizer import (
    PredicateBasedState,
    State,
    StateMetadata,
    Transition,
    TransitionData,
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
        self.action1: Action = 1
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
        self.assertEqual(len(list(self.representation.states)), 0)
        self.assertEqual(len(list(self.representation.transitions)), 0)

    def test_add_state(self):
        """Test adding states to the representation."""
        self.representation.states[self.state0] = StateMetadata(
            frequency=1, probability=0.25
        )
        self.assertTrue(self.state0 in self.representation.states)
        self.assertEqual(
            self.representation.states[self.state0].metadata,
            StateMetadata(frequency=1, probability=0.25),
        )
        self.assertFalse(self.state1 in self.representation.states)
        self.assertFalse(self.state2 in self.representation.states)

    def test_add_state_no_metadata(self):
        """Test adding states to the representation."""
        self.representation.states[self.state0] = StateMetadata()
        self.assertTrue(self.state0 in self.representation.states)
        self.assertEqual(
            self.representation.states[self.state0].metadata, StateMetadata()
        )
        self.assertFalse(self.state1 in self.representation.states)
        self.assertFalse(self.state2 in self.representation.states)

    def test_add_state_with_custom_state_metadata_class(self):
        """Test initialization of policy representation with a custom state metadata class."""

        class CustomStateMetadata(StateMetadata):
            custom_attribute: int = 0

        self.representation_with_custom_state_metadata = GraphRepresentation(
            state_metadata_class=CustomStateMetadata
        )
        self.representation_with_custom_state_metadata.states[self.state0] = (
            CustomStateMetadata(custom_attribute=1)
        )
        self.assertEqual(
            len(list(self.representation_with_custom_state_metadata.states)), 1
        )
        self.assertEqual(
            len(list(self.representation_with_custom_state_metadata.transitions)),
            0,
        )
        self.assertEqual(
            self.representation_with_custom_state_metadata.states[
                self.state0
            ].metadata.custom_attribute,
            1,
        )

    def test_add_states_from(self):
        """Test adding multiple states to the representation."""
        for state in [self.state1, self.state2, self.state3]:
            self.representation.states[state] = StateMetadata(
                frequency=2, probability=0.25
            )

        self.assertTrue(self.state1 in self.representation.states)
        self.assertTrue(self.state2 in self.representation.states)
        self.assertTrue(self.state3 in self.representation.states)
        self.assertEqual(len(list(self.representation.states)), 3)

        state_metadata = self.representation.states.metadata
        self.assertEqual(state_metadata[self.state1].frequency, 2)
        self.assertEqual(state_metadata[self.state2].frequency, 2)
        self.assertEqual(state_metadata[self.state3].frequency, 2)

    def test_add_transition(self):
        """Test adding transitions to the representation."""
        for state in [self.state0, self.state1, self.state2, self.state3]:
            self.representation.states[state] = StateMetadata(frequency=1)

        self.representation.transitions[self.state0][self.state1] = Transition(
            action=self.action0, frequency=5, probability=1.0
        )

        self.assertTrue(
            (self.state0, self.state1, self.action0) in self.representation.transitions
        )
        self.assertEqual(len(list(self.representation.transitions)), 1)

        transition_data = self.representation.transitions[self.state0][self.state1]
        self.assertEqual(transition_data.frequency, 5)
        self.assertEqual(transition_data.probability, 1.0)
        self.assertEqual(transition_data.action, self.action0)

        # Add multiple transitions
        self.representation.transitions[self.state1][self.state2] = Transition(
            action=self.action0, frequency=3, probability=0.75
        )
        self.representation.transitions[self.state2][self.state3] = Transition(
            action=self.action0, frequency=3, probability=0.75
        )
        self.representation.transitions[self.state3][self.state0] = Transition(
            action=self.action0, frequency=3, probability=0.75
        )

        self.assertTrue(
            (self.state1, self.state2, self.action0) in self.representation.transitions
        )
        self.assertTrue(
            (self.state2, self.state3, self.action0) in self.representation.transitions
        )
        self.assertTrue(
            (self.state3, self.state0, self.action0) in self.representation.transitions
        )
        self.assertEqual(len(list(self.representation.transitions)), 4)

        transition_data = self.representation.transitions[self.state1][self.state2]
        self.assertEqual(transition_data.frequency, 3)
        self.assertEqual(transition_data.probability, 0.75)

        # Test updating an existing transition
        self.representation.transitions[self.state0][self.state1] = Transition(
            action=self.action0, frequency=10, probability=0.9
        )
        transition_data = self.representation.transitions[self.state0][self.state1]
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
        for state in [self.state0, self.state1, self.state2, self.state3]:
            self.representation.states[state] = StateMetadata(
                frequency=1, probability=0.25
            )
        self.representation.transitions[self.state0][self.state1] = Transition(
            action=self.action0, frequency=5, probability=1.0
        )

        self.representation.save_csv(self.discretizer, nodes_path, edges_path)
        loaded_representation = GraphRepresentation.load_csv(
            "networkx", self.discretizer, nodes_path, edges_path
        )

        self.assertEqual(len(list(loaded_representation.states)), 4)
        self.assertEqual(len(list(loaded_representation.transitions)), 1)
        self.assertEqual(
            loaded_representation.states.metadata,
            self.representation.states.metadata,
        )
        self.assertEqual(
            loaded_representation.states.metadata,
            self.representation.states.metadata,
        )
        self.assertEqual(
            loaded_representation.transitions[self.state0][self.state1],
            self.representation.transitions[self.state0][self.state1],
        )

        # Test with multiple transitions
        self.representation.clear()
        self.setup_test_graph()
        self.representation.save_csv(self.discretizer, nodes_path, edges_path)
        loaded_representation = GraphRepresentation.load_csv(
            "networkx", self.discretizer, nodes_path, edges_path
        )
        self.assertEqual(len(list(loaded_representation.states)), 4)
        self.assertEqual(len(list(loaded_representation.transitions)), 4)

    def test_save_and_load_gram(self):
        """Test saving and loading a policy representation from gram files."""
        gram_path = self.tmp_dir / "test.gram"
        self.tmp_dir.mkdir(exist_ok=True)
        if gram_path.exists():
            gram_path.unlink()
        self.maxDiff = None
        for state in [self.state0, self.state1, self.state2, self.state3]:
            self.representation.states[state] = StateMetadata(
                frequency=1, probability=0.25
            )
        self.representation.transitions[self.state0][self.state1] = Transition(
            action=self.action0, frequency=5, probability=1.0
        )

        self.representation.save_gram(self.discretizer, gram_path)
        loaded_representation = GraphRepresentation.load_gram(
            "networkx", self.discretizer, gram_path
        )

        self.assertEqual(len(list(loaded_representation.states)), 4)
        self.assertEqual(len(list(loaded_representation.transitions)), 1)
        self.assertEqual(
            loaded_representation.states.metadata,
            self.representation.states.metadata,
        )
        self.assertEqual(
            loaded_representation.states.metadata,
            self.representation.states.metadata,
        )
        self.assertEqual(
            loaded_representation.transitions[self.state0][self.state1],
            self.representation.transitions[self.state0][self.state1],
        )

        # Test with multiple transitions
        self.representation.clear()
        self.setup_test_graph()
        self.representation.save_gram(self.discretizer, gram_path)
        loaded_representation = GraphRepresentation.load_gram(
            "networkx", self.discretizer, gram_path
        )
        self.assertEqual(len(list(loaded_representation.states)), 4)
        self.assertEqual(len(list(loaded_representation.transitions)), 4)

    def test_get_possible_transitions(self):
        """Test getting possible actions from a state."""
        self.setup_test_graph()

        actions = self.representation._get_possible_transitions(self.state0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, self.action0)
        self.assertEqual(actions[0].probability, 1.0)

        nonexistent_state = PredicateBasedState(
            (
                Predicate(DummyState.ZERO),
                Predicate(DummyState.ONE),
            )
        )
        actions = self.representation._get_possible_transitions(nonexistent_state)
        self.assertEqual(len(actions), 0)

        # Test with multiple actions
        self.representation.transitions[self.state0][self.state2] = Transition(
            action=1, probability=0.5
        )
        self.representation.transitions[self.state0][self.state3] = Transition(
            action=self.action0, probability=0.5
        )
        transitions = self.representation._get_possible_transitions(self.state0)
        self.assertEqual(len(transitions), 3)

    def test_get_possible_next_states(self):
        """Test getting possible next states from a state."""
        self.setup_test_graph()

        next_states = self.representation._get_possible_next_states(
            self.state0, self.action0
        )
        self.assertEqual(len(next_states), 1)
        self.assertIn(self.state1, next_states)

        next_states = self.representation._get_possible_next_states(self.state0)
        self.assertEqual(len(next_states), 1)
        self.assertIn(self.state1, next_states)

        nonexistent_state = PredicateBasedState(
            (
                Predicate(DummyState.ZERO),
                Predicate(DummyState.ONE),
            )
        )
        next_states = self.representation._get_possible_next_states(nonexistent_state)
        self.assertEqual(len(next_states), 0)

        # Test with multiple next states
        self.representation.transitions[self.state0][self.state2] = Transition(
            action=self.action0
        )
        next_states = self.representation._get_possible_next_states(
            self.state0, self.action0
        )
        self.assertEqual(len(next_states), 2)
        self.assertIn(self.state1, next_states)
        self.assertIn(self.state2, next_states)

    def test_get_transitions_from_state(self):
        """Test getting transitions from a state, grouped by action."""
        self.setup_test_graph()

        transitions = self.representation._get_transitions_from_state(self.state0)
        self.assertEqual(len(transitions), 1)
        self.assertIn(self.action0, transitions)
        self.assertIn(self.state1, transitions[self.action0])

        nonexistent_state = PredicateBasedState(
            (
                Predicate(DummyState.ZERO),
                Predicate(DummyState.ONE),
            )
        )
        transitions = self.representation._get_transitions_from_state(nonexistent_state)
        self.assertEqual(len(transitions), 0)

        # Test with multiple transitions
        self.representation.transitions[self.state0][self.state2] = Transition(action=1)
        transitions = self.representation._get_transitions_from_state(self.state0)
        self.assertEqual(len(transitions), 2)
        self.assertIn(self.action0, transitions)
        self.assertIn(1, transitions)
        self.assertIn(self.state1, transitions[self.action0])
        self.assertIn(self.state2, transitions[1])

    def test_get_state_metadata(self):
        """Test getting and setting state attributes."""
        for state in [self.state0, self.state1, self.state2, self.state3]:
            self.representation.states[state] = StateMetadata(
                frequency=1, probability=0.25
            )

        state_to_state_metadata = self.representation.states.metadata
        self.assertEqual(len(state_to_state_metadata), 4)
        self.assertEqual(state_to_state_metadata[self.state0].frequency, 1)
        self.assertEqual(state_to_state_metadata[self.state1].frequency, 1)

        updated_state_metadata = {
            state: state_to_state_metadata[state].model_copy(update={"frequency": 10})
            for state, state_metadata in state_to_state_metadata.items()
            if state == self.state0 or state == self.state1
        }
        self.representation._set_state_metadata(updated_state_metadata)

        state_metadata = self.representation.states.metadata
        self.assertEqual(state_metadata[self.state0].frequency, 10)
        self.assertEqual(state_metadata[self.state1].frequency, 10)
        self.assertEqual(state_metadata[self.state2].frequency, 1)  # Unchanged

    def test_get_outgoing_transitions(self):
        """Test getting outgoing transitions from a state."""
        self.setup_test_graph()
        self.representation.transitions[self.state0][self.state2] = Transition(
            action=self.action0, frequency=1, probability=1.0
        )
        transitions = self.representation._get_outgoing_transitions(self.state0)
        self.assertEqual(len(transitions), 2)

    def test_clear(self):
        """Test clearing the representation."""
        self.setup_test_graph()

        self.assertEqual(len(list(self.representation.states)), 4)
        self.assertEqual(len(list(self.representation.transitions)), 4)

        self.representation.clear()

        self.assertEqual(len(list(self.representation.states)), 0)
        self.assertEqual(len(list(self.representation.transitions)), 0)

    def test_simulation_with_environment(self):
        """Test simulating a policy with the TestingEnv environment."""
        env = TestingEnv()
        self.representation.clear()

        for state in [self.state0, self.state1, self.state2, self.state3]:
            self.representation.states[state] = StateMetadata(
                frequency=1, probability=0.25
            )

        self.representation.transitions[self.state0][self.state1] = Transition(
            action=self.action0, frequency=1, probability=1.0
        )
        self.representation.transitions[self.state1][self.state2] = Transition(
            action=self.action0, frequency=1, probability=1.0
        )
        self.representation.transitions[self.state2][self.state3] = Transition(
            action=self.action0, frequency=1, probability=1.0
        )
        self.representation.transitions[self.state3][self.state0] = Transition(
            action=self.action0, frequency=1, probability=1.0
        )

        obs, _ = env.reset()
        initial_state = PredicateBasedState(self.discretizer.discretize(obs))
        self.assertEqual(initial_state, self.state0)

        state = initial_state
        next_obs, _, _, _, _ = env.step(self.action0)
        next_state = PredicateBasedState(self.discretizer.discretize(next_obs))

        self.assertTrue(
            (state, next_state, self.action0) in self.representation.transitions
        )

        possible_next_states = self.representation._get_possible_next_states(
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

        self.assertEqual(len(list(self.representation.states)), 4)
        self.assertEqual(len(list(self.representation.transitions)), 4)

        self.assertTrue(
            (self.state0, self.state1, self.action0) in self.representation.transitions
        )
        self.assertTrue(
            (self.state1, self.state2, self.action0) in self.representation.transitions
        )
        self.assertTrue(
            (self.state2, self.state3, self.action0) in self.representation.transitions
        )
        self.assertTrue(
            (self.state3, self.state0, self.action0) in self.representation.transitions
        )

        transition_data = self.representation.transitions[self.state0][self.state1]
        self.assertEqual(transition_data.frequency, 1)

        # Test with a trajectory of integers
        self.representation.clear()
        trajectory_int = [0, 0, 1, 0, 2, 0, 3, 0, 0]
        self.representation.add_trajectory(trajectory_int)
        self.assertEqual(len(list(self.representation.states)), 4)
        self.assertEqual(len(list(self.representation.transitions)), 4)
        self.assertTrue((0, 1, 0) in self.representation.transitions)
        self.assertTrue((1, 2, 0) in self.representation.transitions)
        self.assertTrue((2, 3, 0) in self.representation.transitions)
        self.assertTrue((3, 0, 0) in self.representation.transitions)
        transition_data = self.representation.transitions[0][1]
        self.assertEqual(transition_data.frequency, 1)

    def setup_test_graph(self):
        """Helper method to set up a test graph for multiple tests."""
        self.representation.clear()

        for state in [self.state0, self.state1, self.state2, self.state3]:
            self.representation.states[state] = StateMetadata(
                frequency=1, probability=0.25
            )

        self.representation.transitions[self.state0][self.state1] = Transition(
            action=self.action0, frequency=1, probability=1.0
        )
        self.representation.transitions[self.state1][self.state2] = Transition(
            action=self.action0, frequency=1, probability=1.0
        )
        self.representation.transitions[self.state2][self.state3] = Transition(
            action=self.action0, frequency=1, probability=1.0
        )
        self.representation.transitions[self.state3][self.state0] = Transition(
            action=self.action0, frequency=1, probability=1.0
        )

    def test_get_possible_transitions_single_transition(self):
        """Test getting possible actions from a state with a single transition."""
        self.representation.states[self.state0] = StateMetadata()
        self.representation.states[self.state1] = StateMetadata()
        self.representation.transitions[self.state0][self.state1] = Transition(
            action=self.action0, frequency=1, probability=1.0
        )
        actions = self.representation._get_possible_transitions(self.state0)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, self.action0)
        self.assertEqual(actions[0].probability, 1.0)

    def test_getitem_setitem_methods(self):
        """Test the __getitem__ and __setitem__ methods of the graph backend."""
        # Add a state first
        self.representation.states[self.state0] = StateMetadata(
            frequency=5, probability=0.25
        )

        # Test __getitem__ - accessing state metadata
        state_metadata = self.representation.graph[self.state0]
        self.assertEqual(state_metadata.frequency, 5)
        self.assertEqual(state_metadata.probability, 0.25)

        # Test __setitem__ - updating state metadata
        new_metadata = StateMetadata(frequency=10, probability=0.5)
        self.representation.graph[self.state0] = new_metadata

        # Verify the update worked
        updated_metadata = self.representation.graph[self.state0]
        self.assertEqual(updated_metadata.frequency, 10)
        self.assertEqual(updated_metadata.probability, 0.5)

        # Verify the change is reflected in the main representation
        representation_metadata = self.representation.states[self.state0].metadata
        self.assertEqual(representation_metadata.frequency, 10)
        self.assertEqual(representation_metadata.probability, 0.5)

    def test_getitem_setitem_with_custom_metadata_class(self):
        """Test the __getitem__ and __setitem__ methods with custom state metadata class."""

        class CustomStateMetadata(StateMetadata):
            custom_attribute: int = 0

        representation_with_custom = GraphRepresentation(
            state_metadata_class=CustomStateMetadata
        )

        # Add a state with custom metadata
        custom_metadata = CustomStateMetadata(
            frequency=3, probability=0.3, custom_attribute=42
        )
        representation_with_custom.states[self.state0] = custom_metadata

        # Test __getitem__ with custom metadata
        retrieved_metadata = representation_with_custom.graph[self.state0]
        self.assertEqual(retrieved_metadata.frequency, 3)
        self.assertEqual(retrieved_metadata.probability, 0.3)
        self.assertEqual(retrieved_metadata.custom_attribute, 42)

        # Test __setitem__ with custom metadata
        new_custom_metadata = CustomStateMetadata(
            frequency=7, probability=0.7, custom_attribute=99
        )
        representation_with_custom.graph[self.state0] = new_custom_metadata

        # Verify the update worked
        updated_metadata = representation_with_custom.graph[self.state0]
        self.assertEqual(updated_metadata.frequency, 7)
        self.assertEqual(updated_metadata.probability, 0.7)
        self.assertEqual(updated_metadata.custom_attribute, 99)

    def test_getitem_nonexistent_state(self):
        """Test __getitem__ with a state that doesn't exist raises KeyError."""
        # Don't add any states to the representation
        with self.assertRaises(KeyError):
            _ = self.representation.graph[self.state0]

    def test_setitem_new_state(self):
        """Test __setitem__ with a state that doesn't exist yet."""
        # Set metadata for a state that hasn't been added yet
        metadata = StateMetadata(frequency=15, probability=0.75)
        self.representation.graph[self.state0] = metadata

        # Verify the state was created and metadata was set
        self.assertTrue(self.state0 in self.representation.states)
        retrieved_metadata = self.representation.graph[self.state0]
        self.assertEqual(retrieved_metadata.frequency, 15)
        self.assertEqual(retrieved_metadata.probability, 0.75)

    def test_getitem_setitem_multiple_states(self):
        """Test __getitem__ and __setitem__ with multiple states."""
        # Add multiple states
        for state in [self.state0, self.state1, self.state2]:
            self.representation.states[state] = StateMetadata(
                frequency=1, probability=0.1
            )

        # Test __getitem__ for each state
        metadata0 = self.representation.graph[self.state0]
        metadata1 = self.representation.graph[self.state1]
        metadata2 = self.representation.graph[self.state2]

        self.assertEqual(metadata0.frequency, 1)
        self.assertEqual(metadata1.frequency, 1)
        self.assertEqual(metadata2.frequency, 1)

        # Test __setitem__ for each state with different values
        self.representation.graph[self.state0] = StateMetadata(
            frequency=10, probability=0.5
        )
        self.representation.graph[self.state1] = StateMetadata(
            frequency=20, probability=0.6
        )
        self.representation.graph[self.state2] = StateMetadata(
            frequency=30, probability=0.7
        )

        # Verify all updates worked
        updated_metadata0 = self.representation.graph[self.state0]
        updated_metadata1 = self.representation.graph[self.state1]
        updated_metadata2 = self.representation.graph[self.state2]

        self.assertEqual(updated_metadata0.frequency, 10)
        self.assertEqual(updated_metadata0.probability, 0.5)
        self.assertEqual(updated_metadata1.frequency, 20)
        self.assertEqual(updated_metadata1.probability, 0.6)
        self.assertEqual(updated_metadata2.frequency, 30)
        self.assertEqual(updated_metadata2.probability, 0.7)

    def test_fluent_api_states_iterator(self):
        """Test the fluent API states iterator."""
        self.setup_test_graph()

        # Test iteration over states
        states = list(self.representation.states)
        self.assertEqual(len(states), 4)
        self.assertIn(self.state0, states)
        self.assertIn(self.state1, states)
        self.assertIn(self.state2, states)
        self.assertIn(self.state3, states)

    def test_fluent_api_states_contains(self):
        """Test the fluent API states contains operator."""
        self.setup_test_graph()

        # Test state existence
        self.assertIn(self.state0, self.representation.states)
        self.assertIn(self.state1, self.representation.states)

        # Test non-existent state
        nonexistent_state = PredicateBasedState(
            (Predicate(DummyState.ZERO), Predicate(DummyState.ONE))
        )
        self.assertNotIn(nonexistent_state, self.representation.states)

    def test_fluent_api_states_metadata(self):
        """Test the fluent API states metadata property."""
        self.setup_test_graph()

        # Test getting all state metadata
        all_metadata = self.representation.states.metadata
        self.assertEqual(len(all_metadata), 4)
        self.assertEqual(all_metadata[self.state0].frequency, 1)
        self.assertEqual(all_metadata[self.state0].probability, 0.25)
        self.assertEqual(all_metadata[self.state1].frequency, 1)
        self.assertEqual(all_metadata[self.state1].probability, 0.25)

    def test_fluent_api_states_getitem(self):
        """Test the fluent API states getitem for individual state metadata."""
        self.setup_test_graph()

        # Test getting metadata for specific state
        state0_metadata = self.representation.states[self.state0].metadata
        self.assertEqual(state0_metadata.frequency, 1)
        self.assertEqual(state0_metadata.probability, 0.25)

        # Test getting predecessors
        predecessors = self.representation.states[self.state0].predecessors
        self.assertEqual(len(predecessors), 1)
        self.assertIn(self.state3, predecessors)

        # Test getting successors
        successors = self.representation.states[self.state0].successors
        self.assertEqual(len(successors), 1)
        self.assertIn(self.state1, successors)

    def test_fluent_api_states_setitem(self):
        """Test the fluent API states setitem for setting state metadata."""
        self.representation.clear()

        # Test setting state metadata
        metadata = StateMetadata(frequency=5, probability=0.5)
        self.representation.states[self.state0] = metadata

        # Verify the state was added with correct metadata
        self.assertIn(self.state0, self.representation.states)
        retrieved_metadata = self.representation.states[self.state0].metadata
        self.assertEqual(retrieved_metadata.frequency, 5)
        self.assertEqual(retrieved_metadata.probability, 0.5)

    def test_fluent_api_transitions_iterator(self):
        """Test the fluent API transitions iterator."""
        self.setup_test_graph()

        # Test iteration over all transitions
        transitions = list(self.representation.transitions)
        self.assertEqual(len(transitions), 4)

        # Verify all transitions are TransitionData objects with correct properties
        for transition_data in transitions:
            self.assertIsInstance(transition_data, TransitionData)
            self.assertIsInstance(transition_data.from_state, State)
            self.assertIsInstance(transition_data.to_state, State)
            self.assertIsInstance(transition_data.action, int)
            self.assertIsInstance(transition_data.probability, float)
            self.assertIsInstance(transition_data.frequency, int)

    def test_fluent_api_transitions_contains(self):
        """Test the fluent API transitions contains operator."""
        self.setup_test_graph()

        # Test existing transition
        self.assertIn(
            (self.state0, self.state1, self.action0), self.representation.transitions
        )
        self.assertIn(
            (self.state1, self.state2, self.action0), self.representation.transitions
        )

        # Test non-existent transition
        self.assertNotIn(
            (self.state0, self.state2, self.action0), self.representation.transitions
        )
        self.assertNotIn(
            (self.state0, self.state1, 999), self.representation.transitions
        )

    def test_fluent_api_transitions_getitem(self):
        """Test the fluent API transitions getitem for state-specific transitions."""
        self.setup_test_graph()

        # Test getting transitions for a specific state
        state0_transitions = list(self.representation.transitions[self.state0])
        self.assertEqual(len(state0_transitions), 1)

        transition_data = state0_transitions[0]
        self.assertEqual(transition_data.from_state, self.state0)
        self.assertEqual(transition_data.to_state, self.state1)
        self.assertEqual(transition_data.action, self.action0)
        self.assertEqual(transition_data.probability, 1.0)
        self.assertEqual(transition_data.frequency, 1)

    def test_fluent_api_transitions_getitem_specific_transition(self):
        """Test the fluent API transitions getitem for specific transition."""
        self.setup_test_graph()

        # Test getting specific transition
        transition = self.representation.transitions[self.state0][self.state1]
        self.assertEqual(transition.action, self.action0)
        self.assertEqual(transition.probability, 1.0)
        self.assertEqual(transition.frequency, 1)

        # Test getting non-existent transition raises KeyError
        with self.assertRaises(KeyError):
            _ = self.representation.transitions[self.state0][self.state2]

    def test_fluent_api_transitions_setitem(self):
        """Test the fluent API transitions setitem for setting transitions."""
        self.representation.clear()
        self.representation.states[self.state0] = StateMetadata()
        self.representation.states[self.state1] = StateMetadata()

        # Test setting a transition
        transition = Transition(action=self.action0, frequency=3, probability=0.8)
        self.representation.transitions[self.state0][self.state1] = transition

        # Verify the transition was added
        self.assertTrue(
            (self.state0, self.state1, self.action0) in self.representation.transitions
        )
        retrieved_transition = self.representation.transitions[self.state0][self.state1]
        self.assertEqual(retrieved_transition.frequency, 3)
        self.assertEqual(retrieved_transition.probability, 0.8)

    def test_fluent_api_complex_workflow(self):
        """Test a complex workflow using the fluent API."""
        self.representation.clear()

        # Add states using fluent API
        self.representation.states[self.state0] = StateMetadata(
            frequency=10, probability=0.4
        )
        self.representation.states[self.state1] = StateMetadata(
            frequency=8, probability=0.3
        )
        self.representation.states[self.state2] = StateMetadata(
            frequency=6, probability=0.2
        )
        self.representation.states[self.state3] = StateMetadata(
            frequency=4, probability=0.1
        )

        # Add transitions using fluent API
        self.representation.transitions[self.state0][self.state1] = Transition(
            action=0, frequency=5, probability=0.8
        )
        self.representation.transitions[self.state0][self.state2] = Transition(
            action=1, frequency=3, probability=0.6
        )
        self.representation.transitions[self.state1][self.state2] = Transition(
            action=0, frequency=4, probability=0.7
        )
        self.representation.transitions[self.state2][self.state3] = Transition(
            action=0, frequency=2, probability=0.9
        )

        # Verify states
        self.assertEqual(len(list(self.representation.states)), 4)
        self.assertIn(self.state0, self.representation.states)

        # Verify state metadata
        state0_metadata = self.representation.states[self.state0].metadata
        self.assertEqual(state0_metadata.frequency, 10)
        self.assertEqual(state0_metadata.probability, 0.4)

        # Verify transitions
        self.assertEqual(len(list(self.representation.transitions)), 4)

        # Verify specific transitions
        state0_transitions = list(self.representation.transitions[self.state0])
        self.assertEqual(len(state0_transitions), 2)

        # Check transition to state1
        transition_to_state1 = self.representation.transitions[self.state0][self.state1]
        self.assertEqual(transition_to_state1.action, 0)
        self.assertEqual(transition_to_state1.probability, 0.8)
        self.assertEqual(transition_to_state1.frequency, 5)

        # Check transition to state2
        transition_to_state2 = self.representation.transitions[self.state0][self.state2]
        self.assertEqual(transition_to_state2.action, 1)
        self.assertEqual(transition_to_state2.probability, 0.6)
        self.assertEqual(transition_to_state2.frequency, 3)

    def test_fluent_api_transition_data_properties(self):
        """Test TransitionData properties and delegation."""
        self.setup_test_graph()

        # Get a transition data object
        transitions = list(self.representation.transitions)
        transition_data = transitions[0]

        # Test basic properties
        self.assertEqual(transition_data.action, self.action0)
        self.assertEqual(transition_data.probability, 1.0)
        self.assertEqual(transition_data.frequency, 1)

        # Test from_state and to_state
        self.assertIsInstance(transition_data.from_state, State)
        self.assertIsInstance(transition_data.to_state, State)

        # Test that it's not the same as the underlying transition
        self.assertIsNot(transition_data, transition_data.transition)

        # Test that we can access the underlying transition
        self.assertIsInstance(transition_data.transition, Transition)

    def test_fluent_api_empty_representation(self):
        """Test fluent API behavior with empty representation."""
        self.representation.clear()

        # Test states iterator
        states = list(self.representation.states)
        self.assertEqual(len(states), 0)

        # Test states metadata
        metadata = self.representation.states.metadata
        self.assertEqual(len(metadata), 0)

        # Test transitions iterator
        transitions = list(self.representation.transitions)
        self.assertEqual(len(transitions), 0)

        # Test state existence
        self.assertNotIn(self.state0, self.representation.states)

        # Test transition existence
        self.assertNotIn(
            (self.state0, self.state1, self.action0), self.representation.transitions
        )

    def test_fluent_api_with_custom_state_metadata(self):
        """Test fluent API with custom state metadata class."""

        class CustomStateMetadata(StateMetadata):
            custom_field: int = 0

        custom_representation = GraphRepresentation(
            state_metadata_class=CustomStateMetadata
        )

        # Add state with custom metadata
        custom_metadata = CustomStateMetadata(
            frequency=5, probability=0.5, custom_field=42
        )
        custom_representation.states[self.state0] = custom_metadata

        # Verify custom metadata is preserved
        retrieved_metadata = custom_representation.states[self.state0].metadata
        self.assertEqual(retrieved_metadata.frequency, 5)
        self.assertEqual(retrieved_metadata.probability, 0.5)
        self.assertEqual(retrieved_metadata.custom_field, 42)

        # Verify it's the correct type
        self.assertIsInstance(retrieved_metadata, CustomStateMetadata)


if __name__ == "__main__":
    unittest.main()
