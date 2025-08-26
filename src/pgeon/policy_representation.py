from __future__ import annotations

import abc
import csv
import re
from pathlib import Path
from typing import (
    Any,
    Collection,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import networkx as nx

from pgeon.discretizer import (
    Action,
    Discretizer,
    State,
    StateMetadata,
    Transition,
    TransitionData,
)

TStateMetadata = TypeVar("TStateMetadata", bound=StateMetadata)


class TransitionView(Generic[TStateMetadata]):
    """A fluent interface for accessing transitions in a policy representation."""

    def __init__(self, representation: PolicyRepresentation[TStateMetadata]):
        self._representation = representation

    def __iter__(self) -> Iterator[TransitionData]:
        """Iterate over all transitions."""
        for from_state, to_state, data in self._representation._get_all_transitions():
            transition = Transition.model_validate(data)
            yield TransitionData(transition, from_state, to_state)

    def __getitem__(self, state: State) -> StateTransitionView[TStateMetadata]:
        """Get transitions for a specific state."""
        return StateTransitionView(self._representation, state)

    def __contains__(self, item: Tuple[State, State, Action]) -> bool:
        """Check if a transition exists."""
        if len(item) != 3:
            return False
        from_state, to_state, action = item
        return self._representation._has_transition(from_state, to_state, action)


class StateTransitionView(Generic[TStateMetadata]):
    """A view of transitions for a specific state."""

    def __init__(
        self, representation: PolicyRepresentation[TStateMetadata], state: State
    ):
        self._representation = representation
        self._state = state

    def __iter__(self) -> Iterator[TransitionData]:
        """Iterate over all transitions from this state."""
        for (
            from_state,
            to_state,
            data,
        ) in self._representation._get_outgoing_transitions(self._state):
            transition = Transition.model_validate(data)
            yield TransitionData(transition, from_state, to_state)

    def __getitem__(self, to_state: State) -> Transition:
        """Get transition to a specific state."""
        for _, target_state, data in self._representation._get_outgoing_transitions(
            self._state
        ):
            if target_state == to_state:
                return Transition.model_validate(data)
        raise KeyError(f"No transition from {self._state} to {to_state}")

    def __setitem__(self, to_state: State, transition: Transition) -> None:
        """Set transition to a specific state."""
        self._representation._add_transition(self._state, to_state, transition)


class StateView(Generic[TStateMetadata]):
    """A fluent interface for accessing states in a policy representation."""

    def __init__(self, representation: PolicyRepresentation[TStateMetadata]):
        self._representation = representation

    def __iter__(self) -> Iterator[State]:
        """Iterate over all states."""
        return iter(self._representation._get_all_states())

    def __contains__(self, state: State) -> bool:
        """Check if a state exists."""
        return self._representation._has_state(state)

    def __getitem__(self, state: State) -> StateMetadataView[TStateMetadata]:
        """Get metadata for a specific state."""
        return StateMetadataView(self._representation, state)

    def __setitem__(self, state: State, metadata: TStateMetadata) -> None:
        """Set metadata for a specific state."""
        self._representation._add_state(state, metadata)

    @property
    def metadata(self) -> Dict[State, TStateMetadata]:
        """Get metadata for all states."""
        return self._representation._get_all_state_metadata()


class StateMetadataView(Generic[TStateMetadata]):
    """A view of metadata for a specific state."""

    def __init__(
        self, representation: PolicyRepresentation[TStateMetadata], state: State
    ):
        self._representation = representation
        self._state = state

    @property
    def metadata(self) -> TStateMetadata:
        """Get the state's metadata."""
        return self._representation._get_state_data(self._state)

    @property
    def predecessors(self) -> Collection[State]:
        """Get all predecessors of this state."""
        return self._representation._get_predecessors(self._state)

    @property
    def successors(self) -> Collection[State]:
        """Get all successors of this state."""
        return self._representation._get_possible_next_states(self._state)


class PolicyRepresentation(abc.ABC, Generic[TStateMetadata]):
    """
    Abstract base class for policy representations.
    A policy representation stores states, actions, and transitions between states.
    """

    def __init__(self, state_metadata_class: Type[TStateMetadata] = StateMetadata):
        self._discretizer: Discretizer
        self.state_metadata_class = state_metadata_class
        # Initialize fluent API views
        self._states_view = StateView(self)
        self._transitions_view = TransitionView(self)

    @property
    def states(self) -> StateView[TStateMetadata]:
        """Access states with a fluent API."""
        return self._states_view

    @property
    def transitions(self) -> TransitionView[TStateMetadata]:
        """Access transitions with a fluent API."""
        return self._transitions_view

    @staticmethod
    @abc.abstractmethod
    def load_csv(
        graph_backend: str, discretizer: Discretizer, nodes_path: Path, edges_path: Path
    ) -> "PolicyRepresentation":
        """Load a policy representation from a set of CSV files."""
        ...

    @abc.abstractmethod
    def save_csv(self, nodes_path: Path, edges_path: Path):
        """Save a policy representation to a set of CSV files."""
        ...

    @abc.abstractmethod
    def save_gram(self, discretizer: Discretizer, path: Path):
        """Save a policy representation to a gram file."""
        ...

    @staticmethod
    @abc.abstractmethod
    def load_gram(
        graph_backend: str, discretizer: Discretizer, path: Path
    ) -> "PolicyRepresentation":
        """Load a policy representation from a gram file."""
        ...

    @abc.abstractmethod
    def _get_possible_transitions(self, state: State) -> List[Transition]:
        """Get all possible actions from a state."""
        ...

    @abc.abstractmethod
    def _get_possible_next_states(
        self, state: State, action: Optional[Action] = None
    ) -> Collection[State]:
        """Get all possible next states from a state, optionally filtered by action."""
        ...

    @abc.abstractmethod
    def _has_state(self, state: State) -> bool:
        """Check if a state exists in the policy representation."""
        ...

    @abc.abstractmethod
    def _get_state_data(self, state: State) -> TStateMetadata:
        """Get data associated with a specific state."""
        ...

    @abc.abstractmethod
    def _add_state(
        self, state: State, state_metadata: Optional[TStateMetadata] = None
    ) -> None:
        """Add a state to the policy representation with optional attributes."""
        ...

    @abc.abstractmethod
    def _add_states_from(
        self,
        states: Collection[State],
        state_metadata: Optional[TStateMetadata] = None,
    ) -> None:
        """Add multiple states to the policy representation with optional attributes."""
        ...

    @abc.abstractmethod
    def _add_transition(
        self,
        from_state: State,
        to_state: State,
        transition: Transition,
    ) -> None:
        """Add a transition between states."""
        ...

    @abc.abstractmethod
    def _add_transitions_from(
        self,
        transitions: Collection[Tuple[State, State, Transition]],
    ) -> None:
        """Add multiple transitions."""
        ...

    @abc.abstractmethod
    def _has_transition(
        self,
        from_state: State,
        to_state: State,
        action: Optional[Action] = None,
    ) -> bool:
        """Check if a transition exists."""
        ...

    @abc.abstractmethod
    def _get_all_state_metadata(self) -> Dict[State, TStateMetadata]:
        """Get metadata for all states."""
        ...

    @abc.abstractmethod
    def _set_state_metadata(
        self, state_to_state_metadata: Dict[State, TStateMetadata]
    ) -> None:
        """Set metadata for states."""
        ...

    @abc.abstractmethod
    def _get_all_states(self) -> Collection[State]:
        """Get all states in the policy representation."""
        ...

    @abc.abstractmethod
    def _get_all_transitions(
        self,
    ) -> Collection[Tuple[State, State, Dict[str, Any]],]:
        """Get all transitions, including associated data."""
        ...

    @abc.abstractmethod
    def _get_outgoing_transitions(
        self, state: State
    ) -> Collection[Tuple[State, State, Dict[str, Any]],]:
        """Get all transitions originating from a state."""
        ...

    @abc.abstractmethod
    def _get_predecessors(self, state: State) -> Collection[State]:
        """Get all predecessors of a state."""
        ...

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear all states and transitions."""
        ...

    @abc.abstractmethod
    def _get_transitions_from_state(
        self, state: State
    ) -> Dict[Action, Collection[State]]:
        """Get a mapping of actions to possible next states from a given state."""
        ...

    @abc.abstractmethod
    def add_trajectory(self, trajectory: list[Any]) -> None:
        """Add a trajectory to the policy representation."""
        ...


class GraphRepresentation(PolicyRepresentation[TStateMetadata]):
    """
    A policy representation implemented using a graph structure.
    States are represented as nodes, and transitions as edges.
    """

    # Package-agnostic
    class Graph(abc.ABC):
        """Abstract base class for graph implementations."""

        @abc.abstractmethod
        def add_node(self, node: State, **kwargs) -> None: ...

        @abc.abstractmethod
        def add_nodes_from(self, nodes: Collection[State], **kwargs) -> None: ...

        @abc.abstractmethod
        def add_edge(self, node_from: State, node_to: State, **kwargs) -> None: ...

        @abc.abstractmethod
        def add_edges_from(
            self,
            edges: Collection[Tuple[State, State, Any]],
            **kwargs,
        ) -> None: ...

        @abc.abstractmethod
        def has_node(self, node: State) -> bool: ...

        @abc.abstractmethod
        def get_node(self, node: State) -> Dict[str, Any]: ...

        @abc.abstractmethod
        def has_edge(
            self,
            node_from: State,
            node_to: State,
            key: Any = None,
        ) -> bool: ...

        @abc.abstractmethod
        def nodes(self, data: bool = False) -> Iterator: ...

        @abc.abstractmethod
        def edges(self, data: bool = False) -> Iterator: ...

        @abc.abstractmethod
        def out_edges(self, node: State, data: bool = False) -> Iterator: ...

        @abc.abstractmethod
        def predecessors(self, node: State) -> Iterator: ...

        @abc.abstractmethod
        def get_node_attributes(self, attribute_name: str) -> Dict[State, Any]: ...

        @abc.abstractmethod
        def set_node_attributes(
            self, attributes: Dict[State, Any], attribute_name: str
        ) -> None: ...

        @abc.abstractmethod
        def clear(self) -> None: ...

        @abc.abstractmethod
        def __getitem__(self, node: State) -> TStateMetadata: ...

        @abc.abstractmethod
        def __setitem__(self, node: State, metadata: TStateMetadata) -> None: ...

        # TODO: Make the return type include other possible backends
        @property
        @abc.abstractmethod
        def backend(self) -> nx.MultiDiGraph: ...

    class NetworkXGraph(Graph):
        """NetworkX implementation of the Graph interface."""

        def __init__(self, state_metadata_class: Type[TStateMetadata] = StateMetadata):
            # Not calling super().__init__() since Graph is an ABC
            self._nx_graph = nx.MultiDiGraph()
            self.state_metadata_class = state_metadata_class

        def __getitem__(self, node: State) -> TStateMetadata:
            return self.state_metadata_class.model_validate(self._nx_graph.nodes[node])

        def __setitem__(self, node: State, metadata: TStateMetadata) -> None:
            if not self._nx_graph.has_node(node):
                self._nx_graph.add_node(node)
            self._nx_graph.nodes[node].update(metadata.model_dump())

        def add_node(self, node: State, **kwargs) -> None:
            self._nx_graph.add_node(node, **kwargs)

        def add_nodes_from(self, nodes: Collection[State], **kwargs) -> None:
            self._nx_graph.add_nodes_from(nodes, **kwargs)

        def add_edge(self, node_from: State, node_to: State, **kwargs) -> None:
            self._nx_graph.add_edge(node_from, node_to, **kwargs)

        def add_edges_from(
            self,
            edges: Collection[Tuple[State, State, Any]],
            **kwargs,
        ) -> None:
            self._nx_graph.add_edges_from(edges, **kwargs)

        def has_node(self, node: State) -> bool:
            return self._nx_graph.has_node(node)

        def get_node(self, node: State) -> Dict[str, Any]:
            return self._nx_graph.nodes[node]

        def has_edge(
            self,
            node_from: State,
            node_to: State,
            key: Any = None,
        ) -> bool:
            return self._nx_graph.has_edge(node_from, node_to, key)

        def nodes(self, data: bool = False) -> nx.reportviews.NodeView:
            return self._nx_graph.nodes(data=data)

        def edges(self, data: bool = False) -> nx.reportviews.OutMultiEdgeView:
            return self._nx_graph.edges(data=data)

        def out_edges(
            self, node: State, data: bool = False
        ) -> nx.reportviews.OutMultiEdgeView:
            return self._nx_graph.out_edges(node, data=data)

        def predecessors(self, node: State) -> Iterator:
            return self._nx_graph.predecessors(node)

        def get_node_attributes(self, attribute_name: str) -> Dict[State, Any]:
            return nx.get_node_attributes(self._nx_graph, attribute_name)

        def set_node_attributes(
            self, attributes: Dict[State, Any], attribute_name: str
        ) -> None:
            nx.set_node_attributes(self._nx_graph, attributes, attribute_name)

        def clear(self) -> None:
            self._nx_graph.clear()

        @property
        def backend(self) -> nx.MultiDiGraph:
            return self._nx_graph

    def __init__(
        self,
        graph_backend: str = "networkx",
        state_metadata_class: Type[TStateMetadata] = StateMetadata,
    ):
        super().__init__(state_metadata_class=state_metadata_class)
        # p(s) and p(s',a | s)
        self.graph: GraphRepresentation.Graph
        self.discretizer: Discretizer
        if graph_backend == "networkx":
            self.graph = GraphRepresentation.NetworkXGraph(
                state_metadata_class=state_metadata_class
            )
        else:
            raise NotImplementedError(f"Graph backend {graph_backend} not implemented")

    # Implementation of PolicyRepresentation interface using graph terminology
    def _get_possible_transitions(self, state: State) -> List[Transition]:
        """Get all possible transitions from a state with their probabilities."""
        if not self._has_state(state):
            return []

        transitions = []
        for _, _, data in self.graph.out_edges(state, data=True):
            transitions.append(Transition.model_validate(data))

        return sorted(transitions, key=lambda item: item.probability, reverse=True)

    def _get_possible_next_states(
        self, state: State, action: Optional[Action] = None
    ) -> Collection[State]:
        """Get all possible next states from a state, optionally filtered by action."""
        if not self._has_state(state):
            return []
        if action is None:
            return [to_state for _, to_state in self.graph.out_edges(state)]
        next_states = []
        for _, to_state, data in self.graph.out_edges(state, data=True):
            transition = Transition.model_validate(data)
            if transition.action == action:
                next_states.append(to_state)
        return next_states

    def _has_state(self, state: State) -> bool:
        """Check if a state exists in the policy representation."""
        return self.graph.has_node(state)

    def _get_state_data(self, state: State) -> TStateMetadata:
        """Get data associated with a specific state."""
        return self.state_metadata_class.model_validate(self.graph.get_node(state))

    def _add_state(
        self, state: State, state_metadata: Optional[TStateMetadata] = None
    ) -> None:
        """Add a state to the policy representation with optional attributes."""
        if state_metadata is None:
            state_metadata = self.state_metadata_class()
        self.graph.add_node(state, **state_metadata.model_dump())

    def _add_states_from(
        self,
        states: Collection[State],
        state_metadata: Optional[TStateMetadata] = None,
    ) -> None:
        """Add multiple states to the policy representation with optional attributes."""
        if state_metadata is None:
            state_metadata = self.state_metadata_class()
        self.graph.add_nodes_from(states, **state_metadata.model_dump())

    def _add_transition(
        self,
        from_state: State,
        to_state: State,
        transition: Transition,
    ) -> None:
        """Add a transition between states with an action and optional attributes."""
        self.graph.add_edge(
            from_state, to_state, key=transition.action, **transition.model_dump()
        )

    def _add_transitions_from(
        self,
        transitions: Collection[Tuple[State, State, Transition]],
    ) -> None:
        """Add multiple transitions with optional attributes."""
        for from_state, to_state, transition in transitions:
            self._add_transition(
                from_state,
                to_state,
                transition,
            )

    def _has_transition(
        self,
        from_state: State,
        to_state: State,
        action: Optional[Action] = None,
    ) -> bool:
        """Check if a transition exists."""
        return self.graph.has_edge(from_state, to_state, action)

    def _get_all_state_metadata(self) -> Dict[State, TStateMetadata]:
        """Get metadata for all states."""
        return {
            state: self.state_metadata_class.model_validate(data)
            for state, data in self.graph.nodes(data=True)
        }

    def _set_state_metadata(
        self, state_to_state_metadata: Dict[State, TStateMetadata]
    ) -> None:
        """Set metadata for states."""
        for state, metadata in state_to_state_metadata.items():
            self.graph.get_node(state).update(metadata.model_dump())

    def _get_all_states(self) -> Collection[State]:
        """Get all states in the policy representation."""
        return list(self.graph.nodes())

    def _get_all_transitions(self) -> Collection:
        """Get all transitions, including associated data."""
        return list(self.graph.edges(data=True))

    def _get_outgoing_transitions(self, state: State) -> Collection:
        """Get all transitions originating from a state."""
        return list(self.graph.out_edges(state, data=True))

    def _get_predecessors(self, state: State) -> Collection[State]:
        """Get all predecessors of a state."""
        return list(self.graph.predecessors(state))

    def clear(self) -> None:
        """Clear all states and transitions."""
        self.graph.clear()

    def _get_transitions_from_state(
        self, state: State
    ) -> Dict[Action, Collection[State]]:
        """Get a mapping of actions to possible next states from a given state."""
        if not self._has_state(state):
            return {}

        result = {}
        for _, to_state, data in self.graph.out_edges(state, data=True):
            if "action" in data:
                action = data["action"]
                if action not in result:
                    result[action] = []
                result[action].append(to_state)
        return result

    def add_trajectory(self, trajectory: list[Any]) -> None:
        """Adds a trajectory to the graph.
        A trajectory is a list of (state, action) tuples or (state, action, next_state) tuples.
        """
        if isinstance(trajectory[0], int):
            for i in range(0, len(trajectory), 2):
                if i + 2 < len(trajectory):
                    state_from = trajectory[i]
                    action = trajectory[i + 1]
                    state_to = trajectory[i + 2]
                    self._add_transition(
                        state_from, state_to, Transition(action=action, frequency=1)
                    )
        elif len(trajectory[0]) == 2:
            for i in range(len(trajectory) - 1):
                state_from, action = trajectory[i]
                state_to, _ = trajectory[i + 1]
                self._add_transition(
                    state_from, state_to, Transition(action=action, frequency=1)
                )
        else:
            for state_from, action, state_to in trajectory:
                self._add_transition(
                    state_from, state_to, Transition(action=action, frequency=1)
                )

    @staticmethod
    def load_csv(
        graph_backend: str, discretizer: Discretizer, nodes_path: Path, edges_path: Path
    ) -> "GraphRepresentation":
        if not nodes_path.suffix == ".csv":
            raise ValueError(f"Nodes file must have a .csv extension, got {nodes_path}")
        if not edges_path.suffix == ".csv":
            raise ValueError(f"Edges file must have a .csv extension, got {edges_path}")

        if not nodes_path.exists():
            raise FileNotFoundError(f"Nodes file {nodes_path} does not exist")
        if not edges_path.exists():
            raise FileNotFoundError(f"Edges file {edges_path} does not exist")

        representation = GraphRepresentation(graph_backend)

        node_ids_to_values = {}
        with open(nodes_path, "r+") as f:
            csv_r = csv.reader(f)
            next(csv_r)

            for id_str, value, prob, freq in csv_r:
                state_id = int(id_str)
                state_prob = float(prob)
                state_freq = int(freq)
                state_value = discretizer.str_to_state(value)

                representation.graph.add_node(
                    state_value,
                    probability=state_prob,
                    frequency=state_freq,
                )
                node_ids_to_values[state_id] = state_value

        with open(edges_path, "r+") as f:
            csv_r = csv.reader(f)
            next(csv_r)

            for node_from_id, node_to_id, action, prob, freq in csv_r:
                node_from = node_ids_to_values[int(node_from_id)]
                node_to = node_ids_to_values[int(node_to_id)]
                # TODO Get discretizer to process the action id correctly;
                #  we cannot assume the action will always be an int
                action = int(action)
                prob = float(prob)
                freq = int(freq)

                representation._add_transition(
                    node_from,
                    node_to,
                    Transition(action=action, frequency=freq, probability=prob),
                )
        return representation

    def save_csv(self, discretizer: Discretizer, nodes_path: Path, edges_path: Path):
        if not nodes_path.suffix == ".csv":
            raise ValueError(f"Nodes file must have a .csv extension, got {nodes_path}")
        if not edges_path.suffix == ".csv":
            raise ValueError(f"Edges file must have a .csv extension, got {edges_path}")

        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        edges_path.parent.mkdir(parents=True, exist_ok=True)

        node_ids = {}
        with open(nodes_path, "w+") as f:
            csv_w = csv.writer(f)
            csv_w.writerow(["id", "value", "p(s)", "frequency"])
            for elem_position, node in enumerate(self._get_all_states()):
                node_ids[node] = elem_position
                csv_w.writerow(
                    [
                        elem_position,
                        discretizer.state_to_str(node),
                        self.graph.get_node(node).get("probability", 0),
                        self.graph.get_node(node).get("frequency", 0),
                    ]
                )

        with open(edges_path, "w+") as f:
            csv_w = csv.writer(f)
            csv_w.writerow(["from", "to", "action", "p(s)", "frequency"])
            for edge in self._get_all_transitions():
                state_from, state_to, data = edge
                transition = Transition.model_validate(data)
                csv_w.writerow(
                    [
                        node_ids[state_from],
                        node_ids[state_to],
                        transition.action,
                        transition.probability,
                        transition.frequency,
                    ]
                )

    def save_gram(self, discretizer: Discretizer, path: Path):
        if not path.suffix == ".gram":
            raise ValueError(f"File must have a .gram extension, got {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        node_info = {
            node: {
                "id": i,
                "value": discretizer.state_to_str(node),
                "probability": self.graph.get_node(node).get("probability", 0),
                "frequency": self.graph.get_node(node).get("frequency", 0),
            }
            for i, node in enumerate(self._get_all_states())
        }
        action_info = {
            action: {"id": i, "value": str(action)}
            for i, action in enumerate(
                set(
                    Transition.model_validate(data).action
                    for _, _, data in self._get_all_transitions()
                )
            )
        }

        with open(path, "w+") as f:
            # Write nodes
            for _, info in node_info.items():
                f.write(
                    f"\nCREATE (s{info['id']}:State "
                    + "{"
                    + f'\n  uid: "s{info["id"]}",\n  value: "{info["value"]}",\n  probability: {info["probability"]}, \n  frequency:{info["frequency"]}'
                    + "\n});"
                )

            # Write actions
            for _, action in action_info.items():
                f.write(
                    f"\nCREATE (a{action['id']}:Action "
                    + "{"
                    + f'\n  uid: "a{action["id"]}",\n  value:{action["value"]}'
                    + "\n});"
                )

            # Write edges
            for edge in self._get_all_transitions():
                n_from, n_to, data = edge
                transition = Transition.model_validate(data)
                action = transition.action
                if action is not None:
                    f.write(
                        f'\nMATCH (s{node_info[n_from]["id"]}:State) WHERE s{node_info[n_from]["id"]}.uid = "s{node_info[n_from]["id"]}" MATCH (s{node_info[n_to]["id"]}:State) WHERE s{node_info[n_to]["id"]}.uid = "s{node_info[n_to]["id"]}" CREATE (s{node_info[n_from]["id"]})-[:a{action_info[action]["id"]} '
                        + "{"
                        + f"probability:{transition.probability}, frequency:{transition.frequency}"
                        + "}"
                        + f"]->(s{node_info[n_to]['id']});"
                    )

    @staticmethod
    def load_gram(
        graph_backend: str, discretizer: Discretizer, path: Path
    ) -> "GraphRepresentation":
        if not path.suffix == ".gram":
            raise ValueError(f"File must have a .gram extension, got {path}")
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")

        representation = GraphRepresentation(graph_backend)
        node_info = {}  # id -> node
        action_info = {}  # id -> action

        def parse_node_block(lines, start_idx):
            node_lines = [lines[start_idx]]
            i = start_idx
            while not node_lines[-1].strip().endswith("});") and i + 1 < len(lines):
                i += 1
                node_lines.append(lines[i].strip())
            node_block = " ".join(node_lines)
            if "{" in node_block and "}" in node_block:
                attrs_str = node_block.split("{", 1)[1].rsplit("}", 1)[0]
                node_id = int(node_block.split("s")[1].split(":")[0])
                attrs = {}
                for attr in attrs_str.split(","):
                    attr = attr.strip()
                    if not attr or ":" not in attr:
                        continue
                    key, value = attr.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "uid":
                        continue
                    elif key == "value":
                        attrs["value"] = value.strip('"')
                    elif key == "probability":
                        attrs["probability"] = float(value)
                    elif key == "frequency":
                        attrs["frequency"] = int(value)
                if "value" not in attrs:
                    return i, None
                state = discretizer.str_to_state(attrs["value"])
                representation._add_state(
                    state,
                    StateMetadata(
                        probability=attrs.get("probability", 0),
                        frequency=attrs.get("frequency", 0),
                    ),
                )
                node_info[node_id] = state
                return i, node_id
            return i, None

        def parse_action_block(lines, start_idx):
            action_lines = [lines[start_idx]]
            i = start_idx
            while not action_lines[-1].strip().endswith("});") and i + 1 < len(lines):
                i += 1
                action_lines.append(lines[i].strip())
            action_block = " ".join(action_lines)
            if "{" in action_block and "}" in action_block:
                attrs_str = action_block.split("{", 1)[1].rsplit("}", 1)[0]
                action_id = int(action_block.split("a")[1].split(":")[0])
                for attr in attrs_str.split(","):
                    attr = attr.strip()
                    if not attr or ":" not in attr:
                        continue
                    key, value = attr.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "uid":
                        continue
                    elif key == "value":
                        try:
                            action_info[action_id] = int(value)
                        except ValueError:
                            pass
                return i
            return i

        def parse_edge_block(line):
            edge_pattern = re.compile(
                r"MATCH \(s(\d+):State\).*MATCH \(s(\d+):State\).*CREATE \(s\d+\)-\[:a(\d+) \{([^}]*)\}\]->\(s\d+\);"
            )
            match = edge_pattern.search(line)
            if not match:
                return
            from_id = int(match.group(1))
            to_id = int(match.group(2))
            action_id = int(match.group(3))
            attrs = match.group(4)
            prob = 0.0
            freq = 0
            for attr in attrs.split(","):
                attr = attr.strip()
                if attr.startswith("probability:"):
                    prob = float(attr.split(":", 1)[1])
                elif attr.startswith("frequency:"):
                    freq = int(attr.split(":", 1)[1])
            if from_id not in node_info or to_id not in node_info:
                return
            representation._add_transition(
                node_info[from_id],
                node_info[to_id],
                Transition(
                    action=action_info[action_id], probability=prob, frequency=freq
                ),
            )

        with open(path, "r") as f:
            lines = list(f)
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue

                if line.startswith("CREATE (s"):
                    i, _ = parse_node_block(lines, i)
                elif line.startswith("CREATE (a"):
                    i = parse_action_block(lines, i)
                elif line.startswith("MATCH"):
                    parse_edge_block(line)
                i += 1

        return representation
