import abc
import csv
from pathlib import Path
from typing import (
    Any,
    Collection,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    cast,
)

import networkx as nx

from pgeon.discretizer import Action, Discretizer, StateRepresentation


class ProbabilityQuery: ...


class IntentionMixin: ...


class PolicyRepresentation(abc.ABC):
    """
    Abstract base class for policy representations.
    A policy representation stores states, actions, and transitions between states.
    """

    def __init__(self):
        self._discretizer: Discretizer

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
    def get_possible_actions(self, state: StateRepresentation) -> Collection[Action]:
        """Get all possible actions from a state."""
        ...

    @abc.abstractmethod
    def get_possible_next_states(
        self, state: StateRepresentation, action: Optional[Action] = None
    ) -> Collection[StateRepresentation]:
        """Get all possible next states from a state, optionally filtered by action."""
        ...

    @abc.abstractmethod
    def has_state(self, state: StateRepresentation) -> bool:
        """Check if a state exists in the policy representation."""
        ...

    @abc.abstractmethod
    def add_state(self, state: StateRepresentation, **attributes) -> None:
        """Add a state to the policy representation with optional attributes."""
        ...

    @abc.abstractmethod
    def add_states_from(
        self, states: Collection[StateRepresentation], **attributes
    ) -> None:
        """Add multiple states to the policy representation with optional attributes."""
        ...

    @abc.abstractmethod
    def add_transition(
        self,
        from_state: StateRepresentation,
        to_state: StateRepresentation,
        action: Action,
        **attributes,
    ) -> None:
        """Add a transition between states with an action and optional attributes."""
        ...

    @abc.abstractmethod
    def add_transitions_from(
        self,
        transitions: Collection[
            Tuple[StateRepresentation, StateRepresentation, Action]
        ],
        **attributes,
    ) -> None:
        """Add multiple transitions with optional attributes."""
        ...

    @abc.abstractmethod
    def get_transition_data(
        self,
        from_state: StateRepresentation,
        to_state: StateRepresentation,
        action: Action,
    ) -> Dict[str, Any]:
        """Get data associated with a specific transition."""
        ...

    @abc.abstractmethod
    def has_transition(
        self,
        from_state: StateRepresentation,
        to_state: StateRepresentation,
        action: Optional[Action] = None,
    ) -> bool:
        """Check if a transition exists."""
        ...

    @abc.abstractmethod
    def get_state_attributes(
        self, attribute_name: str
    ) -> Dict[StateRepresentation, Any]:
        """Get attributes for all states by name."""
        ...

    @abc.abstractmethod
    def set_state_attributes(
        self, attributes: Dict[StateRepresentation, Any], attribute_name: str
    ) -> None:
        """Set attributes for states."""
        ...

    @abc.abstractmethod
    def get_all_states(self) -> Collection[StateRepresentation]:
        """Get all states in the policy representation."""
        ...

    @abc.abstractmethod
    def get_all_transitions(
        self, include_data: bool = False
    ) -> Collection[Tuple[StateRepresentation, StateRepresentation, Dict[str, Any]],]:
        """Get all transitions, optionally including associated data."""
        ...

    @abc.abstractmethod
    def get_outgoing_transitions(
        self, state: StateRepresentation, include_data: bool = False
    ) -> Collection[Tuple[StateRepresentation, StateRepresentation, Dict[str, Any]],]:
        """Get all transitions originating from a state."""
        ...

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear all states and transitions."""
        ...

    @abc.abstractmethod
    def get_transitions_from_state(
        self, state: StateRepresentation
    ) -> Dict[Action, Collection[StateRepresentation]]:
        """Get a mapping of actions to possible next states from a given state."""
        ...


class GraphRepresentation(PolicyRepresentation):
    """
    A policy representation implemented using a graph structure.
    States are represented as nodes, and transitions as edges.
    """

    # Package-agnostic
    class Graph(abc.ABC):
        """Abstract base class for graph implementations."""

        @abc.abstractmethod
        def add_node(self, node: StateRepresentation, **kwargs) -> None: ...

        @abc.abstractmethod
        def add_nodes_from(
            self, nodes: Collection[StateRepresentation], **kwargs
        ) -> None: ...

        @abc.abstractmethod
        def add_edge(
            self, node_from: StateRepresentation, node_to: StateRepresentation, **kwargs
        ) -> None: ...

        @abc.abstractmethod
        def add_edges_from(
            self,
            edges: Collection[Tuple[StateRepresentation, StateRepresentation, Any]],
            **kwargs,
        ) -> None: ...

        @abc.abstractmethod
        def get_edge_data(
            self, node_from: StateRepresentation, node_to: StateRepresentation, key: Any
        ) -> Dict[str, Any]: ...

        @abc.abstractmethod
        def has_node(self, node: StateRepresentation) -> bool: ...

        @abc.abstractmethod
        def get_node(self, node: StateRepresentation) -> Dict[str, Any]: ...

        @abc.abstractmethod
        def has_edge(
            self,
            node_from: StateRepresentation,
            node_to: StateRepresentation,
            key: Any = None,
        ) -> bool: ...

        @abc.abstractmethod
        def nodes(self, data: bool = False) -> Iterator: ...

        @abc.abstractmethod
        def edges(self, data: bool = False) -> Iterator: ...

        @abc.abstractmethod
        def out_edges(
            self, node: StateRepresentation, data: bool = False
        ) -> Iterator: ...

        @abc.abstractmethod
        def clear(self) -> None: ...

        @abc.abstractmethod
        def __getitem__(self, node: StateRepresentation) -> Any: ...

        # TODO: Make the return type include other possible backends
        @property
        @abc.abstractmethod
        def backend(self) -> nx.MultiDiGraph: ...

    class NetworkXGraph(Graph):
        """NetworkX implementation of the Graph interface."""

        def __init__(self):
            # Not calling super().__init__() since Graph is an ABC
            self._nx_graph = nx.MultiDiGraph()

        def __getitem__(self, node: StateRepresentation) -> Any:
            return cast(
                Dict[StateRepresentation, Dict[Any, Dict[str, Any]]],
                self._nx_graph[node],
            )

        def add_node(self, node: StateRepresentation, **kwargs) -> None:
            self._nx_graph.add_node(node, **kwargs)

        def add_nodes_from(
            self, nodes: Collection[StateRepresentation], **kwargs
        ) -> None:
            self._nx_graph.add_nodes_from(nodes, **kwargs)

        def add_edge(
            self, node_from: StateRepresentation, node_to: StateRepresentation, **kwargs
        ) -> None:
            self._nx_graph.add_edge(node_from, node_to, **kwargs)

        def add_edges_from(
            self,
            edges: Collection[Tuple[StateRepresentation, StateRepresentation, Any]],
            **kwargs,
        ) -> None:
            self._nx_graph.add_edges_from(edges, **kwargs)

        def get_edge_data(
            self, node_from: StateRepresentation, node_to: StateRepresentation, key: Any
        ) -> Dict[str, Any]:
            data = self._nx_graph.get_edge_data(node_from, node_to, key)
            return cast(Dict[str, Any], data) if data else {}

        def has_node(self, node: StateRepresentation) -> bool:
            return self._nx_graph.has_node(node)

        def get_node(self, node: StateRepresentation) -> Dict[str, Any]:
            return self._nx_graph.nodes[node]

        def has_edge(
            self,
            node_from: StateRepresentation,
            node_to: StateRepresentation,
            key: Any = None,
        ) -> bool:
            return self._nx_graph.has_edge(node_from, node_to, key)

        def nodes(self, data: bool = False) -> nx.reportviews.NodeView:
            return self._nx_graph.nodes(data=data)

        def edges(self, data: bool = False) -> nx.reportviews.OutMultiEdgeView:
            return self._nx_graph.edges(data=data)

        def out_edges(
            self, node: StateRepresentation, data: bool = False
        ) -> nx.reportviews.OutMultiEdgeView:
            return self._nx_graph.out_edges(node, data=data)

        def clear(self) -> None:
            self._nx_graph.clear()

        @property
        def backend(self) -> nx.MultiDiGraph:
            return self._nx_graph

    def __init__(self, graph_backend: str = "networkx"):
        super().__init__()
        # p(s) and p(s',a | s)
        self.graph: GraphRepresentation.Graph
        if graph_backend == "networkx":
            self.graph = GraphRepresentation.NetworkXGraph()
        else:
            raise NotImplementedError(f"Graph backend {graph_backend} not implemented")

    def prob(self, query: ProbabilityQuery) -> float:
        """Calculate probability for a given query."""
        ...

    # Implementation of PolicyRepresentation interface using graph terminology
    def get_possible_actions(self, state: StateRepresentation) -> Collection[Action]:
        """Get all possible actions from a state."""
        if not self.has_state(state):
            return []
        actions = set()
        for _, _, data in self.graph.out_edges(state, data=True):
            if "action" in data:
                actions.add(data["action"])
        return list(actions)

    def get_possible_next_states(
        self, state: StateRepresentation, action: Optional[Action] = None
    ) -> Collection[StateRepresentation]:
        """Get all possible next states from a state, optionally filtered by action."""
        if not self.has_state(state):
            return []
        if action is None:
            return [to_state for _, to_state in self.graph.out_edges(state)]
        next_states = []
        for _, to_state, data in self.graph.out_edges(state, data=True):
            if "action" in data and data["action"] == action:
                next_states.append(to_state)
        return next_states

    def has_state(self, state: StateRepresentation) -> bool:
        """Check if a state exists in the policy representation."""
        return self.graph.has_node(state)

    def add_state(self, state: StateRepresentation, **attributes) -> None:
        """Add a state to the policy representation with optional attributes."""
        self.graph.add_node(state, **attributes)

    def add_states_from(
        self, states: Collection[StateRepresentation], **attributes
    ) -> None:
        """Add multiple states to the policy representation with optional attributes."""
        self.graph.add_nodes_from(states, **attributes)

    def add_transition(
        self,
        from_state: StateRepresentation,
        to_state: StateRepresentation,
        action: Action,
        **attributes,
    ) -> None:
        """Add a transition between states with an action and optional attributes."""
        all_attributes = attributes.copy()
        all_attributes["action"] = action
        self.graph.add_edge(from_state, to_state, key=action, **all_attributes)

    def add_transitions_from(
        self,
        transitions: Collection[
            Tuple[StateRepresentation, StateRepresentation, Action]
        ],
        **attributes,
    ) -> None:
        """Add multiple transitions with optional attributes."""
        for from_state, to_state, action in transitions:
            self.add_transition(from_state, to_state, action, **attributes)

    def get_transition_data(
        self,
        from_state: StateRepresentation,
        to_state: StateRepresentation,
        action: Action,
    ) -> Dict[str, Any]:
        """Get data associated with a specific transition."""
        data = self.graph.get_edge_data(from_state, to_state, action)
        return data if data else {}

    def has_transition(
        self,
        from_state: StateRepresentation,
        to_state: StateRepresentation,
        action: Optional[Action] = None,
    ) -> bool:
        """Check if a transition exists."""
        return self.graph.has_edge(from_state, to_state, action)

    def get_state_attributes(
        self, attribute_name: str
    ) -> Dict[StateRepresentation, Any]:
        """Get attributes for all states by name."""
        return nx.get_node_attributes(self.graph.backend, attribute_name)

    def set_state_attributes(
        self, attributes: Dict[StateRepresentation, Any], attribute_name: str
    ) -> None:
        """Set attributes for states."""
        nx.set_node_attributes(self.graph.backend, attributes, attribute_name)

    def get_all_states(self) -> Collection[StateRepresentation]:
        """Get all states in the policy representation."""
        return list(self.graph.nodes())

    def get_all_transitions(self, include_data: bool = False) -> Collection:
        """Get all transitions, optionally including associated data."""
        return list(self.graph.edges(data=include_data))

    def get_outgoing_transitions(
        self, state: StateRepresentation, include_data: bool = False
    ) -> Collection:
        """Get all transitions originating from a state."""
        return list(self.graph.out_edges(state, data=include_data))

    def clear(self) -> None:
        """Clear all states and transitions."""
        self.graph.clear()

    def get_transitions_from_state(
        self, state: StateRepresentation
    ) -> Dict[Action, List[StateRepresentation]]:
        """Get a mapping of actions to possible next states from a given state."""
        if not self.has_state(state):
            return {}

        result = {}
        for _, to_state, data in self.graph.out_edges(state, data=True):
            if "action" in data:
                action = data["action"]
                if action not in result:
                    result[action] = []
                result[action].append(to_state)
        return result

    # Legacy methods for backward compatibility
    def has_node(self, node: StateRepresentation) -> bool:
        return self.has_state(node)

    def add_node(self, node: StateRepresentation, **kwargs) -> None:
        self.add_state(node, **kwargs)

    def get_node(self, node: StateRepresentation) -> Dict[str, Any]:
        return self.graph.get_node(node)

    def add_nodes_from(self, nodes: Collection[StateRepresentation], **kwargs) -> None:
        self.add_states_from(nodes, **kwargs)

    def add_edge(
        self, node_from: StateRepresentation, node_to: StateRepresentation, **kwargs
    ) -> None:
        action = kwargs.pop("action", None)
        if action is not None:
            self.add_transition(node_from, node_to, action, **kwargs)
        else:
            self.graph.add_edge(node_from, node_to, **kwargs)

    def add_edges_from(
        self,
        edges: Collection[Tuple[StateRepresentation, StateRepresentation, Action]],
        **kwargs,
    ) -> None:
        self.add_transitions_from(edges, **kwargs)

    def get_edge_data(
        self, node_from: StateRepresentation, node_to: StateRepresentation, key: Any
    ) -> Dict:
        return self.get_transition_data(node_from, node_to, key)

    def has_edge(
        self,
        node_from: StateRepresentation,
        node_to: StateRepresentation,
        key: Any = None,
    ) -> bool:
        return self.has_transition(node_from, node_to, key)

    def get_node_attributes(self, name: str) -> Dict[StateRepresentation, Any]:
        return self.get_state_attributes(name)

    def set_node_attributes(
        self, attributes: Dict[StateRepresentation, Any], name: str
    ) -> None:
        self.set_state_attributes(attributes, name)

    def nodes(self) -> Collection[StateRepresentation]:
        return self.get_all_states()

    def edges(self, data: bool = False):
        return self.get_all_transitions(include_data=data)

    def out_edges(self, node: StateRepresentation, data: bool = False):
        return self.get_outgoing_transitions(node, include_data=data)

    def __getitem__(self, state: StateRepresentation) -> Any:
        """Get the transitions from a state, organized by destination state."""
        return self.graph[state]

    # minimum P(s',a|p) forall possible probs.
    def get_overall_minimum_state_transition_probability(self) -> float: ...

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

                representation.graph.add_edge(
                    node_from,
                    node_to,
                    key=action,
                    frequency=freq,
                    probability=prob,
                    action=action,
                )
        return representation

    def save_csv(self, discretizer: Discretizer, nodes_path: Path, edges_path: Path):
        if not nodes_path.suffix == ".csv":
            raise ValueError(f"Nodes file must have a .csv extension, got {nodes_path}")
        if not edges_path.suffix == ".csv":
            raise ValueError(f"Edges file must have a .csv extension, got {edges_path}")

        if nodes_path.exists():
            raise FileExistsError(f"Nodes file {nodes_path} already exists")
        if edges_path.exists():
            raise FileExistsError(f"Edges file {edges_path} already exists")

        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        edges_path.parent.mkdir(parents=True, exist_ok=True)

        node_ids = {}
        with open(nodes_path, "w+") as f:
            csv_w = csv.writer(f)
            csv_w.writerow(["id", "value", "p(s)", "frequency"])
            for elem_position, node in enumerate(self.nodes()):
                node_ids[node] = elem_position
                csv_w.writerow(
                    [
                        elem_position,
                        discretizer.state_to_str(node),
                        self.get_node(node).get("probability", 0),
                        self.get_node(node).get("frequency", 0),
                    ]
                )

        with open(edges_path, "w+") as f:
            csv_w = csv.writer(f)
            csv_w.writerow(["from", "to", "action", "p(s)", "frequency"])
            for edge in self.edges(data=True):
                state_from, state_to, action = edge
                csv_w.writerow(
                    [
                        node_ids[state_from],
                        node_ids[state_to],
                        action.get("action", None),
                        action.get("probability", 0),
                        action.get("frequency", 0),
                    ]
                )


class IntentionalPolicyGraphRepresentation(GraphRepresentation, IntentionMixin): ...
