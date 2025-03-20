import abc
from typing import Collection, Optional

import networkx as nx

from pgeon.discretizer import Discretizer, StateRepresentation
from pgeon.policy_approximator import Action, ProbabilityQuery

class IntentionMixin:
    ...


class PolicyRepresentation(abc.ABC):
    def __init__(self):
        self._discretizer: Discretizer

    @staticmethod
    @abc.abstractmethod
    def load(path: str) -> "PolicyRepresentation":
        ...

    @abc.abstractmethod
    def save(self, ext: str, path: str):
        ...

    @abc.abstractmethod
    def get_possible_actions(self, state: StateRepresentation) -> Collection[Action]:
        ...

    @abc.abstractmethod
    def get_possible_next_states(self, state: StateRepresentation, action: Optional[Action] = None) -> Collection[StateRepresentation]:
        ...


class GraphRepresentation(PolicyRepresentation):

    # Prolly not needed as actual classes
    # class Node:
    #     ...
    #
    # class Edge:
    #     ...

    # Package-agnostic
    class Graph(abc.ABC):
        ...

    class NetworkXGraph(Graph):
        def __init__(self):
            self.graph_backend = nx.MultiDiGraph()

    def __init__(self, graph_backend: str = "networkx"):
        super().__init__()
        # p(s) and p(s',a | s)
        self.graph: GraphRepresentation.Graph
        match graph_backend:
            case "networkx":
                self.graph = GraphRepresentation.NetworkXGraph()
            case _:
                raise NotImplementedError

    def prob(self, query: ProbabilityQuery) -> float:
        ...

    # This refers to getting all states present in graph. Some representations may not be able to iterate over
    #   all states.
    def get_states_in_graph(self) -> Collection[StateRepresentation]:
        ...

    def get_possible_actions(self, state: StateRepresentation) -> Collection[Action]:
        ...

    def get_possible_next_states(self, state: StateRepresentation, action: Optional[Action] = None) -> Collection[StateRepresentation]:
        ...

    # minimum P(s',a|p) forall possible probs.
    def get_overall_minimum_state_transition_probability(self) -> float:
        ...

    @staticmethod
    def load(path: str) -> "PolicyRepresentation":
        ...

    def save(self, ext: str, path: str):
        pass


class IntentionalPolicyGraphRepresentation(GraphRepresentation, IntentionMixin):
    ...