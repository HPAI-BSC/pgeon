import abc
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from pgeon.agent import Agent
from pgeon.desire import Desire
from pgeon.discretizer import Discretizer, Predicate, PredicateBasedStateRepresentation
from pgeon.policy_graph import PolicyGraph, PolicyRepresentation

ActionID = str
StateID = PredicateBasedStateRepresentation


# TODO: Check this typing everywhere, tests are still passing


@dataclass
class ProbQuery:
    s: Optional[StateID] = None
    a: Optional[ActionID] = None
    s_prima: Optional[StateID] = None
    given_s: Optional[StateID] = None
    given_a: Optional[ActionID] = None
    given_do_a: Optional[ActionID] = None

    def __post_init__(self):
        # CURRENT ACCEPTABLE USAGES #
        # s | a,given_s | s',a,given_s | s',given_a,given_s |s', given_do_a,given_s | s', s #
        self.validate(**asdict(self))

    @classmethod
    def validate(
        cls,
        s: StateID = None,
        a: ActionID = None,
        s_prima: StateID = None,
        given_s: StateID = None,
        given_a: ActionID = None,
        given_do_a: ActionID = None,
    ):
        assert any(
            [var is not None for var in [s, a, s_prima, given_s, given_a, given_do_a]]
        )
        assert (
            s is None or given_s is None
        ), "Invalid usage, can't use s and given_s simultaneously"
        assert [v is None for v in [a, given_a, given_do_a]].count(
            False
        ) <= 1, "Invalid usage, can't use a, given_a, or given_do_a simultaneously"  # At most 1 not None
        assert s is None or all(
            v is None for v in [a, s_prima, given_s, given_a, given_do_a]
        )
        assert (
            given_a is None or s_prima is not None
        ), "given_a requires setting s_prima"
        assert (
            given_do_a is None or s_prima is not None
        ), "given_do_a requires setting s_prima"


class AbstractIPG(abc.ABC):
    def __init__(self):
        self.registered_desires: List[Desire] = list()

    def prob(self, query: ProbQuery) -> float:
        s, a, s_prima, given_s, given_a, given_do_a = (
            query.s,
            query.a,
            query.s_prima,
            query.given_s,
            query.given_a,
            query.given_do_a,
        )
        if s is not None:
            return self._prob_s(s)
        if a is not None:
            if s_prima is not None:
                return self._prob_s_prima_a_given_s(s_prima, a, given_s)
            else:
                return self._prob_a_given_s(a, given_s)
        elif given_a is not None:
            return self._prob_s_prima_given_a_s(s_prima, given_a, given_s)
        elif given_do_a is not None:
            return self._prob_s_prima_given_do_a_given_s(s_prima, given_a, given_s)
        else:
            return self._prob_s_prima_given_s(s_prima, given_s)

    def register_all_desires(self, desires: List[Desire], stop_criterion: float = 1e-4):
        for desire in desires:
            self.register_desire(desire, stop_criterion)

    def register_desire(self, desire: Desire, stop_criterion: float = 1e-4):
        self.registered_desires.append(desire)

        for s in self.get_all_state_ids():
            self._set_intention(s, desire, 0)

        for s in self.get_all_state_ids():
            node = s
            p = self.check_desire(node, desire)
            if p is not None:
                self.propagate_intention(node, desire, p, stop_criterion)

    def compute_desire_statistics(self, desire: Desire):
        action_prob_distribution = []
        nodes_fulfilled = []
        clause, action_idx = desire.clause, desire.action_idx
        for s in self.get_all_state_ids():
            node = self.stateID_to_node(s)
            p = node.check_desire(clause, action_idx)
            if p is not None:
                action_prob_distribution.append(p)
                nodes_fulfilled.append(node)

        return action_prob_distribution, nodes_fulfilled

    def compute_commitment_stats(self, desire_name, commitment_threshold):
        intention_score = []
        nodes_with_intent = []
        for s in tqdm(
            self.get_all_state_ids()
        ):  # TODO DEcide to parametrise TQDM usage
            node = self.stateID_to_node(s)
            intention = node.intention.get(desire_name, 0)
            if intention >= commitment_threshold:
                intention_score.append(intention)
                nodes_with_intent.append(node)
        return intention_score, nodes_with_intent

    def compute_intention_metrics(self, c_threshold):
        attributed_intention_probabilities, expected_intentions = {}, {}
        nodes_with_any_intention = set()
        for desire in self.registered_desires:
            intention_vals, nodes = self.compute_commitment_stats(
                desire.name, commitment_threshold=c_threshold
            )
            int_states = np.array([self.prob(ProbQuery(s=n.node_id)) for n in nodes])
            nodes_with_any_intention.update(set([n.node_id for n in nodes]))
            attributed_intention_probability = int_states.sum()
            expected_intention = (
                np.dot(np.array(intention_vals), int_states)
                / attributed_intention_probability
            )
            attributed_intention_probabilities[desire.name] = (
                attributed_intention_probability
            )
            expected_intentions[desire.name] = expected_intention
        int_states = np.array(
            [self.prob(ProbQuery(s=n_idx)) for n_idx in nodes_with_any_intention]
        )
        int_total_probability = int_states.sum()
        intention_max_vals = [
            max(list(self.stateID_to_node(n_idx).intention.values()))
            for n_idx in nodes_with_any_intention
        ]
        attributed_intention_probabilities["Any"] = int_total_probability
        expected_intentions["Any"] = (
            np.dot(np.array(intention_max_vals), int_states) / int_total_probability
        )

        return attributed_intention_probabilities, expected_intentions

    @abc.abstractmethod
    def stateID_to_node(self, s: StateID) -> StateID:
        pass

    @abc.abstractmethod
    def check_desire(self, node, desire: Desire) -> float:
        pass

    @abc.abstractmethod
    def get_possible_actions(self, s: StateID) -> List[ActionID]:
        pass

    @abc.abstractmethod
    def get_possible_s_prima(self, s: StateID, a: ActionID = None) -> List[StateID]:
        raise NotImplemented

    @abc.abstractmethod
    def get_all_state_ids(self) -> Iterable[StateID]:
        pass

    @abc.abstractmethod
    def _prob_s(self, s: StateID) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _prob_s_prima_a_given_s(
        self, s_prima: StateID, a: ActionID, given_s: StateID
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _prob_a_given_s(self, a: ActionID, given_s: StateID) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _prob_s_prima_given_a_s(
        self, s_prima: StateID, given_a: ActionID, given_s: StateID
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _prob_s_prima_given_do_a_given_s(
        self, s_prima: StateID, given_do_a: ActionID, given_s: StateID
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _prob_s_prima_given_s(self, s_prima: StateID, given_s: StateID) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def propagate_intention(
        self, node: StateID, desire: Desire, p: float, stop_criterion: float
    ):
        pass

    @abc.abstractmethod
    def _set_intention(self, s, desire, param):
        pass


class IPG(PolicyGraph, AbstractIPG):
    def __init__(
        self,
        discretizer: Discretizer,
        policy_representation: PolicyRepresentation,
        environment: gym.Env,
        agent: Agent,
        verbose=False,
    ):
        super().__init__(discretizer, policy_representation, environment, agent)
        self.registered_desires: List[Desire] = (
            list()
        )  # TODO: temp fix since AbstractIPG init is not callable here
        self.verbose = verbose

    def get_possible_actions(self, s: StateID) -> List[ActionID]:
        # Returns any a s.t. P(a|s)>0
        if type(s) == int:
            # TODO: Legacy: to be removed
            node = self.graph.nodes[s]
        else:
            node = s
        actions_with_probs = [
            (data["action"], dest, data["prob"])
            for orig, dest, data in self.graph.out_edges(node, data=True)
        ]
        # TODO: Check on this
        actions_with_probs = list(
            set(actions_with_probs)
        )  # deduplicate, for some reason needed (?)
        actions = list(
            set([act for act, _, p in actions_with_probs if p > 0])
        )  # Just in case there's 0-prob edges
        return actions

    def get_possible_s_prima(self, s: StateID, a: ActionID = None) -> List[StateID]:
        # Returns any a s.t. P(a|s)>0
        if type(s) == int:
            # TODO: Legacy: to be removed
            node = self.graph.nodes[s]
        else:
            node = s
        edges_with_probs = [
            (data["action"], dest, data["prob"])
            for orig, dest, data in self.graph.out_edges(node, data=True)
        ]
        # TODO: Check on this
        edges_with_probs = list(
            set(edges_with_probs)
        )  # deduplicate, for some reason needed (?)
        if a is not None:
            # Filter for edges annotated with action= a only
            destinies = list(
                set([dest for act, dest, p in edges_with_probs if p > 0 and act == a])
            )
        else:
            destinies = list(set([dest for act, dest, p in edges_with_probs if p > 0]))
        return destinies

    def get_all_state_ids(self) -> Iterable[StateID]:
        return self.graph.nodes

    def check_desire(self, node: Tuple[Predicate], desire: Desire) -> Optional[float]:
        # If it is a node where desire can be fulfilled, returns the immediate probability of fulfilling it
        # Else returns None
        if desire.type == "achievement":
            # TODO: Convert desire clauses into predicates and do it fancy
            predicates_in_state = set(node.predicates)
            fulfillable_desire = desire.clause.issubset(predicates_in_state)
            if fulfillable_desire:
                return self.prob(
                    ProbQuery(a=desire.action_idx, given_s=node)
                )  # TODO:Typing error to be verified and checked
            else:
                return None
        else:
            raise NotImplementedError

    def get_intention(self, s: StateID, desire: Desire):
        try:
            return self.graph.nodes[s]["intention"][desire]
        except KeyError:
            return 0

    def get_intentions(self, s: StateID) -> Dict[Desire, float]:
        try:
            return self.graph.nodes[s]["intention"]
        except KeyError:
            return dict()

    def _set_intention(self, s, desire, new_int):
        try:
            self.graph.nodes[s]["intention"][desire] = new_int
        except KeyError:
            self.graph.nodes[s]["intention"] = dict()
            self.graph.nodes[s]["intention"][desire] = new_int

    def _prob_s(self, s: StateID):
        return self.graph.nodes[s].probability

    def _prob_s_prima_a_given_s(self, s_prima: StateID, a: ActionID, given_s: StateID):
        # Assuming the 's' is always in predicates format for simplicity
        try:
            prob = [
                data["prob"]
                for orig, dest, data in self.graph.out_edges(given_s, data=True)
                if a == data["action"] and dest == s_prima
            ]
            if len(prob) > 1:
                raise AssertionError(
                    f"More than one possible edge in query considering "
                    f"{s_prima},{a}|{given_s}:\n"
                    f"{self.graph.out_edges(given_s, data=True)}"
                )
            if len(prob) == 0:
                warnings.warn(
                    "_prob_s_prima_a_given_s checked a non-existent probability."
                )
                return 0
            else:
                return prob[0]
        except KeyError:
            return 0

    def _prob_a_given_s(self, a: ActionID, given_s: StateID):
        # Assuming the 's' is always in predicates format for simplicity
        try:
            prob = [
                data["prob"]
                for orig, dest, data in self.graph.out_edges(given_s, data=True)
                if a == data["action"]
            ]
            if len(prob) == 0:
                warnings.warn(
                    "_prob_s_prima_a_given_s checked a non-existent probability."
                )
                return 0
            else:
                return sum(prob)
        except KeyError:
            return 0

    def _prob_s_prima_given_a_s(
        self, s_prima: StateID, given_a: ActionID, given_s: StateID
    ):
        p_a__s = self._prob_a_given_s(given_a, given_s)
        if p_a__s == 0:
            return 0
        else:
            prob = [
                data["prob"]
                for orig, dest, data in self.graph.out_edges(given_s, data=True)
                if given_a == data["action"] and dest == s_prima
            ]
            if len(prob) > 1:
                raise AssertionError(
                    f"More than one possible edge in query considering "
                    f"{s_prima}|{given_a},{given_s}:\n"
                    f"{self.graph.out_edges(given_s, data=True)}"
                )
            elif len(prob) == 0:
                return 0
            else:
                return prob[0] / p_a__s

    def _prob_s_prima_given_do_a_given_s(
        self, s_prima: StateID, given_do_a: ActionID, given_s: StateID
    ):
        raise NotImplementedError("Basic PG can't handle do(a)")

    def _prob_s_prima_given_s(self, s_prima: StateID, given_s: StateID):
        prob = [
            data["prob"]
            for orig, dest, data in self.graph.out_edges(given_s, data=True)
            if dest == s_prima
        ]
        return sum(prob)

    def stateID_to_node(self, s: StateID) -> StateID:
        return self.graph.nodes[s]

    def propagate_intention(
        self,
        node: StateID,
        desire: Desire,
        propagated_intention: float,
        stop_criterion: float,
    ):
        # TODO: This should eventually be a method of AbstractIPG
        desire_name = desire.name
        self._update_intention(node, desire, propagated_intention)

        parents = set([orig for orig, _ in self.graph.in_edges(node)])

        for parent in parents:
            if self.check_desire(parent, desire) is None:
                prob_transition = self.prob(ProbQuery(s_prima=node, given_s=parent))
            else:
                # If coincider can fulfill desire themselves, do not propagate it through the action_idx branch
                # (as that would compute Expected #desires, instead of desire probability, and that can be >1).

                prob_transition = self.prob(ProbQuery(s_prima=node, given_s=parent))
                if desire.type == "achievement":
                    # (We want to remove from P(s' |s) all P(s',a=Desired action | s)
                    prob_transition_through_action = self.prob(
                        ProbQuery(s_prima=node, a=desire.action_idx, given_s=parent)
                    )
                    prob_transition -= prob_transition_through_action
                else:
                    raise NotImplementedError

            new_coincider_intention_value = prob_transition * propagated_intention

            if new_coincider_intention_value >= stop_criterion:
                # avoid infinite loops
                try:
                    self.propagate_intention(
                        parent, desire, new_coincider_intention_value, stop_criterion
                    )
                except RecursionError:
                    print(
                        "Maximum recursion reach, skipping branch with intention of",
                        new_coincider_intention_value,
                    )

    def _update_intention(self, node, desire, intention):
        graph_node = self.graph.nodes[node]
        current_intention_val = graph_node["intention"][desire]
        graph_node["intention"][desire] = current_intention_val + intention

    def get_action_probability(self, state: StateID) -> Dict[ActionID, float]:
        # TODO: This should go in the representation parent class
        return {
            a: self.prob(ProbQuery(a=a, given_s=state))
            for a in self.get_possible_actions(state)
        }
