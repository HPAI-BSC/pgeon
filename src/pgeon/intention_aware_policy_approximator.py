import abc
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from pgeon.agent import Agent
from pgeon.desire import Desire, Goal
from pgeon.discretizer import (
    Action,
    Discretizer,
    Predicate,
    PredicateBasedStateRepresentation,
    StateRepresentation,
    Transition,
)
from pgeon.policy_approximator import (
    PolicyApproximatorFromBasicObservation,
)
from pgeon.policy_representation import PolicyRepresentation

HowTrace = List[
    Tuple[Action, StateRepresentation, float]
]  # Doing action, arriving at state, which has intention
WhyTrace = Dict[str, float | str]


@dataclass
class ProbQuery:
    s: Optional[StateRepresentation] = None
    a: Optional[Action] = None
    s_prima: Optional[StateRepresentation] = None
    given_s: Optional[StateRepresentation] = None
    given_a: Optional[Action] = None
    given_do_a: Optional[Action] = None

    def __post_init__(self):
        # CURRENT ACCEPTABLE USAGES #
        # s | a,given_s | s',a,given_s | s',given_a,given_s |s', given_do_a,given_s | s', s #
        self.validate(**asdict(self))

    @classmethod
    def validate(
        cls,
        s: StateRepresentation = None,
        a: Action = None,
        s_prima: StateRepresentation = None,
        given_s: StateRepresentation = None,
        given_a: Action = None,
        given_do_a: Action = None,
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


class IntentionAwarePolicyApproximator(PolicyApproximatorFromBasicObservation):
    def __init__(
        self,
        discretizer: Discretizer,
        policy_representation: PolicyRepresentation,
        environment: gym.Env,
        agent: Agent,
        verbose=False,
    ):
        super().__init__(discretizer, policy_representation, environment, agent)
        self.registered_desires: List[Desire] = list()
        self.verbose = verbose
        self.c_threshold = 0.5

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
    def stateID_to_node(self, s: StateRepresentation) -> StateRepresentation:
        pass

    @abc.abstractmethod
    def check_desire(
        self, node, desire_clause: Set[Predicate], action_id: int
    ) -> float:
        pass

    @abc.abstractmethod
    def get_possible_actions(self, s: StateRepresentation) -> List[Action]:
        pass

    @abc.abstractmethod
    def get_possible_s_prima(
        self, s: StateRepresentation, a: Action = None
    ) -> List[StateRepresentation]:
        raise NotImplemented

    @abc.abstractmethod
    def get_all_state_ids(self) -> Iterable[StateRepresentation]:
        pass

    @abc.abstractmethod
    def _prob_s(self, s: StateRepresentation) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _prob_s_prima_a_given_s(
        self, s_prima: StateRepresentation, a: Action, given_s: StateRepresentation
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _prob_a_given_s(self, a: Action, given_s: StateRepresentation) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _prob_s_prima_given_a_s(
        self,
        s_prima: StateRepresentation,
        given_a: Action,
        given_s: StateRepresentation,
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _prob_s_prima_given_do_a_given_s(
        self,
        s_prima: StateRepresentation,
        given_do_a: Action,
        given_s: StateRepresentation,
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _prob_s_prima_given_s(
        self, s_prima: StateRepresentation, given_s: StateRepresentation
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def propagate_intention(
        self, node: StateRepresentation, desire: Desire, p: float, stop_criterion: float
    ):
        pass

    @abc.abstractmethod
    def _set_intention(self, s, desire, param):
        pass


class IntentionAwarePolicyApproximator(PolicyApproximatorFromBasicObservation):
    def __init__(
        self,
        discretizer: Discretizer,
        policy_representation: PolicyRepresentation,
        environment: gym.Env,
        agent: Agent,
        verbose=False,
    ):
        super().__init__(discretizer, policy_representation, environment, agent)
        self.registered_desires: List[Desire] = list()
        self.verbose = verbose
        self.c_threshold = 0.5

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

    def find_intentions(self, desires: Set[Desire], commitment_threshold: float):
        self.register_all_desires(desires)
        return self

    def atom_in_state(self, node: Set[Predicate], atom: Predicate):
        return atom in node

    @staticmethod
    def get_prob(unknown_dict: Optional[Dict[str, float]]) -> float:
        if unknown_dict is None:
            return 0
        else:
            return unknown_dict.get("probability", 0)

    def register_desire(self, desire: Desire, stop_criterion=1e-4):
        self.registered_desires.append(desire)
        for node in self.policy_representation.get_all_states():
            node_data = self.policy_representation.get_state_data(node)
            if "intention" not in node_data:
                node_data["intention"] = {}
            node_data["intention"][desire] = 0.0

        for node in self.policy_representation.get_all_states():
            p = self.check_desire(node, desire)
            if p > 0:
                self.propagate_intention(node, desire, p, stop_criterion)

    def get_possible_actions(self, s: StateRepresentation) -> List[Action]:
        # Returns any a s.t. P(a|s)>0
        if isinstance(s, int):
            # TODO: Legacy: to be removed
            s_node = self.stateID_to_node(s)
            s_transitions = self.policy_representation.get_transitions_from_state(
                s_node
            )
            return list(s_transitions.keys())
        else:
            return list(self.policy_representation.get_transitions_from_state(s).keys())

    def get_possible_s_prima(
        self, s: StateRepresentation, a: Action = None
    ) -> List[StateRepresentation]:
        # Returns any a s.t. P(a|s)>0
        if isinstance(s, int):
            # TODO: Legacy: to be removed
            node = self.stateID_to_node(s)
        else:
            node = s
        edges_with_probs = [
            (data["action"], dest, data["probability"])
            for orig, dest, data in self.policy_representation.get_outgoing_transitions(
                node
            )
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

    def get_all_state_ids(self) -> Iterable[StateRepresentation]:
        return self.policy_representation.get_all_states()

    def check_desire(
        self, node: PredicateBasedStateRepresentation, desire: Desire
    ) -> float:
        if isinstance(node, PredicateBasedStateRepresentation):
            predicates_in_state = set(node.predicates)
        else:
            predicates_in_state = set(node)

        if desire.clause.issubset(predicates_in_state):
            return 1.0
        return 0.0

    def get_intention(self, s: StateRepresentation, desire: Desire):
        try:
            node_data = self.policy_representation.get_state_data(s)
            if "intention" in node_data and desire in node_data["intention"]:
                return node_data["intention"][desire]
            return 0
        except KeyError:
            return 0

    def get_intentions(self, s: StateRepresentation) -> Dict[Desire, float]:
        try:
            return self.policy_representation.get_state_data(s)["intention"]
        except KeyError:
            return dict()

    def _set_intention(self, s, desire, new_int):
        try:
            self.policy_representation.get_state_data(s)["intention"][desire] = new_int
        except KeyError:
            self.policy_representation.get_state_data(s)["intention"] = dict()
            self.policy_representation.get_state_data(s)["intention"][desire] = new_int

    def _prob_s(self, s: StateRepresentation):
        return self.policy_representation.get_state_data(s).probability

    def _prob_s_prima_a_given_s(
        self, s_prima: StateRepresentation, a: Action, given_s: StateRepresentation
    ):
        # Assuming the 's' is always in predicates format for simplicity
        try:
            prob = [
                Transition.model_validate(data).probability
                for _, dest, data in self.policy_representation.get_outgoing_transitions(
                    given_s
                )
                if a == Transition.model_validate(data).action and dest == s_prima
            ]
            if len(prob) > 1:
                raise AssertionError(
                    f"More than one possible edge in query considering "
                    f"{s_prima},{a}|{given_s}:\n"
                    f"{self.policy_representation.get_outgoing_transitions(given_s)}"
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

    def _prob_a_given_s(self, a: Action, given_s: StateRepresentation):
        # Assuming the 's' is always in predicates format for simplicity
        try:
            prob = [
                Transition.model_validate(data).probability
                for _, _, data in self.policy_representation.get_outgoing_transitions(
                    given_s
                )
                if a == Transition.model_validate(data).action
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

    def _prob_s_prima_given_a_s(self, s_prima, given_a, given_s):
        """p(s'|a,s)"""
        transitions = self.policy_representation.get_outgoing_transitions(given_s)
        prob = [
            Transition.model_validate(data).probability
            for _, dest, data in transitions
            if given_a == Transition.model_validate(data).action and dest == s_prima
        ]
        return sum(prob)

    def _prob_s_prima_given_do_a_given_s(
        self,
        s_prima: StateRepresentation,
        given_do_a: Action,
        given_s: StateRepresentation,
    ):
        raise NotImplementedError("Basic PG can't handle do(a)")

    def _prob_s_prima_given_s(self, s_prima, given_s):
        """p(s'|s)"""
        # TODO: this needs to be marginalized over actions
        transitions = self.policy_representation.get_outgoing_transitions(given_s)
        prob = [
            Transition.model_validate(data).probability
            for _, dest, data in transitions
            if dest == s_prima
        ]
        return sum(prob)

    def stateID_to_node(self, s: StateRepresentation) -> dict[str, Any]:
        return self.policy_representation.get_state_data(s)

    def propagate_intention(
        self,
        node: StateRepresentation,
        desire: Desire,
        propagated_intention: float,
        stop_criterion: float,
    ):
        # TODO: This should eventually be a method of AbstractIPG
        self._update_intention(node, desire, propagated_intention)

        for parent in self.policy_representation.get_predecessors(node):
            prob_transition = self.prob(ProbQuery(s_prima=node, given_s=parent))
            new_coincider_intention_value = prob_transition * propagated_intention

            if new_coincider_intention_value >= stop_criterion:
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
        graph_node = self.policy_representation.get_state_data(node)
        current_intention_val = graph_node["intention"].get(desire, 0.0)
        graph_node["intention"][desire] = current_intention_val + intention

    def get_action_probability(self, state: StateRepresentation) -> Dict[Action, float]:
        # TODO: This should go in the representation parent class
        return {
            a: self.prob(ProbQuery(a=a, given_s=state))
            for a in self.get_possible_actions(state)
        }

    def answer_what(self, state: StateRepresentation) -> List[Tuple[Goal, float]]:
        """Answers the question: What are the intentions in a given state?"""
        intentions = self.policy_representation.get_state_attributes("intention")
        if state in intentions:
            return [(d, v) for d, v in intentions[state].items()]
        return []

    def answer_how(
        self,
        state: StateRepresentation,
        desires: List[Desire],
    ) -> Dict[Desire, HowTrace]:
        """Answers the question: How to achieve a desire from a given state?"""
        how_paths_per_desire = dict()
        for desire in desires:
            x = self.check_desire(state, desire)
            if x is not None:
                how_paths_per_desire[desire] = [
                    (desire.action_idx, None, None)
                ]  # Desire will get fulfilled here
                continue
            best_node, best_action, best_intention = None, None, 0
            possible_actions = self.get_possible_actions(state)
            for action in possible_actions:
                pos_sprima = self.get_possible_s_prima(state, action)
                for s_prima in pos_sprima:
                    desc_intention = self.get_intention(s_prima, desire)
                    if desc_intention > best_intention:
                        best_node, best_action, best_intention = (
                            s_prima,
                            action,
                            desc_intention,
                        )
                    elif desc_intention == best_intention:
                        # if there's a more probable action choose that one
                        act_prob = self.get_action_probability(state)
                        if act_prob[action] > act_prob.get(best_action, 0):
                            best_node, best_action, best_intention = (
                                s_prima,
                                action,
                                desc_intention,
                            )

            tail = self.answer_how(best_node, [desire])[desire]
            how_paths_per_desire[desire] = [
                (best_action, best_node, best_intention)
            ] + tail
        return how_paths_per_desire

    def answer_why(
        self,
        state: StateRepresentation,
        action: Action,
        minimum_probability_of_increase: float = 0,
    ) -> Dict[Desire, WhyTrace]:
        """Answers the question: Why was an action taken in a given state?"""
        # minimum_probability_of_increase: minimum probability of intention increase by which we attribute the agent
        # is trying to further an intention. eg: if it has 5% prob of increasing a desire but 95% of decreasing it
        current_intentions = self.get_intentions(state)
        current_attr_ints = {
            d: I_d for d, I_d in current_intentions.items() if I_d >= self.c_threshold
        }
        if len(current_attr_ints) == 0:
            return {}
        else:
            successors = [
                (
                    s_pr,
                    self.prob(ProbQuery(s_prima=s_pr, given_a=action, given_s=state)),
                    self.get_intentions(s_pr),
                )
                for s_pr in self.get_possible_s_prima(state, action)
            ]

            int_increase: Dict[Desire, WhyTrace] = {}
            for d, curr_int in current_attr_ints.items():
                if self.check_desire(state, d) is not None:
                    if action == d.action_idx:
                        int_increase[d] = "fulfilled"
                        continue
                # For each desire attributed, compute expected increase (sum_s' P(s'|a,s)*I_d(s') -I_d(s) ),
                # probability of increase ( P(s'|a,s) iff I_d(s')>=I_d(s) ) and expected increase (but only if it is >0)
                int_increase[d] = dict()
                int_increase[d]["expected"] = 0
                int_increase[d]["prob_increase"] = 0
                int_increase[d]["expected_pos_increase"] = 0
                for _, p, ints in successors:
                    int_increase[d]["expected"] += p * ints[d]
                    int_increase[d]["prob_increase"] += p if ints[d] >= curr_int else 0
                    int_increase[d]["expected_pos_increase"] += (
                        p * ints[d] if ints[d] >= curr_int else 0
                    )
                int_increase[d]["expected"] -= curr_int
                int_increase[d]["expected_pos_increase"] = (
                    int_increase[d]["expected_pos_increase"]
                    / int_increase[d]["prob_increase"]
                    if int_increase[d]["prob_increase"] > 0
                    else 0
                )
                int_increase[d]["expected_pos_increase"] -= curr_int
                if int_increase[d]["prob_increase"] <= minimum_probability_of_increase:
                    # Action detracts from intention, or has too small a probability to increase to be considered
                    del int_increase[d]
            return int_increase
