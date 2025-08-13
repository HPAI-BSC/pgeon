import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import gymnasium as gym

from pgeon.agent import Agent
from pgeon.desire import Desire, Goal, IntentionalStateMetadata
from pgeon.discretizer import (
    Action,
    Discretizer,
    Predicate,
    PredicateBasedState,
    State,
    Transition,
)
from pgeon.policy_approximator import (
    PolicyApproximatorFromBasicObservation,
)
from pgeon.policy_representation import PolicyRepresentation

HowTrace = List[
    Tuple[Action, State, float]
]  # Doing action, arriving at state, which has intention
WhyTrace = Dict[str, float | str]


@dataclass(frozen=True)
class ProbQuery:
    s: Optional[State] = None
    a: Optional[Action] = None
    s_prima: Optional[State] = None
    given_s: Optional[State] = None
    given_a: Optional[Action] = None
    given_do_a: Optional[Action] = None

    def __post_init__(self):
        # CURRENT ACCEPTABLE USAGES #
        # s | a,given_s | s',a,given_s | s',given_a,given_s |s', given_do_a,given_s | s', s #
        assert any(
            [
                var is not None
                for var in [
                    self.s,
                    self.a,
                    self.s_prima,
                    self.given_s,
                    self.given_a,
                    self.given_do_a,
                ]
            ]
        )
        assert (
            self.s is None or self.given_s is None
        ), "Invalid usage, can't use s and given_s simultaneously"
        assert [v is None for v in [self.a, self.given_a, self.given_do_a]].count(
            False
        ) <= 1, "Invalid usage, can't use a, given_a, or given_do_a simultaneously"  # At most 1 not None
        assert self.s is None or all(
            v is None
            for v in [self.a, self.s_prima, self.given_s, self.given_a, self.given_do_a]
        )
        assert (
            self.given_a is None or self.s_prima is not None
        ), "given_a requires setting s_prima"
        assert (
            self.given_do_a is None or self.s_prima is not None
        ), "given_do_a requires setting s_prima"


class IntentionAwarePolicyApproximator(
    PolicyApproximatorFromBasicObservation[IntentionalStateMetadata]
):
    def __init__(
        self,
        discretizer: Discretizer,
        policy_representation: PolicyRepresentation[IntentionalStateMetadata],
        environment: gym.Env,
        agent: Agent,
        verbose=False,
    ):
        assert issubclass(
            policy_representation.state_metadata_class, IntentionalStateMetadata
        )
        super().__init__(discretizer, policy_representation, environment, agent)
        self.policy_representation: PolicyRepresentation[IntentionalStateMetadata] = (
            policy_representation
        )
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
        for state in self.policy_representation.states:
            state_metadata = self.policy_representation.states[state].metadata
            state_metadata.intention[desire] = 0.0
            self.policy_representation.states[state] = state_metadata

        for node in self.policy_representation.states:
            p = self.check_desire(node, desire)
            if p > 0:
                self.propagate_intention(node, desire, p, stop_criterion)

    def get_possible_actions(self, s: State) -> List[Action]:
        # Returns any a s.t. P(a|s)>0
        return [
            transition_data.action
            for transition_data in self.policy_representation.transitions[s]
        ]

    def get_possible_s_prima(self, s: State, a: Action = None) -> List[State]:
        # Returns any a s.t. P(a|s)>0
        transitions_with_nonzero_probability = [
            transition_data
            for transition_data in self.policy_representation.transitions[s]
            if transition_data.probability > 0
        ]
        # TODO: Check on this
        transitions_with_nonzero_probability = list(
            set(transitions_with_nonzero_probability)
        )  # deduplicate, for some reason needed (?)
        if a is not None:
            # Filter for edges annotated with action= a only
            destinies = list(
                set(
                    [
                        transition_data.to_state
                        for transition_data in transitions_with_nonzero_probability
                        if transition_data.action == a
                    ]
                )
            )
        else:
            destinies = list(
                set(
                    [
                        transition_data.to_state
                        for transition_data in transitions_with_nonzero_probability
                    ]
                )
            )
        return destinies

    def get_all_state_ids(self) -> Iterable[State]:
        return self.policy_representation.states

    def check_desire(self, node: PredicateBasedState, desire: Desire) -> float:
        if desire.clause.predicates.issubset(node.predicates):
            return 1.0
        return 0.0

    def get_intention(self, s: State, desire: Desire):
        try:
            return self.policy_representation.states[s].metadata.intention.get(
                desire, 0.0
            )
        except KeyError:
            return 0

    def get_intentions(self, s: State) -> Dict[Desire, float]:
        try:
            return self.policy_representation.states[s].metadata.intention
        except KeyError:
            return dict()

    def _set_intention(self, s, desire, new_int):
        node_data = self.policy_representation.states[s].metadata
        node_data.intention[desire] = new_int

    def _prob_s(self, s: State):
        return self.policy_representation.states[s].metadata.probability

    def _prob_s_prima_a_given_s(self, s_prima: State, a: Action, given_s: State):
        # Assuming the 's' is always in predicates format for simplicity
        try:
            prob = [
                transition_data.probability
                for transition_data in self.policy_representation.transitions[given_s]
                if a == transition_data.action and transition_data.to_state == s_prima
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

    def _prob_a_given_s(self, a: Action, given_s: State):
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
        transitions = list(self.policy_representation.transitions[given_s])
        prob = [
            transition_data.probability
            for transition_data in transitions
            if given_a == transition_data.action and transition_data.to_state == s_prima
        ]
        return sum(prob)

    def _prob_s_prima_given_do_a_given_s(
        self,
        s_prima: State,
        given_do_a: Action,
        given_s: State,
    ):
        raise NotImplementedError("Basic PG can't handle do(a)")

    def _prob_s_prima_given_s(self, s_prima, given_s):
        """p(s'|s)"""
        # TODO: this needs to be marginalized over actions
        transitions = list(self.policy_representation.transitions[given_s])
        prob = [
            transition_data.probability
            for transition_data in transitions
            if transition_data.to_state == s_prima
        ]
        return sum(prob)

    def stateID_to_node(self, s: State) -> dict[str, Any]:
        return self.policy_representation.states[s].metadata

    def propagate_intention(
        self,
        state: State,
        desire: Desire,
        propagated_intention: float,
        stop_criterion: float,
    ):
        self._update_intention(state, desire, propagated_intention)

        for parent in self.policy_representation.states[state].predecessors:
            prob_transition = self.prob(ProbQuery(s_prima=state, given_s=parent))
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

    def _update_intention(self, state: State, desire: Desire, intention_value: float):
        current_metadata = self.policy_representation.states[state].metadata
        intention = current_metadata.intention
        intention[desire] = 0 + intention_value
        updated_metadata = current_metadata.model_copy(update={"intention": intention})
        self.policy_representation.states[state] = updated_metadata

    def get_action_probability(self, state: State) -> Dict[Action, float]:
        # TODO: This should go in the representation parent class
        return {
            a: self.prob(ProbQuery(a=a, given_s=state))
            for a in self.get_possible_actions(state)
        }

    def answer_what(self, state: State) -> List[Tuple[Goal, float]]:
        """Answers the question: What are the intentions in a given state?"""
        intentions = self.policy_representation.states[state].metadata.intention
        if intentions:
            return list(intentions.items())
        return []

    def answer_how(
        self,
        state: State,
        desires: List[Desire],
    ) -> Dict[Desire, HowTrace]:
        """Answers the question: How to achieve a desire from a given state?"""
        how_paths_per_desire = dict()
        for desire in desires:
            x = self.check_desire(state, desire)
            if x > 0:
                how_paths_per_desire[desire] = [
                    (desire.action, None, None)
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
        state: State,
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
                if self.check_desire(state, d) > 0:
                    if action == d.action:
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
