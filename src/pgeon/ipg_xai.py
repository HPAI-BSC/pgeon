from typing import Dict, List, Tuple

from pgeon.desire import Desire
from pgeon.discretizer import PredicateBasedStateRepresentation
from pgeon.intention_aware_policy_graph import IPG, ProbQuery

StateDiferences = Tuple[List[str], List[str]]
Action = str
State = PredicateBasedStateRepresentation
Intention = float
HowTrace = List[
    Tuple[Action, State, Intention]
]  # Doing action, arriving at state, which has intention
WhyTrace = Dict[str, float | str]


class IPG_XAI_analyser:
    def __init__(self, ipg: IPG, c_threshold: float):
        self.ipg = ipg
        self.c_threshold = c_threshold

    def answer_what(self, state: State, c_threshold: float) -> Dict[str, float]:
        return {
            dname: value
            for dname, value in self.ipg.policy_representation.get_state_attributes(
                "intention"
            )[state].items()
            if value >= c_threshold
        }

    def answer_how(
        self,
        state: State,
        desires: List[Desire],
    ) -> Dict[Desire, HowTrace]:
        how_paths_per_desire = dict()
        for desire in desires:
            x = self.ipg.check_desire(state, desire)
            if x is not None:
                how_paths_per_desire[desire] = [
                    [desire.action_idx, None, None]
                ]  # Desire will get fulfilled here
                continue
            best_node, best_action, best_intention = None, None, 0
            possible_actions = self.ipg.get_possible_actions(state)
            for action in possible_actions:
                pos_sprima = self.ipg.get_possible_s_prima(state, action)
                for s_prima in pos_sprima:
                    desc_intention = self.ipg.get_intention(s_prima, desire)
                    if desc_intention > best_intention:
                        best_node, best_action, best_intention = (
                            s_prima,
                            action,
                            desc_intention,
                        )
                    elif desc_intention == best_intention:
                        # if there's a more probable action choose that one
                        act_prob = self.ipg.get_action_probability(state)
                        if act_prob[action] > act_prob.get(best_action, 0):
                            best_node, best_action, best_intention = (
                                s_prima,
                                action,
                                desc_intention,
                            )

            tail = self.answer_how(best_node, [desire])[desire]
            how_paths_per_desire[desire] = [
                [best_action, best_node, best_intention]
            ] + tail
        return how_paths_per_desire

    def answer_why(
        self,
        state: State,
        action: Action,
        minimum_probability_of_increase: float = 0,
    ) -> Dict[Desire, WhyTrace]:
        # minimum_probability_of_increase: minimum probability of intention increase by which we attribute the agent
        # is trying to further an intention. eg: if it has 5% prob of increasing a desire but 95% of decreasing it
        current_intentions = self.ipg.get_intentions(state)
        current_attr_ints = {
            d: I_d for d, I_d in current_intentions.items() if I_d >= self.c_threshold
        }
        if len(current_attr_ints) == 0:
            return {}
        else:
            successors = [
                (
                    s_pr,
                    self.ipg.prob(
                        ProbQuery(s_prima=s_pr, given_a=action, given_s=state)
                    ),
                    self.ipg.get_intentions(s_pr),
                )
                for s_pr in self.ipg.get_possible_s_prima(state, action)
            ]

            int_increase = {}
            for d, curr_int in current_attr_ints.items():
                if self.ipg.check_desire(state, d) is not None:
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

    def answer_how_stochastic(
        self, desires: List[Desire], num_paths_per_desire: int = 10
    ) -> Dict[Desire, List[HowTrace]]: ...

    def get_attributed_intentions(self, state: State) -> Dict[Desire, Intention]:
        return {
            d: val
            for d, val in self.ipg.get_intentions(state).items()
            if val >= self.c_threshold
        }
