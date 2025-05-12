from typing import Dict, List, Tuple

from pgeon.desire import Desire
from pgeon.discretizer import PredicateBasedStateRepresentation
from pgeon.intention_aware_policy_graph import IPG

StateDiferences = Tuple[List[str], List[str]]
Action = str
State = PredicateBasedStateRepresentation
Intention = float
HowTrace = List[
    Tuple[Action, State, Intention]
]  # Doing action, arriving at state, which has intention


class IPG_XAI_analyser:
    def __init__(self, ipg: IPG):
        self.ipg = ipg

    def answer_what(self, state: State, c_threshold: float) -> Dict[str, float]:
        return {
            dname: value
            for dname, value in self.ipg.graph.nodes[state]["intention"].items()
            if value >= c_threshold
        }

    def answer_how(
        self, state: State, desires: List[Desire], c_threshold: float
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

            tail = self.answer_how(best_node, [desire], c_threshold)[desire]
            how_paths_per_desire[desire] = [
                [best_action, best_node, best_intention]
            ] + tail
        return how_paths_per_desire

    def answer_why(self, state: State, action: Action): ...

    def answer_how_stochastic(
        self, desires: List[Desire], c_threshold: float, num_paths_per_desire: int = 10
    ) -> Dict[Desire, List[HowTrace]]: ...

    def get_attributed_intentions(
        self, state: State, c_threshold: float
    ) -> Dict[Desire, Intention]:
        return {
            d: val
            for d, val in self.ipg.get_intentions(state).items()
            if val >= c_threshold
        }
