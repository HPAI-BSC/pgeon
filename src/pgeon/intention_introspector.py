from typing import Set, List, Dict, Optional

from pgeon import PolicyGraph, Predicate
from pgeon.desire import Desire
from pgeon.intention_aware_policy_graph import IntentionAwarePolicyGraph


class IntentionIntrospector(object):
    def __init__(self, desires: Set[Desire]):
        self.desires = desires

    def find_intentions(
        self, pg: PolicyGraph, commitment_threshold: float
    ) -> IntentionAwarePolicyGraph:
        iapg = IntentionAwarePolicyGraph(pg)
        discretizer = pg.discretizer
        intention_full_nodes = set()
        total_results = {
            k: [] for k in ["intention_probability", "expected_int_probability"]
        }
        self.register_all_desires(pg, self.desires, iapg)
        return iapg

    def atom_in_state(self, node: Set[Predicate], atom: Predicate):
        return atom in node

    @staticmethod
    def get_prob(unknown_dict: Optional[Dict[str, float]]) -> float:
        if unknown_dict is None:
            return 0
        else:
            return unknown_dict.get("probability", 0)

    def get_action_probability(
        self, pg: PolicyGraph, node: Set[Predicate], action_id: int
    ):
        try:
            destinations = pg[node]
            return sum(
                [
                    self.get_prob(pg.get_edge_data(node, destination, key=action_id))
                    for destination in destinations
                ]
            )
        except KeyError:
            print(
                f"Warning: State {node} has no sampled successors which were asked for"
            )
            return 0

    def check_desire(
        self,
        pg: PolicyGraph,
        node: Set[Predicate],
        desire_clause: Set[Predicate],
        action_id: int,
    ):
        # Returns None if desire is not satisfied. Else, returns probability of fulfilling desire
        #   ie: executing the action when in Node
        desire_clause_satisfied = True
        for atom in desire_clause:
            desire_clause_satisfied = desire_clause_satisfied and self.atom_in_state(
                node, atom
            )
            if not desire_clause_satisfied:
                return None
        return self.get_action_probability(pg, node, action_id)

    def update_intention(
        self,
        node: Set[Predicate],
        desire: Desire,
        probability: float,
        iapg: IntentionAwarePolicyGraph,
    ):
        if node not in iapg.intention:
            iapg.intention[node] = {}
        current_intention_val = iapg.intention[node].get(desire, 0)
        iapg.intention[node][desire] = current_intention_val + probability

    def propagate_intention(
        self,
        pg: PolicyGraph,
        node: Set[Predicate],
        desire: Desire,
        probability,
        iapg: IntentionAwarePolicyGraph,
        stop_criterion=1e-4,
    ):
        self.update_intention(node, desire, probability, iapg)
        for coincider in pg.predecessors(node):
            if (
                self.check_desire(
                    pg,
                    coincider,
                    desire.clause,
                    desire.action_idx or hash(desire.clause),
                )
                is None
            ):
                successors = pg.successors(coincider)
                coincider_transitions: List[Dict[Set[Predicate], float]] = [
                    {
                        successor: self.get_prob(
                            pg.get_edge_data(coincider, successor, key=action_id)
                        )
                        for successor in successors
                    }
                    for action_id in pg.discretizer.all_actions()
                ]
            else:
                successors = pg.successors(coincider)
                # If coincider can fulfill desire themselves, do not propagate it through the action_idx branch
                coincider_transitions: List[Dict[Set[Predicate], float]] = [
                    {
                        successor: self.get_prob(
                            pg.get_edge_data(coincider, successor, key=action_id)
                        )
                        for successor in successors
                    }
                    for action_id in pg.discretizer.all_actions()
                    if action_id != desire.action_idx
                ]

            prob_of_transition = 0
            for action_transitions in coincider_transitions:
                prob_of_transition += action_transitions.get(node, 0)
            # self.transitions = {n_idx: {action1:{dest_node1: P(dest1, action1|n_idx), ...}

            new_coincider_intention_value = prob_of_transition * probability
            if new_coincider_intention_value >= stop_criterion:
                try:
                    coincider.propagate_intention(desire, new_coincider_intention_value)
                except RecursionError:
                    print(
                        "Maximum recursion reach, skipping branch with intention of",
                        new_coincider_intention_value,
                    )

    def register_desire(
        self, pg: PolicyGraph, desire: Desire, iapg: IntentionAwarePolicyGraph
    ):
        for node in pg.nodes:
            p = self.check_desire(
                pg, node, desire.clause, desire.action_idx or hash(desire.clause)
            )
            if p is not None:
                self.propagate_intention(pg, node, desire, p, iapg)

    def register_all_desires(
        self, pg: PolicyGraph, desires: Set[Desire], iapg: IntentionAwarePolicyGraph
    ):
        for desire in desires:
            self.register_desire(pg, desire, iapg)
