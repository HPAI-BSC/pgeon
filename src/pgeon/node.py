from typing import Dict, List, Set

import numpy as np


class Node:
    def __init__(
        self, node_id: int, prob: float, state_rep, nodes: List, transitions: Dict, pg
    ):
        self.probability = prob
        self.state_rep = state_rep
        self.nodes = nodes
        self.transitions = transitions
        self.node_id = node_id
        self.coinciders = set()
        self.intention = dict()
        self.pg = pg

    def check_desire(self, desire_clause: Set, action_id: int):
        raise NotImplementedError()

    def atom_in_state(self, atom):
        raise NotImplementedError()

    def get_action_probability(self):
        try:
            p_dist = self.transitions[self.node_id]
            action_prob_distro = dict()
            for action_idx, dest_state_distr in p_dist.items():
                action_prob_distro[action_idx] = sum(
                    [p for p in dest_state_distr.values()]
                )
            return action_prob_distro
        except KeyError:
            return dict()

    def max_intention(self):
        return max(list(self.intention.values()) + [0])

    def sample_next_state(self):
        try:
            transitions = self.transitions[self.node_id]
            result = {
                (action, next_id): p
                for action, possible_nexts in transitions.items()
                for next_id, p in possible_nexts.items()
            }
            result_idx = np.random.choice(
                len(result.keys()), 1, p=list(result.values())
            )
            s_action, s_next = list(result.keys())[result_idx[0]]
            return s_action, s_next
        except KeyError:
            raise ValueError(
                f"State {self.node_id} has no known successors in the edge file"
            )

    def check_out_transitions(self):
        return self.transitions[self.node_id]

    def descendants(self, action=None):  # List[Node, str(action)]
        if action is not None:
            return [
                (self.nodes[descendant_idx], action)
                for descendant_idx in self.transitions[self.node_id][action].keys()
            ]
        else:
            desc_list = []
            for action in self.transitions[self.node_id].keys():
                desc_list += self.descendants(action)
            return desc_list

    def get_attributed_intentions(self, c_threshold):
        return {d: val for d, val in self.intention.items() if val >= c_threshold}

    def answer_how(
        self, desires, stochastic=False, c_threshold=None, num_paths_per_desire=10
    ):
        # C_threshold and num_paths are only used if stochastic
        if stochastic:
            to_return = dict()
            for d in desires:
                d_name = d.name
                to_return[d_name] = {"SUCCESS": [], "FAILURE": []}
            for i in range(num_paths_per_desire):
                paths = self._answer_how_stochastic(desires, c_threshold)
                for d in desires:
                    d_name = d.name
                    path_state = paths[d_name][-1]
                    to_return[d_name][path_state].append(paths[d_name][:-1])
            return to_return  # dict per desire, dict per state, of paths
        else:
            how_paths_per_desire = dict()
            for desire in desires:
                clause, action_idx = desire.clause, desire.action_idx
                d_name = desire.name
                if (
                    self.check_desire(desire_clause=clause, action_id=action_idx)
                    is not None
                ):
                    how_paths_per_desire[d_name] = [
                        [action_idx, None, None]
                    ]  # Desire will get fulfilled here
                    continue
                best_node, best_action, best_intention = None, None, 0
                for descendant, action in self.descendants():
                    desc_intention = descendant.intention[d_name]
                    if desc_intention > best_intention:
                        best_node, best_action, best_intention = (
                            descendant,
                            action,
                            desc_intention,
                        )
                    elif desc_intention == best_intention:
                        # if there's a more probable action choose that one
                        action_probabilities = self.get_action_probability()
                        if (
                            action_probabilities[action]
                            > action_probabilities[best_action]
                        ):
                            best_node, best_action, best_intention = (
                                descendant,
                                action,
                                desc_intention,
                            )
                tail = best_node.answer_how([desire])[d_name]
                how_paths_per_desire[d_name] = [
                    [best_action, best_node, best_intention]
                ] + tail
            return how_paths_per_desire

    def answer_why(self, action_idx: str, c_threshold, probability_threshold=0):
        # probability_threshold: minimum probability of intention increase by which we attribute the agent is trying to
        # further an intention. eg: if it has 5% prob of increasing a desire but 95% of decreasing it
        attr_ints = {d: I_d for d, I_d in self.intention.items() if I_d >= c_threshold}
        if len(attr_ints) == 0:
            return {}
        else:
            successors = [
                (s, p, self.nodes[s].intention)
                for s, p in self.transitions[self.node_id][action_idx].items()
            ]
            p_a_s = (
                0  # Prob of a given s to scale later (as successors contain P(s',a|s) )
            )
            for _, p, _ in successors:
                p_a_s += p
            successors = [
                (s, p / p_a_s, i) for s, p, i in successors
            ]  # successors contain P(s'|a,s)
            int_increase = {}
            for d, val in attr_ints.items():
                int_increase[d] = dict()
                int_increase[d]["expected"] = 0
                int_increase[d]["prob_increase"] = 0
                int_increase[d]["expected_pos_increase"] = 0
                for _, p, ints in successors:
                    int_increase[d]["expected"] += p * ints[d]
                    int_increase[d]["prob_increase"] += p if ints[d] >= val else 0
                    int_increase[d]["expected_pos_increase"] += (
                        p * ints[d] if ints[d] >= val else 0
                    )
                int_increase[d]["expected"] -= val
                int_increase[d]["expected_pos_increase"] = (
                    int_increase[d]["expected_pos_increase"]
                    / int_increase[d]["prob_increase"]
                    if int_increase[d]["prob_increase"] > 0
                    else 0
                )
                int_increase[d]["expected_pos_increase"] -= val
                if int_increase[d]["prob_increase"] <= probability_threshold:
                    # Action detracts from intention. If threshold =0, it always detracts. Else: it has at least
                    # 1-threshold probability of decreasing intention.
                    del int_increase[d]
            return int_increase

    def compute_differences(self, node):
        raise NotImplementedError()

    def _answer_how_stochastic(self, desires, c_threshold):
        how_paths_per_desire = dict()
        for desire in desires:
            d_name = desire.name
            clause, action_idx = desire.clause, desire.action_idx
            if (
                self.check_desire(desire_clause=clause, action_id=action_idx)
                is not None
            ):
                how_paths_per_desire[d_name] = [
                    [action_idx, None, None],
                    "SUCCESS",
                ]  # Desire will get fulfilled here
                continue
            if self.intention[d_name] < c_threshold:
                how_paths_per_desire[d_name] = ["FAILURE"]
                continue
            action, next_state_id = self.sample_next_state()
            next_state: Node = self.nodes[next_state_id]
            tail = next_state._answer_how_stochastic([desire], c_threshold)[d_name]
            how_paths_per_desire[d_name] = [
                [action, next_state, next_state.intention[d_name]]
            ] + tail
        return how_paths_per_desire


class PropoNode(Node):
    def __init__(
        self, node_id: int, prob: float, state_rep, nodes: List, transitions: Dict, pg
    ):  # TODO: state_rep set of enum
        super().__init__(node_id, prob, state_rep, nodes, transitions, pg)
        self.probability = prob
        self.state_rep = set(state_rep.split("+"))  # temp

    @classmethod
    def from_graph_node(cls, node):
        probability = node.probability
        raise NotImplementedError("Pending integration with nx")
        # return PropoNode(node, probability, state_rep=node, nodes=None, transitions=None, pg=None)

    def check_desire(self, desire_clause: Set, action_id: int):
        # Returns None if desire is not satisfied. Else, returns probability of fulfilling desire
        #   ie: executing the action when in Node
        desire_clause_satisfied = True
        for atom in desire_clause:
            desire_clause_satisfied = desire_clause_satisfied and self.atom_in_state(
                atom
            )
            if not desire_clause_satisfied:
                return None
        return self.get_action_probability().get(action_id, 0)

    def compute_differences(self, node: Node):
        self_rep, other_rep = self.state_rep, node.state_rep
        shared = self_rep.intersection(other_rep)
        added = other_rep - shared
        removed = self_rep - shared
        return shared, added, removed

    def atom_in_state(self, atom):
        return atom in self.state_rep
