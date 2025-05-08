from typing import Dict, List, Tuple

from pgeon.desire import Desire
from pgeon.discretizer import PredicateBasedStateRepresentation
from pgeon.intention_aware_policy_graph import IPG

StateDiferences = Tuple[List[str], List[str]]
Action = str
State = PredicateBasedStateRepresentation
Intention = float
HowTrace = List[StateDiferences, Action, Intention]


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
    ) -> Dict[Desire:HowTrace]: ...

    def answer_why(self, state: State, action: Action): ...

    def answer_how_stochastic(
        self, desires: List[Desire], c_threshold: float, num_paths_per_desire: int = 10
    ) -> Dict[Desire : List[HowTrace]]: ...

    def get_attributed_intentions(self, desires: List[Desire], c_threshold: float): ...
