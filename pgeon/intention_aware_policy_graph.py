from typing import Dict, Set

from pgeon import PolicyGraph, Predicate
from pgeon.desire import Desire


class IntentionAwarePolicyGraph(object):
    def __init__(self, pg: PolicyGraph):
        self.pg: PolicyGraph = pg
        self.intention: Dict[Set[Predicate], Dict[Desire, float]] = {}

    def __str__(self):
        return str(self.intention)
