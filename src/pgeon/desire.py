from typing import Set

from pgeon.discretizer import Predicate


class Goal:
    def __init__(self, name: str, clause: Set[Predicate]):
        self.name = name
        self.clause = clause


class Desire(object):
    def __init__(self, name: str, action_idx: str, clause: Set[Predicate]):
        self.name = name
        self.action_idx = action_idx
        self.clause = clause
        self.type = "achievement"

    def __repr__(self):
        return f"Desire[{self.name}]=<{self.clause}>|{self.action_idx}"

    def __str__(self):
        return f"Desire[{self.name}]=<{self.clause}>|{self.action_idx}"


Any = Desire("any", None, set())
