from typing import Set, Optional

from pgeon import Predicate


class Desire(object):
    def __init__(self, name: str, action_idx: Optional[int], clause: Set[Predicate]):
        self.name = name
        self.action_idx = action_idx
        self.clause = clause

    def __repr__(self):
        return f"Desire[{self.name}]=<{self.clause}>|{self.action_idx}"

    def __str__(self):
        return f"Desire[{self.name}]=<{self.clause}>|{self.action_idx}"


Any = Desire("any", None, set())
