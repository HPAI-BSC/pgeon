from dataclasses import dataclass, field
from typing import Any, Dict, Set

from pgeon.discretizer import Action, Predicate, PredicateBasedState, StateMetadata


class IntentionalStateMetadata(StateMetadata):
    intention: Dict[Any, float] = field(default_factory=dict)


class Goal:
    def __init__(self, name: str, clause: Set[Predicate]):
        self.name = name
        self.clause = clause


@dataclass(frozen=True)
class Desire:
    name: str
    action: Action
    clause: PredicateBasedState
    type: str = "achievement"

    def __repr__(self) -> str:
        return f"Desire[{self.name}]=<{self.clause}>|{self.action}"

    def __str__(self) -> str:
        return f"Desire[{self.name}]=<{self.clause}>|{self.action}"

    def __hash__(self) -> int:
        return hash(self.name)


Any = Desire("any", None, PredicateBasedState([]))
