from typing import Any, Dict, Set

from pydantic import BaseModel, Field

from pgeon.discretizer import Action, Predicate, PredicateBasedState, StateMetadata


class Goal:
    def __init__(self, name: str, clause: Set[Predicate]):
        self.name = name
        self.clause = clause


class Desire(BaseModel):
    name: str = Field(frozen=True)
    action: Action = Field(frozen=True)
    clause: PredicateBasedState = Field(frozen=True)
    type: str = Field(frozen=True, default="achievement")

    class Config:
        frozen = True

    def __init__(self, name: str, action: Action, clause: PredicateBasedState):
        super().__init__(name=name, action=action, clause=clause)

    def __repr__(self) -> str:
        return f"Desire[{self.name}]=<{self.clause}>|{self.action}"

    def __str__(self) -> str:
        return f"Desire[{self.name}]=<{self.clause}>|{self.action}"

    def __hash__(self) -> int:
        return hash(self.name)


class IntentionalStateMetadata(StateMetadata):
    intention: dict[Desire, float] = Field(default_factory=dict, frozen=True)

    class Config:
        frozen = True

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        base_dict = {
            "frequency": self.frequency,
            "probability": self.probability,
            "intention": self.intention,
        }
        return base_dict
