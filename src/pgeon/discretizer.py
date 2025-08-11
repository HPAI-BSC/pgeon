from __future__ import annotations

import abc
from enum import Enum
from typing import Any, Collection, FrozenSet, Iterator, Sequence, Type, Union

from pydantic import BaseModel


class Predicate:
    def __init__(self, value: Enum):
        self.predicate: Type[Enum] = type(value)
        self.value: Enum = value

    def __eq__(self, other):
        if not isinstance(other, Predicate):
            return False
        return self.predicate == other.predicate and self.value == other.value

    def __str__(self):
        return f"{self.predicate.__name__}({self.value.name})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, other):
        if not isinstance(other, Predicate):
            raise ValueError
        else:
            return hash(self.predicate) < hash(other.predicate)


class State(abc.ABC):
    @abc.abstractmethod
    def __eq__(self, other: State) -> bool: ...

    @abc.abstractmethod
    def __hash__(self) -> int: ...


class PredicateBasedState(State):
    predicates: FrozenSet[Predicate]

    def __init__(self, predicates: Collection[Predicate]):
        self.predicates = frozenset(predicates)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "predicates" and hasattr(self, "predicates"):
            raise AttributeError("Cannot modify predicates")
        super().__setattr__(name, value)

    def __eq__(self, other: Union[PredicateBasedState, tuple[Predicate, ...]]) -> bool:
        if isinstance(other, tuple):
            if len(self.predicates) != len(other):
                return False
            return self.predicates == frozenset(other)

        if isinstance(other, PredicateBasedState):
            if len(self.predicates) != len(other.predicates):
                return False
            return self.predicates == other.predicates
        return False

    def __hash__(self):
        return hash(tuple(self.predicates))

    def __lt__(self, other):
        if not isinstance(other, PredicateBasedState):
            raise ValueError
        return hash(self.predicates) < hash(other.predicates)


# TODO: allow for more complex representations
Action = int


class Transition(BaseModel):
    action: Action
    probability: float = 0.0
    frequency: int = 0


class StateMetadata(BaseModel):
    probability: float = 0.0
    frequency: int = 0


class Discretizer(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "discretize")
            and callable(subclass.discretize)
            and hasattr(subclass, "state_to_str")
            and callable(subclass.state_to_str)
            and hasattr(subclass, "str_to_state")
            and callable(subclass.str_to_state)
            and hasattr(subclass, "nearest_state")
            and callable(subclass.nearest_state)
        )

    @abc.abstractmethod
    def discretize(self, state):
        pass

    @abc.abstractmethod
    def state_to_str(self, state) -> str:
        pass

    @abc.abstractmethod
    def str_to_state(self, state: str) -> State:
        pass

    @abc.abstractmethod
    def nearest_state(self, state) -> Iterator[Type[Enum]]:
        pass

    @abc.abstractmethod
    def all_actions(self) -> Sequence[Type[Enum]]:
        pass

    @abc.abstractmethod
    def get_predicate_space(self) -> Sequence[Type[Enum]]:
        pass
