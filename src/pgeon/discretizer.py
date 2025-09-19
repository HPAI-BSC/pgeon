from __future__ import annotations

import abc
from enum import Enum
from typing import Collection, FrozenSet, Iterator, Sequence, Tuple, Union

from pydantic import BaseModel, Field


class Predicate(BaseModel):
    name: Enum
    arguments: Tuple[Enum, ...]

    def __init__(self, name: Enum, arguments: Tuple[Enum, ...] | None = None):
        super().__init__(name=name, arguments=arguments or ())

    def __str__(self):
        return f"{self.name.name}({';'.join([arg.name for arg in self.arguments])})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, other):
        if not isinstance(other, Predicate):
            raise ValueError
        else:
            return str(self) < str(other)


class State(abc.ABC):
    @abc.abstractmethod
    def __eq__(self, other: State) -> bool: ...

    @abc.abstractmethod
    def __hash__(self) -> int: ...


class PredicateBasedState(State, BaseModel):
    predicates: FrozenSet[Predicate] = Field(frozen=True)

    def __init__(self, predicates: Collection[Predicate]):
        super().__init__(predicates=frozenset(predicates))

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

    def __str__(self):
        return "+".join(sorted([str(p) for p in self.predicates]))

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.predicates)

    def __lt__(self, other):
        if not isinstance(other, PredicateBasedState):
            raise ValueError
        return hash(self.predicates) < hash(other.predicates)


Action = int


class Transition(BaseModel):
    action: Action
    probability: float = 0.0
    frequency: int = 0


class TransitionData:
    """A wrapper around Transition that includes from_state and to_state for fluent API."""

    def __init__(self, transition: Transition, from_state: State, to_state: State):
        self.transition = transition
        self.from_state = from_state
        self.to_state = to_state

    @property
    def action(self) -> Action:
        return self.transition.action

    @property
    def probability(self) -> float:
        return self.transition.probability

    @property
    def frequency(self) -> int:
        return self.transition.frequency

    def __getattr__(self, name):
        """Delegate other attributes to the underlying transition."""
        return getattr(self.transition, name)


class StateMetadata(BaseModel):
    probability: float = 0.0
    frequency: int = 0


class Discretizer(abc.ABC):
    @abc.abstractmethod
    def discretize(self, non_discrete_state) -> State: ...

    @abc.abstractmethod
    def state_to_str(self, state: State) -> str: ...

    @abc.abstractmethod
    def str_to_state(self, state_str: str) -> Predicate: ...

    @abc.abstractmethod
    def nearest_state(self, state) -> Iterator[State]: ...

    @abc.abstractmethod
    def all_actions(self) -> Sequence[Action]: ...

    @abc.abstractmethod
    def get_predicate_space(self) -> Sequence[Predicate]: ...
