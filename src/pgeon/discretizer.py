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
    PREDICATE_TYPES: Sequence[type[Enum]] = ()

    @abc.abstractmethod
    def discretize(self, non_discrete_state) -> State:
        ...

    @abc.abstractmethod
    def state_to_str(self, state: State) -> str:
        predicates = self._extract_predicates(state)

        ordered_predicates = sorted(
            predicates,
            key=lambda predicate: self._predicate_sort_key(predicate),
        )

        return "&".join(
            self._predicate_to_str(predicate)
            for predicate in ordered_predicates
        )

    @abc.abstractmethod
    def str_to_state(self, state_str: str) -> State:
        if not self.PREDICATE_TYPES:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define PREDICATE_TYPES "
                f"or implement str_to_state manually."
            )

        parsed_predicates = {}

        for raw_predicate in state_str.split("&"):
            if not raw_predicate.strip():
                continue

            predicate = self._str_to_predicate(raw_predicate)

            enum_value = self._predicate_enum_value(predicate)
            enum_type = type(enum_value)

            if enum_type in parsed_predicates:
                raise ValueError(
                    f"Duplicated predicate type {enum_type.__name__} "
                    f"in serialized state: {state_str}"
                )

            parsed_predicates[enum_type] = predicate

        missing_types = [
            enum_type.__name__
            for enum_type in self.PREDICATE_TYPES
            if enum_type not in parsed_predicates
        ]

        if missing_types:
            raise ValueError(
                f"Serialized state is missing predicates of type "
                f"{missing_types}: {state_str}"
            )

        return tuple(
            parsed_predicates[enum_type]
            for enum_type in self.PREDICATE_TYPES
        )

    def _extract_predicates(self, state: State) -> tuple[Predicate, ...]:
        if isinstance(state, PredicateBasedState):
            return tuple(state.predicates)

        if isinstance(state, tuple) and all(
            isinstance(predicate, Predicate)
            for predicate in state
        ):
            return state

        raise ValueError(
            f"Cannot serialize state of type {type(state).__name__}. "
            f"Expected PredicateBasedState or tuple[Predicate, ...]."
        )

    def _predicate_to_str(self, predicate: Predicate) -> str:
        enum_value = self._predicate_enum_value(predicate)
        enum_type = type(enum_value)

        return f"Predicate({enum_type.__name__}.{enum_value.name})"

    def _str_to_predicate(self, raw_predicate: str) -> Predicate:
        raw_predicate = raw_predicate.strip()

        if not raw_predicate.startswith("Predicate(") or not raw_predicate.endswith(")"):
            raise ValueError(f"Invalid predicate format: {raw_predicate}")

        content = raw_predicate[len("Predicate("):-1]

        if "." not in content:
            raise ValueError(f"Invalid predicate content: {content}")

        enum_type_name, enum_member_name = content.split(".", 1)

        for enum_type in self.PREDICATE_TYPES:
            if enum_type.__name__ == enum_type_name:
                try:
                    return Predicate(enum_type[enum_member_name])
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown enum member {enum_member_name} "
                        f"for enum type {enum_type_name}"
                    ) from exc

        raise ValueError(f"Unknown enum type: {enum_type_name}")

    def _predicate_sort_key(self, predicate: Predicate) -> int:
        enum_value = self._predicate_enum_value(predicate)
        enum_type = type(enum_value)

        if enum_type not in self.PREDICATE_TYPES:
            raise ValueError(f"Unknown predicate type: {enum_type}")

        return list(self.PREDICATE_TYPES).index(enum_type)

    def _predicate_enum_value(self, predicate: Predicate) -> Enum:
        value = predicate.name

        if not isinstance(value, Enum):
            raise ValueError(f"Predicate does not contain an enum name: {predicate}")

        return value

    @abc.abstractmethod
    def nearest_state(self, state) -> Iterator[State]:
        ...

    @abc.abstractmethod
    def all_actions(self) -> Sequence[Action]:
        ...

    @abc.abstractmethod
    def get_predicate_space(self) -> Sequence[Predicate]:
        ...