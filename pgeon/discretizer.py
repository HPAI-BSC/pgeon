import abc
from enum import Enum
from typing import TypeVar, Sequence

_Enum = TypeVar('_Enum', bound=Enum)


class Predicate:
    def __init__(self, predicate, value):
        self.predicate: _Enum = predicate
        self.value: Sequence[_Enum] = value

    def __eq__(self, other):
        if not isinstance(other, Predicate):
            return False
        return self.predicate == other.predicate and self.value == other.value

    def __str__(self):
        return f'{self.predicate.__name__}({";".join(str(pred.name) for pred in self.value)})'

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(str(self))


class Discretizer(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, 'discretize') \
           and callable(subclass.discretize) \
           and hasattr(subclass, 'state_to_str') \
           and callable(subclass.state_to_str) \
           and hasattr(subclass, 'str_to_state') \
           and callable(subclass.str_to_state) \
           and hasattr(subclass, 'nearest_state') \
           and callable(subclass.nearest_state)

    @abc.abstractmethod
    def discretize(self, state):
        pass

    @abc.abstractmethod
    def state_to_str(self, state) -> str:
        pass

    @abc.abstractmethod
    def str_to_state(self, state: str):
        pass

    @abc.abstractmethod
    def nearest_state(self, state):
        pass

    @abc.abstractmethod
    def all_actions(self):
        pass

    @abc.abstractmethod
    def get_predicate_space(self):
        pass
