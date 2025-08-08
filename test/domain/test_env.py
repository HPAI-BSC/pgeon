from enum import Enum, auto
from typing import Any, List, SupportsFloat, Tuple

import gymnasium
import numpy as np
from gymnasium.core import ActType, ObsType

from pgeon import Agent, Discretizer, Predicate
from pgeon.discretizer import PredicateBasedStateRepresentation


class State(Enum):
    ZERO = auto()
    ONE = auto()
    TWO = auto()
    THREE = auto()


class TestingEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.state: np.ndarray = None
        self.steps: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.state = np.array([0])
        self.steps = 0
        return self.state, {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert action == 0

        self.steps += 1

        self.state += 1
        if self.state[0] > 3:
            self.state = np.array([0])

        return self.state, 1, self.steps >= 30, self.steps >= 30, {}


class TestingAgent(Agent):
    def __init__(self): ...

    def act(self, _):
        return 0


class TestingDiscretizer(Discretizer):
    def __init__(self):
        super(TestingDiscretizer, self).__init__()

    def discretize(self, state: np.ndarray) -> Tuple[Predicate]:
        correct_predicate = [State.ZERO, State.ONE, State.TWO, State.THREE][state[0]]
        return (Predicate(correct_predicate),)

    def all_actions(self):
        return [0]

    def get_predicate_space(self) -> List[Tuple[Predicate, ...]]: ...

    def nearest_state(self, state):
        while True:
            value = state[0].value[0].value - 1
            yield (
                Predicate([State.ZERO, State.ONE, State.TWO, State.THREE][value % 4]),
            )

    def state_to_str(self, state):
        return state.predicates[0].value.name

    def str_to_state(self, state_str):
        return PredicateBasedStateRepresentation((Predicate(State[state_str]),))
