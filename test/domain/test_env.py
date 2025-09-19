from enum import Enum, auto
from typing import Any, List, SupportsFloat, Tuple

import gymnasium
import numpy as np
from gymnasium.core import ActType, ObsType

from pgeon import Agent, Discretizer, Predicate
from pgeon.discretizer import PredicateBasedState


class DummyState(Enum):
    ZERO = auto()
    ONE = auto()
    TWO = auto()
    THREE = auto()
    FOUR = auto()
    FIVE = auto()
    SIX = auto()
    SEVEN = auto()
    EIGHT = auto()


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


class TestingDiscretizer(Discretizer):
    def discretize(self, state: np.ndarray) -> Tuple[Predicate]:
        correct_predicate = [
            DummyState.ZERO,
            DummyState.ONE,
            DummyState.TWO,
            DummyState.THREE,
        ][state[0]]
        return (Predicate(correct_predicate),)

    def all_actions(self):
        return [0]

    def nearest_state(self, state: PredicateBasedState):
        while True:
            value = state[0].value[0].value - 1
            yield (
                Predicate(
                    [DummyState.ZERO, DummyState.ONE, DummyState.TWO, DummyState.THREE][
                        value % 4
                    ]
                ),
            )

    def get_predicate_space(self) -> List[Tuple[Predicate, ...]]:
        return (DummyState,)

    def state_to_str(self, state: PredicateBasedState):
        return list(state.predicates)[0].value.name

    def str_to_state(self, state_str: str) -> PredicateBasedState:
        return PredicateBasedState((Predicate(DummyState[state_str]),))


class TestingAgent(Agent):
    def act(self, _):
        return 0
