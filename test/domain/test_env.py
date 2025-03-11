from enum import auto, Enum
from typing import Tuple, Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import ObsType, ActType
from pgeon import Discretizer, Agent, Predicate


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
        if self.state[0] > 3: self.state = np.array([0])

        return self.state, 1, self.steps >= 30, self.steps >= 30, {}


class TestingAgent(Agent):
    def __init__(self):
        ...

    def act(self, _):
        return 0


class TestingDiscretizer(Discretizer):
    def __init__(self):
        super(TestingDiscretizer, self).__init__()

    def discretize(self,
                   state: np.ndarray
                   ) -> Tuple[Predicate]:
        correct_predicate = [State.ZERO, State.ONE, State.TWO, State.THREE][state[0]]
        return (Predicate(State, [correct_predicate]), )

    def all_actions(self):
        return [0]

    def get_predicate_space(self):
        ...

    def nearest_state(self, state):
        while True:
            value = state[0].value[0].value - 1
            yield (Predicate(State, [State.ZERO, State.ONE, State.TWO, State.THREE][value % 4]), )

    def state_to_str(self, state):
        return ""

    def str_to_state(self, state_str):
        ...