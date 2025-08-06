from abc import ABC, abstractmethod
from typing import Any

from gymnasium import Space


class Agent(ABC):
    @abstractmethod
    def act(self, observation: Any) -> Any: ...


class RandomAgent(Agent):
    def __init__(self, action_space: Space):
        self.action_space = action_space

    def act(self, observation: Any) -> Any:
        return self.action_space.sample()
