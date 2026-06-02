from abc import ABC, abstractmethod
from typing import Any

from gymnasium import Space

import torch
import numpy as np


class Agent(ABC):
    @abstractmethod
    def act(self, observation: Any) -> Any: ...


class CallableAgent(Agent):
    """
    Adapter that turns any callable or object with `.act()`
    into a valid PGEON Agent.
    """

    def __init__(self, policy):
        # If it's already an Agent, just reuse its act
        if isinstance(policy, Agent):
            self._act_fn = policy.act
            return

        # Case 1: function / callable
        if callable(policy):
            self._act_fn = policy
            return

        # Case 2: object with .act()
        if hasattr(policy, "act") and callable(policy.act):
            self._act_fn = policy.act
            return

        raise TypeError(
            "Provided object is neither callable or has not an 'act' method."
        )

    def act(self, observation: Any) -> Any:
        return self._act_fn(observation)


class TorchPolicy:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def act(self, observation):
        # Convert numpy to tensor
        state_t = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            action = self.model(state_t).argmax(dim=1).item()

        return action


class RandomAgent(Agent):
    def __init__(self, action_space: Space):
        self.action_space = action_space

    def act(self, observation: Any) -> Any:
        return self.action_space.sample()