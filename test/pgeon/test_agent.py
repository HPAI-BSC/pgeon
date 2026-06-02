import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
from gymnasium.spaces import Discrete

from pgeon.agent import (
    Agent,
    CallableAgent,
    TorchPolicy,
    RandomAgent,
)


class DummyAgent(Agent):
    def act(self, observation):
        return "dummy_action"


class DummyPolicyObject:
    def act(self, observation):
        return "object_action"


class DummyTorchModel(torch.nn.Module):
    def forward(self, x):
        # Return logits for 3 actions
        return torch.tensor([[0.1, 0.9, 0.0]], device=x.device)


class TestCallableAgent(unittest.TestCase):
    def test_initialization_with_agent(self):
        agent = DummyAgent()
        callable_agent = CallableAgent(agent)

        self.assertEqual(callable_agent.act(None), "dummy_action")

    def test_initialization_with_callable(self):
        def policy(obs):
            return "callable_action"

        callable_agent = CallableAgent(policy)

        self.assertEqual(callable_agent.act(None), "callable_action")

    def test_initialization_with_object_act_method(self):
        policy_obj = DummyPolicyObject()
        callable_agent = CallableAgent(policy_obj)

        self.assertEqual(callable_agent.act(None), "object_action")

    def test_invalid_policy_raises_type_error(self):
        with self.assertRaises(TypeError):
            CallableAgent(123)

    def test_act_delegates_correctly(self):
        mock_policy = MagicMock(return_value="mocked_action")
        callable_agent = CallableAgent(mock_policy)

        result = callable_agent.act("observation")

        mock_policy.assert_called_once_with("observation")
        self.assertEqual(result, "mocked_action")


class TestTorchPolicy(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = DummyTorchModel()
        self.policy = TorchPolicy(self.model, self.device)

    def test_act_returns_argmax_action(self):
        observation = np.array([1.0, 2.0, 3.0])

        action = self.policy.act(observation)

        self.assertEqual(action, 1)

    def test_observation_converted_to_tensor(self):
        observation = np.array([1.0, 2.0])

        with unittest.mock.patch(
            "torch.tensor", wraps=torch.tensor
        ) as mock_tensor:
            self.policy.act(observation)

            mock_tensor.assert_any_call(
                observation,
                dtype=torch.float32,
                device=self.device,
            )

    def test_model_called(self):
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.2, 0.8]])

        policy = TorchPolicy(mock_model, self.device)

        observation = np.array([1.0, 2.0])
        policy.act(observation)

        self.assertTrue(mock_model.called)


class TestRandomAgent(unittest.TestCase):
    def setUp(self):
        self.action_space = Discrete(5)
        self.agent = RandomAgent(self.action_space)

    def test_initialization(self):
        self.assertEqual(self.agent.action_space, self.action_space)

    def test_act_returns_valid_action(self):
        observation = None

        action = self.agent.act(observation)

        self.assertIn(action, range(self.action_space.n))

    def test_act_calls_sample(self):
        mock_action_space = MagicMock()
        mock_action_space.sample.return_value = 3

        agent = RandomAgent(mock_action_space)

        action = agent.act(None)

        mock_action_space.sample.assert_called_once()
        self.assertEqual(action, 3)


if __name__ == "__main__":
    unittest.main()