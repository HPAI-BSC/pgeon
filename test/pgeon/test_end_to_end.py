import unittest
from test.domain.cartpole import CartpoleDiscretizer

import gymnasium

from pgeon import (
    GraphRepresentation,
    PolicyApproximatorFromBasicObservation,
)
from pgeon.agent import RandomAgent


class TestEndToEnd(unittest.TestCase):
    def test_cartpole(self):
        env = gymnasium.make("CartPole-v1")
        discretizer = CartpoleDiscretizer()
        representation = GraphRepresentation()
        agent = RandomAgent(env.action_space)

        approximator = PolicyApproximatorFromBasicObservation(
            discretizer, representation, env, agent
        )

        approximator.fit(n_episodes=10)

        self.assertGreater(len(representation.get_all_states()), 0)
        self.assertGreater(len(representation.get_all_transitions()), 0)

        obs, _ = env.reset()
        for _ in range(100):
            action = agent.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break


if __name__ == "__main__":
    unittest.main()
