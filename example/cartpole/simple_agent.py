"""
Simple agent for CartPole demonstration that doesn't require large checkpoint files.
This provides a basic policy that can be used to demonstrate the pgeon library.
"""

import numpy as np

from pgeon import Agent


class SimpleCartpoleAgent(Agent):
    """
    A simple agent for CartPole that implements a basic policy.
    This agent moves left when the pole is falling left, and right when falling right.
    """

    def __init__(self):
        pass

    def act(self, state):
        """
        Simple policy: move left when pole is falling left, right when falling right.

        Args:
            state: CartPole state [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

        Returns:
            action: 0 (left) or 1 (right)
        """
        # Extract pole angle and angular velocity
        pole_angle = state[2]
        pole_angular_velocity = state[3]

        # Simple policy: move in the direction the pole is falling
        # If pole is leaning left (negative angle) or moving left (negative angular velocity), move left
        if pole_angle < 0 or pole_angular_velocity < 0:
            return 0  # Move left
        else:
            return 1  # Move right


class RandomCartpoleAgent(Agent):
    """
    A random agent for CartPole that can be used for demonstration.
    """

    def __init__(self):
        pass

    def act(self, state):
        """
        Random policy for demonstration.

        Args:
            state: CartPole state

        Returns:
            action: Random 0 or 1
        """
        return np.random.randint(0, 2)


# Alternative: Create a simple trained model using stable-baselines3
try:
    import os

    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    class StableBaselinesAgent(Agent):
        """
        A simple agent using stable-baselines3 for demonstration.
        This creates a small trained model that can be saved and loaded easily.
        """

        def __init__(self, model_path=None, train_episodes=1000):
            if model_path and os.path.exists(model_path):
                self.model = PPO.load(model_path)
            else:
                # Create and train a simple model
                env = gym.make("CartPole-v1")
                env = DummyVecEnv([lambda: env])
                self.model = PPO("MlpPolicy", env, verbose=0)
                self.model.learn(total_timesteps=train_episodes)

                # Save the model if path provided
                if model_path:
                    self.model.save(model_path)

        def act(self, state):
            """
            Get action from the trained model.

            Args:
                state: CartPole state

            Returns:
                action: Model prediction
            """
            action, _ = self.model.predict(state, deterministic=True)
            return action

except ImportError:
    # If stable-baselines3 is not available, use the simple agent
    StableBaselinesAgent = SimpleCartpoleAgent
