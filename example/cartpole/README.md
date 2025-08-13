# CartPole Demo

This directory contains a demonstration of the pgeon library using the CartPole environment.

## Files

- `demo.py`: Python script demonstrating the fluent API (can be run from project root)
- `demo.ipynb`: Jupyter notebook with detailed examples
- `discretizer.py`: CartPole-specific discretizer implementation
- `simple_agent.py`: Simple agent implementations that don't require large checkpoint files
- `checkpoints/`: Directory containing large training artifacts (legacy)

## New Approach: Simple Agent

Instead of using large checkpoint files (which can be several MB), this demo now uses a simple agent implementation that provides a basic policy for CartPole. This approach has several benefits:

### Benefits

1. **No Large Files**: The simple agent doesn't require downloading or storing large checkpoint files
2. **Faster Setup**: No need to download pre-trained models or run long training sessions
3. **Educational**: The simple policy is easy to understand and demonstrates basic CartPole control
4. **Portable**: The entire demo can be run without external dependencies beyond the core pgeon library

### Agent Options

The `simple_agent.py` file provides several agent implementations:

1. **SimpleCartpoleAgent**: A basic policy that moves left when the pole is falling left, and right when falling right
2. **RandomCartpoleAgent**: A random agent for demonstration purposes
3. **StableBaselinesAgent**: A trained agent using stable-baselines3 (if available)

### Usage

```python
from example.cartpole.simple_agent import SimpleCartpoleAgent
from pgeon import PolicyApproximatorFromBasicObservation, GraphRepresentation
import gymnasium as gym

# Create environment and discretizer
environment = gym.make("CartPole-v1")
discretizer = CartpoleDiscretizer()
representation = GraphRepresentation()

# Use simple agent
agent = SimpleCartpoleAgent()

# Generate policy approximator
approximator = PolicyApproximatorFromBasicObservation(
    discretizer, representation, environment, agent
)
approximator.fit(n_episodes=10)

# Access the policy representation
pg = approximator.policy_representation
```

## Fluent API

The demo has been updated to use the new fluent API for policy representations:

```python
# Old API (deprecated)
print(f"Number of nodes: {len(pg.nodes)}")
print(f"Number of edges: {len(pg.edges)}")
state_data = pg.nodes[state]["frequency"]

# New Fluent API
print(f"Number of states: {len(list(pg.states))}")
print(f"Number of transitions: {len(list(pg.transitions))}")
state_data = pg.states[state].metadata.frequency
```

## Legacy Checkpoints

The `checkpoints/` directory contains large training artifacts from previous versions. These files are no longer needed for the demo and can be safely removed to save space.

## Running the Demo

```bash
# Run from project root
uv run python example/cartpole/demo.py

# Or open the Jupyter notebook
jupyter notebook demo.ipynb
```

The demo will generate a policy graph using the simple agent and demonstrate various analysis capabilities of the pgeon library, including:

- State and transition analysis
- Action probability queries
- State-action mapping
- Counterfactual reasoning (when applicable)

## Demo Output

The demo shows:
- Number of states and transitions in the policy graph
- Example state metadata (frequency and probability)
- Example transition data (action, frequency, probability)
- Possible actions from a given state with probabilities
- States where a specific action is most likely to be taken
- Counterfactual reasoning (when the query state is well-represented)
