import os
import sys

import gymnasium as gym

# Add the example directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from discretizer import (
    Action,
    Angle,
    CartpoleDiscretizer,
    Position,
    Velocity,
)
from simple_agent import SimpleCartpoleAgent

from pgeon import (
    GraphRepresentation,
    PolicyApproximatorFromBasicObservation,
    Predicate,
    PredicateBasedState,
)

if __name__ == "__main__":
    environment = gym.make("CartPole-v1")
    discretizer = CartpoleDiscretizer()
    representation = GraphRepresentation()

    # Use the simple agent instead of loading large checkpoint files
    agent = SimpleCartpoleAgent()

    approximator = PolicyApproximatorFromBasicObservation(
        discretizer, representation, environment, agent
    )
    approximator.fit(n_episodes=100)

    # Get the policy representation
    pg = approximator.policy_representation

    print(f"Number of states: {len(list(pg.states))}")
    print(f"Number of transitions: {len(list(pg.transitions))}")

    arbitrary_state = next(iter(pg.states))

    print(arbitrary_state)
    print(f"  Times visited: {pg.states[arbitrary_state].metadata.frequency}")
    print(f"  p(s):          {pg.states[arbitrary_state].metadata.probability:.3f}")

    arbitrary_transition = next(iter(pg.transitions))
    # The transition is already a TransitionData object
    from_state = arbitrary_transition.from_state
    to_state = arbitrary_transition.to_state
    transition_data = arbitrary_transition

    print(f"From:    {from_state}")
    print(f"Action:  {transition_data.transition.action}")
    print(f"To:      {to_state}")
    print(f"  Times visited:      {transition_data.transition.frequency}")
    print(f"  p(s_to,a | s_from): {transition_data.transition.probability:.3f}")

    # Create a PredicateBasedState for the query
    query_state = PredicateBasedState(
        [
            Predicate(Position.MIDDLE),
            Predicate(Velocity.RIGHT),
            Predicate(Angle.STANDING),
        ]
    )

    possible_actions = approximator.question1(query_state)

    print("I will take one of these actions:")
    for action, prob in possible_actions:
        action_name = "LEFT" if action == 0 else "RIGHT"
        print(f"\t-> {action_name}\tProb: {round(prob * 100, 2)}%")

    best_states = approximator.question2(Action.LEFT.value)
    print(f"I will perform action {0} in these states:")
    print("\n".join([str(state) for state in best_states]))

    print(
        f"Supposing I was in the middle, moving right, with the pole standing upright, "
        f"if I did not choose to move left was due to..."
    )
    try:
        counterfactuals = approximator.question3(query_state, Action.LEFT)
        for ct in counterfactuals:
            print(
                f"...{' and '.join([str(i[0]) + ' -> ' + str(i[1]) for i in ct.values()])}"
            )
    except ValueError as e:
        print(f"Could not generate counterfactuals: {e}")
        print(
            "This might be because the query state is not well-represented in the policy graph."
        )
