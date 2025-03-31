import abc
from typing import List, Any, Collection, Optional, Union, Tuple
from gymnasium import Env
from enum import Enum

from pgeon.agent import Agent
from pgeon.discretizer import Discretizer, StateRepresentation
from pgeon.policy_representation import PolicyRepresentation


class Base:
    def __init__(self,
                 discretizer: Discretizer,
                 ):
        super().__init__()
        self.discretizer = discretizer

        self._is_fit = False
        self._trajectories_of_last_fit: List[List[Any]] = []

    def fit(self):
        ...


# INTENTIONAL POLICY GRAPHS = BASE POLICY GRAPH + INTENTION FUNCTIONALITY




class Desire:
    ...


class ProbabilityQuery:
    ...


class Action:
    ...


class PolicyApproximator(abc.ABC):
    def __init__(self,
                 discretizer: Discretizer,
                 policy_representation: PolicyRepresentation
                 ):
        self.discretizer: Discretizer = discretizer
        self.policy_represenation: PolicyRepresentation = policy_representation

    @abc.abstractmethod
    def save(self, format: str, path: Union[str, List[str]]):
        """Save the policy approximator"""
        ...

    @abc.abstractmethod
    def fit(self):
        ...


# From agent and environment
class OnlinePolicyApproximator(PolicyApproximator):
    ...


# From trajectories
class OfflinePolicyApproximator(PolicyApproximator):
    ...


class PolicyApproximatorFromBasicObservation(OnlinePolicyApproximator):
    def __init__(self,
                 discretizer: Discretizer,
                 policy_representation: PolicyRepresentation,
                 environment: Env,
                 agent: Agent
                 ):
        super().__init__(discretizer, policy_representation)
        self.environment = environment
        self.agent = agent

    def fit(self, n_episodes: int):
        assert n_episodes > 0, "The number of episodes must be a positive integer number!"

        for ep_i in range(n_episodes):

            episode_done = False
            # Alternative, more space-efficient albeit slightly less readable trajectory representation in comments
            episode_trajectory: List[Tuple[StateRepresentation, Action, StateRepresentation]] = []
            # episode_trajectory: List[Union[StateRepresentation, Action]] = []

            state = self.environment.reset()
            discretized_state = self.discretizer.discretize(state)

            # episode_trajectory = [discretized_state]

            while not episode_done:
                action = self.agent.act(state)

                # next_state, _, episode_done, _ = self.environment.step(action)

                # discretized_next_state = self.discretizer.discretize(next_state)
                # episode_trajectory.append((discretized_state, action, discretized_next_state))
                # episode_trajectory.extend([action, discretized_next_state])

                # discretized_state = discretized_next_state
                # TODO self.policy_represenation.update_representation(episode_trajectory)
            # TODO Current pgeon version stores the episode trajectory (discretized states).
            #      Consider whether we want to keep doing that.

            # TODO May `self.policy_representation` perform any post-process?

    def get_nearest_predicate(self, input_predicate: Tuple[Enum], verbose: bool = False):
        """Returns the nearest predicate. If already exists, returns same predicate."""
        raise NotImplementedError()

    def get_possible_actions(self, predicate):
        """Get possible actions and probabilities for a predicate"""
        raise NotImplementedError()

    def get_when_perform_action(self, action):
        """When do you perform action X?"""
        raise NotImplementedError()

    def question2(self, action, verbose: bool = False):
        """When do you perform action X?"""
        raise NotImplementedError()

    def substract_predicates(self, origin, destination):
        """Subtracts 2 predicates, getting only different values"""
        raise NotImplementedError()

    def nearby_predicates(self, state, greedy: bool = False, verbose: bool = False):
        """Gets nearby states from state"""
        raise NotImplementedError()

    def question3(self, predicate, action, greedy: bool = False, verbose: bool = False):
        """Why do you perform action X in state Y?"""
        raise NotImplementedError()

    def get_most_probable_option(self, predicate, greedy: bool = False, verbose: bool = False):
        """Get most probable action for a predicate"""
        raise NotImplementedError()


class InterventionalPGConstruction(PolicyApproximator):
    def fit(self):
        ...
