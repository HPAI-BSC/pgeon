import abc
import random
from collections import defaultdict
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
import tqdm
from gymnasium import Env

from pgeon.agent import Agent
from pgeon.discretizer import Action, Discretizer, StateRepresentation
from pgeon.policy_representation import PolicyRepresentation


class Desire: ...


class PolicyApproximator(abc.ABC):

    def __init__(
        self, discretizer: Discretizer, policy_representation: PolicyRepresentation
    ):
        self.discretizer: Discretizer = discretizer
        self.policy_representation: PolicyRepresentation = policy_representation
        self._is_fit = False
        self._trajectories_of_last_fit: List[List[Any]] = []

    @abc.abstractmethod
    def save(self, format: str, path: Union[str, List[str]]):
        """Save the policy approximator"""
        ...

    @abc.abstractmethod
    def fit(self): ...


# From agent and environment
class OnlinePolicyApproximator(PolicyApproximator): ...


# From trajectories
class OfflinePolicyApproximator(PolicyApproximator): ...


class PolicyApproximatorFromBasicObservation(OnlinePolicyApproximator):
    """
    Policy approximator that approximates the policy from basic observations,
    given an agent and an environment.
    """

    def __init__(
        self,
        discretizer: Discretizer,
        policy_representation: PolicyRepresentation,
        environment: Env,
        agent: Agent,
    ):
        super().__init__(discretizer, policy_representation)
        self.environment = environment
        self.agent = agent

    def _run_episode(
        self, agent: Agent, max_steps: Optional[int] = None, seed: Optional[int] = None
    ) -> List[Any]:
        observation, _ = self.environment.reset(seed=seed)
        done = False
        trajectory = [self.discretizer.discretize(observation)]

        step_counter = 0
        while not done:
            if max_steps is not None and step_counter >= max_steps:
                break

            action = agent.act(observation)
            observation, _, done, done2, _ = self.environment.step(action)
            done = done or done2

            trajectory.extend([action, self.discretizer.discretize(observation)])

            step_counter += 1

        return trajectory

    def _update_with_trajectory(self, trajectory: List[Any]) -> None:
        # Only even numbers are states
        states_in_trajectory = [
            trajectory[i] for i in range(len(trajectory)) if i % 2 == 0
        ]
        all_new_states_in_trajectory = {
            state
            for state in set(states_in_trajectory)
            if not self.policy_representation.has_state(state)
        }
        self.policy_representation.add_states_from(
            all_new_states_in_trajectory, frequency=0
        )

        state_frequencies = {
            s: states_in_trajectory.count(s) for s in set(states_in_trajectory)
        }

        states = self.policy_representation.get_all_states()
        for state in state_frequencies:
            for node in states:
                if node == state:
                    state_attrs = self.policy_representation.get_state_attributes(
                        "frequency"
                    )
                    state_attrs[node] += state_frequencies[state]
                    self.policy_representation.set_state_attributes(
                        {node: state_attrs[node]}, "frequency"
                    )
                    break

        pointer = 0
        while (pointer + 1) < len(trajectory):
            state_from, action, state_to = trajectory[pointer : pointer + 3]
            if not self.policy_representation.has_transition(
                state_from, state_to, action
            ):
                self.policy_representation.add_transition(
                    state_from, state_to, action, frequency=0
                )

            edge_data = self.policy_representation.get_transition_data(
                state_from, state_to, action
            )
            if edge_data:
                edge_data["frequency"] += 1
            pointer += 2

    def _normalize(self) -> None:
        weights = self.policy_representation.get_state_attributes("frequency")
        total_frequency = sum([weights[state] for state in weights])
        self.policy_representation.set_state_attributes(
            {state: weights[state] / total_frequency for state in weights},
            "probability",
        )

        for state in self.policy_representation.get_all_states():
            transitions = self.policy_representation.get_outgoing_transitions(
                state, include_data=True
            )
            total_frequency = 0

            for transition in transitions:
                if len(transition) >= 3:  # Check if we have data
                    _, _, data = transition
                    if isinstance(data, dict) and "frequency" in data:
                        total_frequency += data["frequency"]

            if total_frequency > 0:
                for transition in transitions:
                    if len(transition) >= 3:  # Check if we have data
                        _, dest_state, data = transition
                        if isinstance(data, dict) and "frequency" in data:
                            action_val = data.get("action")
                            if action_val is not None:
                                edge_data = (
                                    self.policy_representation.get_transition_data(
                                        state, dest_state, action_val
                                    )
                                )
                                if edge_data:
                                    edge_data["probability"] = (
                                        edge_data["frequency"] / total_frequency
                                    )

    def fit(
        self,
        n_episodes: int = 10,
        max_steps: Optional[int] = None,
        update: bool = False,
    ) -> "PolicyApproximatorFromBasicObservation":
        assert (
            n_episodes > 0
        ), "The number of episodes must be a positive integer number!"

        if not update:
            self.policy_representation.clear()
            self._trajectories_of_last_fit = []
            self._is_fit = False

        progress_bar = tqdm.tqdm(range(n_episodes))
        progress_bar.set_description("Fitting policy approximator...")
        for ep in progress_bar:
            trajectory_result: List[Any] = self._run_episode(
                self.agent, max_steps=max_steps, seed=ep
            )
            self._update_with_trajectory(trajectory_result)
            self._trajectories_of_last_fit.append(trajectory_result)

        self._normalize()

        self._is_fit = True

        return self

    def get_nearest_predicate(
        self,
        input_predicate: Union[StateRepresentation, Tuple[Enum, ...]],
        verbose: bool = False,
    ) -> StateRepresentation:
        """Returns the nearest predicate on the representation. If already exists, then we return the same predicate. If not,
        then tries to change the predicate to find a similar state (Maximum change: 1 value).
        If we don't find a similar state, then we return None

        :param input_predicate: Existent or non-existent predicate in the representation
        :return: Nearest predicate
        :param verbose: Prints additional information
        """
        # Predicate exists in the MDP
        cast_predicate = cast(StateRepresentation, input_predicate)
        if self.policy_representation.has_state(cast_predicate):
            if verbose:
                print("NEAREST PREDICATE of existing predicate:", input_predicate)
            return cast_predicate
        else:
            if verbose:
                print("NEAREST PREDICATE of NON existing predicate:", input_predicate)

            predicate_space = self.discretizer.get_predicate_space()
            # TODO: Implement distance function
            new_pred = random.choice(predicate_space)
            if verbose:
                print("\tNEAREST PREDICATE in representation:", new_pred)
            return cast(StateRepresentation, new_pred)

    def get_possible_actions(
        self, predicate: Union[StateRepresentation, Tuple[Enum, ...]]
    ) -> List[Tuple[Any, float]]:
        """Given a predicate, get the possible actions and it's probabilities

        3 cases:

        - Predicate not in representation but similar predicate found: Return actions of the similar predicate
        - Predicate not in representation and no similar predicate found: Return all actions same probability
        - Predicate in MDP: Return actions of the predicate in representation

        :param predicate: Existing or not existing predicate
        :return: Action probabilities of a given state
        """
        result = defaultdict(float)
        cast_predicate = cast(StateRepresentation, predicate)

        # Predicate not in representation
        if not self.policy_representation.has_state(cast_predicate):
            # Nearest predicate not found -> Random action
            if cast_predicate is None:
                result = {
                    action: 1 / len(self.discretizer.all_actions())
                    for action in self.discretizer.all_actions()
                }
                return sorted(result.items(), key=lambda x: x[1], reverse=True)

            cast_predicate = self.get_nearest_predicate(cast_predicate)
            if cast_predicate is None:
                result = {
                    a: 1 / len(self.discretizer.all_actions())
                    for a in self.discretizer.all_actions()
                }
                return list(result.items())

        # Out edges with actions
        transitions = self.policy_representation.get_outgoing_transitions(
            cast_predicate, include_data=True
        )
        possible_actions = []

        for transition in transitions:
            if len(transition) >= 3:  # Check if we have data
                _, _, data = transition
                if (
                    isinstance(data, dict)
                    and "action" in data
                    and "probability" in data
                ):
                    action = data["action"]
                    probability = data["probability"]
                    possible_actions.append((cast_predicate, action, None, probability))

        # Drop duplicated edges
        possible_actions = list(set(possible_actions))
        # Predicate has at least 1 out edge.
        if len(possible_actions) > 0:
            for _, action, _, weight in possible_actions:
                all_actions = self.discretizer.all_actions()
                if isinstance(action, int) and 0 <= action < len(all_actions):
                    result[all_actions[action]] += weight
            return sorted(result.items(), key=lambda x: x[1], reverse=True)
        # Predicate does not have out edges. Then return all the actions with same probability
        else:
            result = {
                a: 1 / len(self.discretizer.all_actions())
                for a in self.discretizer.all_actions()
            }
            return list(result.items())

    def question1(
        self,
        predicate: Union[StateRepresentation, Tuple[Enum, ...]],
        verbose: bool = False,
    ) -> List[Tuple[Any, float]]:
        """
        Answers the question: What actions would you take in state X?

        :param predicate: The state to query
        :param verbose: Whether to print verbose output
        :return: List of (action, probability) tuples
        """
        possible_actions = self.get_possible_actions(predicate)
        if verbose:
            print("I will take one of these actions:")
            for action, prob in possible_actions:
                if hasattr(action, "name"):
                    print("\t->", action.name, "\tProb:", round(prob * 100, 2), "%")
                else:
                    print("\t->", action, "\tProb:", round(prob * 100, 2), "%")
        return possible_actions

    def get_when_perform_action(
        self, action: Action
    ) -> Tuple[List[StateRepresentation], List[StateRepresentation]]:
        """When do you perform action

        :param action: Valid action
        :return: A tuple of (all_states_with_action, states_where_action_is_best)
        """
        # Nodes where 'action' it's a possible action
        # All the nodes that has the same action (It has repeated nodes)
        all_transitions = self.policy_representation.get_all_transitions(
            include_data=True
        )
        all_nodes = []

        for transition in all_transitions:
            if len(transition) >= 3:  # Check if we have data
                u, _, data = transition
                if (
                    isinstance(data, dict)
                    and "action" in data
                    and data["action"] == action
                ):
                    all_nodes.append(u)

        # Drop all the repeated nodes
        all_nodes = list(set(all_nodes))

        # Nodes where 'action' it's the most probable action
        all_edges = []
        for u in all_nodes:
            out_edges = self.policy_representation.get_outgoing_transitions(
                u, include_data=True
            )
            all_edges.append(out_edges)

        all_best_actions = []
        for edges in all_edges:
            best_actions = []
            for edge in edges:
                if len(edge) >= 3:  # Check if we have data
                    u, _, data = edge
                    if (
                        isinstance(data, dict)
                        and "action" in data
                        and "probability" in data
                    ):
                        best_actions.append((u, data["action"], data["probability"]))

            if best_actions:
                best_actions.sort(key=lambda x: x[2], reverse=True)
                all_best_actions.append(best_actions[0])

        best_nodes = [u for u, a, w in all_best_actions if a == action]

        all_nodes.sort()
        best_nodes.sort()
        return all_nodes, best_nodes

    def question2(
        self, action: Action, verbose: bool = False
    ) -> List[StateRepresentation]:
        """
        Answers the question: When do you perform action X?

        :param action: The action to query
        :param verbose: Whether to print verbose output
        :return: List of states where the action is the most probable
        """
        if verbose:
            print("*********************************")
            print("* When do you perform action X?")
            print("*********************************")

        all_nodes, best_nodes = self.get_when_perform_action(action)
        if verbose:
            print(f"Most probable in {len(best_nodes)} states:")
        for i in range(len(all_nodes)):
            if i < len(best_nodes) and verbose:
                print(f"\t-> {best_nodes[i]}")
        # TODO: Extract common factors of resulting states
        return best_nodes

    def substract_predicates(
        self,
        origin: Union[str, List[str], StateRepresentation, Tuple[Enum, ...]],
        destination: Union[str, List[str], StateRepresentation, Tuple[Enum, ...]],
    ) -> Dict[str, Tuple[str, str]]:
        """
        Subtracts 2 predicates, getting only the values that are different

        :param origin: Origin predicate
        :param destination: Destination predicate
        :return: Dict with the different values
        """
        origin_list: List[str] = []
        destination_list: List[str] = []

        if isinstance(origin, str):
            origin_list = origin.split("-")
        elif isinstance(origin, list):
            origin_list = origin
        elif isinstance(origin, tuple):
            # Handle Tuple case
            origin_list = [str(item) for item in origin]
        else:
            # Handle StateRepresentation case - convert to string first
            origin_list = [str(origin)]

        if isinstance(destination, str):
            destination_list = destination.split("-")
        elif isinstance(destination, list):
            destination_list = destination
        elif isinstance(destination, tuple):
            # Handle Tuple case
            destination_list = [str(item) for item in destination]
        else:
            # Handle StateRepresentation case - convert to string first
            destination_list = [str(destination)]

        result = {}
        for value1, value2 in zip(origin_list, destination_list):
            if value1 != value2:
                result[value1] = (value1, value2)
        return result

    def nearby_predicates(
        self,
        state: Union[StateRepresentation, Tuple[Enum, ...]],
        greedy: bool = False,
        verbose: bool = False,
    ) -> List[Tuple[Action, StateRepresentation, Dict[str, Tuple[str, str]]]]:
        """
        Gets nearby states from state

        :param state: State to analyze
        :param greedy: Whether to use greedy action selection
        :param verbose: Whether to print verbose output
        :return: List of [Action, destination_state, difference]
        """
        cast_state = cast(StateRepresentation, state)
        out_edges = self.policy_representation.get_outgoing_transitions(
            cast_state, include_data=True
        )
        outs = []

        for edge in out_edges:
            if len(edge) >= 3:  # Check if we have data
                u, v, d = edge
                if isinstance(d, dict) and "action" in d and "probability" in d:
                    outs.append((u, v, d["action"], d["probability"]))

        result = []
        for u, v, a, w in outs:
            most_probable = self.get_most_probable_option(
                v, greedy=greedy, verbose=verbose
            )
            if most_probable:
                result.append((most_probable, v, self.substract_predicates(u, v)))

        result = sorted(result, key=lambda x: x[1])
        return result

    def get_most_probable_option(
        self,
        predicate: Union[StateRepresentation, Tuple[Enum, ...]],
        greedy: bool = False,
        verbose: bool = False,
    ) -> Optional[Action]:
        """
        Get most probable action for a predicate

        :param predicate: The state to query
        :param greedy: Whether to use greedy action selection
        :param verbose: Whether to print verbose output
        :return: The most probable action or None
        """
        cast_predicate = cast(StateRepresentation, predicate)
        if greedy:
            nearest_predicate = self.get_nearest_predicate(
                cast_predicate, verbose=verbose
            )
            possible_actions = self.get_possible_actions(nearest_predicate)

            # Possible actions always will have 1 element since for each state we only save the best action
            if possible_actions:
                return possible_actions[0][0]
            return None
        else:
            nearest_predicate = self.get_nearest_predicate(
                cast_predicate, verbose=verbose
            )
            possible_actions = self.get_possible_actions(nearest_predicate)
            if possible_actions:
                possible_actions = sorted(possible_actions, key=lambda x: x[1])
                return possible_actions[-1][0]
            return None

    def question3(
        self,
        predicate: Union[StateRepresentation, Tuple[Enum, ...]],
        action: Action,
        greedy: bool = False,
        verbose: bool = False,
    ) -> List[Dict[str, Tuple[str, str]]]:
        """
        Answers the question: Why do you perform action X in state Y?

        :param predicate: The state to query
        :param action: The action to query
        :param greedy: Whether to use greedy action selection
        :param verbose: Whether to print verbose output
        :return: List of explanations as dictionaries
        """
        if verbose:
            print("***********************************************")
            print("* Why did not you perform X action in Y state?")
            print("***********************************************")

        # Need to define appropriate policy modes for this function
        class PGBasedPolicyMode(Enum):
            GREEDY = auto()
            STOCHASTIC = auto()

        # This is a simplified version without the actual policy class
        if greedy:
            mode = PGBasedPolicyMode.GREEDY
        else:
            mode = PGBasedPolicyMode.STOCHASTIC

        # Determine best action based on mode
        if mode == PGBasedPolicyMode.GREEDY:
            possible_actions = self.get_possible_actions(predicate)
            if possible_actions:
                best_action = possible_actions[0][0]
            else:
                best_action = None
        else:
            possible_actions = self.get_possible_actions(predicate)
            if possible_actions:
                actions, probs = zip(*possible_actions)
                best_action = np.random.choice(actions, p=probs)
            else:
                best_action = None

        result = self.nearby_predicates(predicate)
        explanations = []

        if verbose:
            print("I would have chosen:", best_action)
            print(f"I would have chosen {action} under the following conditions:")
        for a, v, diff in result:
            # Only if performs the input action
            if a == action:
                if verbose:
                    print(f"Hypothetical state: {v}")
                    for predicate_key, predicate_value in diff.items():
                        print(
                            f"   Actual: {predicate_key} = {predicate_value[0]} -> Counterfactual: {predicate_key} = {predicate_value[1]}"
                        )
                explanations.append(diff)
        if len(explanations) == 0 and verbose:
            print("\tI don't know where I would have ended up")
        return explanations

    def save(self, format: str, path: Union[str, List[str]]):
        """
        Save the policy approximator

        :param format: The format to save in (e.g., 'csv')
        :param path: The path to save to
        """
        if not self._is_fit:
            raise Exception("Policy approximator cannot be saved before fitting!")

        # Implement appropriate save functionality based on the format
        if format == "csv":
            raise NotImplementedError("CSV format has not been implemented yet")
        else:
            raise NotImplementedError(f"Format {format} not supported for saving")


class InterventionalPGConstruction(PolicyApproximator):
    def fit(self): ...
