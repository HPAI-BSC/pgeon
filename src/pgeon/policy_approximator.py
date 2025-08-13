from __future__ import annotations

import abc
import csv
import pickle
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import tqdm
from gymnasium import Env

from pgeon.agent import Agent
from pgeon.discretizer import (
    Action,
    Discretizer,
    PredicateBasedState,
    State,
    StateMetadata,
    Transition,
)
from pgeon.policy_representation import (
    GraphRepresentation,
    PolicyRepresentation,
    TStateMetadata,
)


class PolicyApproximator(abc.ABC, Generic[TStateMetadata]):
    """Abstract base class for policy approximators."""

    def __init__(
        self,
        discretizer: Discretizer,
        policy_representation: PolicyRepresentation[TStateMetadata],
    ):
        self.discretizer = discretizer
        self.policy_representation = policy_representation
        self._is_fit = False
        self._trajectories_of_last_fit: List[List[Any]] = []

    def get_nearest_state(self, state: State, verbose: bool = False) -> Optional[State]:
        """Get the nearest state in the policy representation to a given state."""
        if state in self.policy_representation.states:
            return state

        input_preds = state.predicates
        all_states = list(self.policy_representation.states)
        if not all_states:
            return None

        max_similarity = -1
        nearest_state = None

        for s in all_states:
            similarity = 0
            for p1, p2 in zip(input_preds, s.predicates):
                if p1 == p2:
                    similarity += 1
            if similarity > max_similarity:
                max_similarity = similarity
                nearest_state = s

        if verbose:
            print("\tNEAREST STATE in representation:", nearest_state)
        return nearest_state

    @staticmethod
    def from_pickle(path: str):
        path_includes_pickle = path[-7:] == ".pickle"
        with open(f"{path}{'' if path_includes_pickle else '.pickle'}", "rb") as f:
            return pickle.load(f)

    def save(self, format: str, path: Union[str, List[str]]):
        if not self._is_fit:
            raise Exception("Policy Approximator cannot be saved before fitting!")

        if format not in ["pickle", "csv", "gram"]:
            raise NotImplementedError("format must be one of pickle, csv or gram")

        if format == "csv":
            assert (
                len(path) == 3
            ), "When saving in CSV format, path must be a list of 3 elements (nodes, edges, trajectories)!"
            self._save_csv(*path)
        elif format == "gram":
            assert isinstance(
                path, str
            ), "When saving in gram format, path must be a string!"
            self._save_gram(path)
        elif format == "pickle":
            assert isinstance(
                path, str
            ), "When saving in pickle format, path must be a string!"
            self._save_pickle(path)
        else:
            raise NotImplementedError

    def _save_pickle(self, path: str):
        path_includes_pickle = path[-7:] == ".pickle"
        with open(f"{path}{'' if path_includes_pickle else '.pickle'}", "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @abc.abstractmethod
    def fit(self): ...

    def _normalize(self) -> None:
        state_to_state_metadata = self.policy_representation.states.metadata
        total_frequency = sum(
            state_metadata.frequency
            for state_metadata in state_to_state_metadata.values()
        )
        if total_frequency == 0:
            return

        for state, state_metadata in state_to_state_metadata.items():
            self.policy_representation.states[state] = state_metadata.model_copy(
                update={"probability": state_metadata.frequency / total_frequency}
            )

        for state in self.policy_representation.states:
            transitions = self.policy_representation.transitions[state]
            total_transition_frequency = sum(
                transition_data.frequency for transition_data in transitions
            )

            if total_transition_frequency > 0:
                for transition_data in transitions:
                    transition = transition_data.transition
                    updated_probability = (
                        transition.frequency / total_transition_frequency
                    )
                    self.policy_representation.transitions[state][
                        transition_data.to_state
                    ] = transition.model_copy(
                        update={"probability": updated_probability}
                    )


class OnlinePolicyApproximator(PolicyApproximator, Generic[TStateMetadata]):
    @abc.abstractmethod
    def fit(self, n_episodes: int) -> None: ...


# From trajectories
class OfflinePolicyApproximator(PolicyApproximator, Generic[TStateMetadata]):
    @staticmethod
    def from_nodes_and_edges(
        path_nodes: str,
        path_edges: str,
        discretizer: Discretizer,
    ):
        policy_representation = GraphRepresentation.load_csv(
            "networkx", discretizer, Path(path_nodes), Path(path_edges)
        )
        approximator = OfflinePolicyApproximator(discretizer, policy_representation)
        approximator._is_fit = True
        return approximator

    @staticmethod
    def from_nodes_and_trajectories(
        path_nodes: str,
        path_trajectories: str,
        discretizer: Discretizer,
    ):
        approximator = OfflinePolicyApproximator(discretizer, GraphRepresentation())

        path_to_nodes_includes_csv = Path(path_nodes).suffix == ".csv"
        path_to_trajs_includes_csv = Path(path_trajectories).suffix == ".csv"

        node_info = {}
        with open(
            f"{path_nodes}{'' if path_to_nodes_includes_csv else '.csv'}", "r+"
        ) as f:
            csv_r = csv.reader(f)
            next(csv_r)

            for state_id, value, prob, freq in csv_r:
                state_prob = float(prob)
                state_freq = int(freq)

                node_info[int(state_id)] = {
                    "value": discretizer.str_to_state(value),
                    "probability": state_prob,
                    "frequency": state_freq,
                }

        with open(path_trajectories, "r") as f:
            csv_r = csv.reader(f, delimiter=" ")
            next(csv_r)  # Skip header
            for row in csv_r:
                trajectory = [int(x) for x in row]
                approximator.policy_representation.add_trajectory(trajectory)
                approximator._trajectories_of_last_fit.append(trajectory)

        approximator._is_fit = True
        approximator._normalize()
        return approximator

    def fit(self):
        # Offline approximator is already fitted by definition
        pass


class PolicyApproximatorFromBasicObservation(
    OnlinePolicyApproximator[TStateMetadata], Generic[TStateMetadata]
):
    """
    Policy approximator that approximates the policy from basic observations,
    given an agent and an environment.
    """

    def __init__(
        self,
        discretizer: Discretizer,
        policy_representation: PolicyRepresentation[TStateMetadata],
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

    def _update_with_trajectory(self, trajectory: List[Tuple[State, Action]]):
        # Only even numbers are states
        states_in_trajectory = [
            trajectory[i] for i in range(len(trajectory)) if i % 2 == 0
        ]
        all_new_states_in_trajectory = {
            PredicateBasedState(state)
            for state in set(states_in_trajectory)
            if PredicateBasedState(state) not in self.policy_representation.states
        }
        for state in all_new_states_in_trajectory:
            self.policy_representation.states[state] = StateMetadata()

        for state in states_in_trajectory:
            state_representation = PredicateBasedState(state)
            state_metadata = self.policy_representation.states[
                state_representation
            ].metadata

            updated_metadata = state_metadata.model_copy(
                update={
                    "frequency": state_metadata.frequency + 1,
                    "probability": state_metadata.probability,
                },
            )
            self.policy_representation.states[state_representation] = updated_metadata

        pointer = 0
        while (pointer + 1) < len(trajectory):
            state_from, action, state_to = trajectory[pointer : pointer + 3]
            state_from = PredicateBasedState(state_from)
            state_to = PredicateBasedState(state_to)
            if (
                state_from,
                state_to,
                action,
            ) not in self.policy_representation.transitions:
                self.policy_representation.transitions[state_from][state_to] = (
                    Transition(action=action, frequency=1)
                )
            else:
                edge_data = self.policy_representation.transitions[state_from][state_to]
                if edge_data:
                    updated_transition = edge_data.model_copy(
                        update={"frequency": edge_data.frequency + 1}
                    )
                    self.policy_representation.graph._nx_graph.remove_edge(
                        state_from, state_to, key=action
                    )
                    self.policy_representation.transitions[state_from][
                        state_to
                    ] = updated_transition
            pointer += 2

    def fit(
        self,
        n_episodes: int = 10,
        max_steps: Optional[int] = None,
        update: bool = False,
    ) -> PolicyApproximatorFromBasicObservation:
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

    def question1(
        self,
        state: State,
        verbose: bool = False,
    ) -> list[tuple[Any, float]]:
        """
        Answers the question: What actions would you take in state X?
        :param state: The state to query
        :param verbose: Whether to print verbose output
        :return: List of (action, probability) tuples
        """
        nearest_state = self.get_nearest_state(state, verbose=verbose)
        if nearest_state is None:
            return []

        possible_transitions = self.policy_representation.transitions[nearest_state]

        if verbose:
            print("I will take one of these actions:")
            for transition in possible_transitions:
                if hasattr(transition.action, "name"):
                    print(
                        "\t->",
                        transition.action.name,
                        "\tProb:",
                        round(transition.probability * 100, 2),
                        "%",
                    )
                else:
                    print(
                        "\t->",
                        transition.action,
                        "\tProb:",
                        round(transition.probability * 100, 2),
                        "%",
                    )
        return [
            (t.action, t.probability)
            for t in sorted(
                possible_transitions, key=lambda item: item.probability, reverse=True
            )
        ]

    def get_when_perform_action(
        self, action: Action
    ) -> Tuple[List[State], List[State]]:
        """When do you perform action
        :param action: Valid action
        :return: A tuple of (all_states_with_action, states_where_action_is_best)
        """
        # Nodes where 'action' it's a possible action
        # All the nodes that has the same action (It has repeated nodes)
        all_transitions = list(self.policy_representation.transitions)
        all_nodes = []

        for transition_data in all_transitions:
            if transition_data.action == action:
                all_nodes.append(transition_data.from_state)

        # Drop all the repeated nodes
        all_nodes = list(set(all_nodes))

        # Nodes where 'action' it's the most probable action
        all_edges = []
        for u in all_nodes:
            out_edges = list(self.policy_representation.transitions[u])
            all_edges.append(out_edges)

        all_best_actions = []
        for edges in all_edges:
            best_actions = []
            for transition_data in edges:
                best_actions.append(
                    (
                        transition_data.from_state,
                        transition_data.action,
                        transition_data.probability,
                    )
                )

            if best_actions:
                best_actions.sort(key=lambda x: x[2], reverse=True)
                all_best_actions.append(best_actions[0])

        best_nodes = [u for u, a, w in all_best_actions if a == action]

        all_nodes.sort()
        best_nodes.sort()
        return all_nodes, best_nodes

    def question2(self, action: Action, verbose: bool = False) -> List[State]:
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
        origin: Union[str, List[str], State, Tuple[Enum, ...]],
        destination: Union[str, List[str], State, Tuple[Enum, ...]],
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

    def nearby_states(
        self,
        state: State,
        greedy: bool = False,
        verbose: bool = False,
    ) -> List[Tuple[Action, State, Dict[str, Tuple[str, str]]]]:
        """
        Gets nearby states from state
        :param state: State to analyze
        :param greedy: Whether to use greedy action selection
        :param verbose: Whether to print verbose output
        :return: List of [Action, destination_state, difference]
        """
        transition_data_list = list(self.policy_representation.transitions[state])

        result = []
        for transition_data in transition_data_list:
            most_probable = self.get_most_probable_option(
                transition_data.to_state, greedy=greedy, verbose=verbose
            )
            if most_probable:
                result.append(
                    (
                        most_probable,
                        transition_data.to_state,
                        self.substract_predicates(state, transition_data.to_state),
                    )
                )

        result = sorted(result, key=lambda x: x[1])
        return result

    def get_most_probable_option(
        self,
        state: State,
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
        if greedy:
            nearest_state = self.get_nearest_state(state, verbose=verbose)
            possible_transitions = self.policy_representation.transitions[nearest_state]

            # Possible actions always will have 1 element since for each state we only save the best action
            if possible_transitions:
                return possible_transitions[0].action
            return None
        else:
            nearest_state = self.get_nearest_state(state, verbose=verbose)
            possible_transitions = self.policy_representation.transitions[nearest_state]
            if possible_transitions:
                return next(iter(possible_transitions)).action
            return None

    def question3(
        self,
        state: State,
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
            possible_transitions = self.policy_representation.transitions[state]
            if possible_transitions:
                best_action = possible_transitions[0].action
            else:
                best_action = None
        else:
            possible_transitions = self.policy_representation.transitions[state]
            if possible_transitions:
                actions = [t.action for t in possible_transitions]
                probs = [t.probability for t in possible_transitions]
                best_action = np.random.choice(actions, p=probs)
            else:
                best_action = None

        result = self.nearby_states(state)
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


class InterventionalPGConstruction(PolicyApproximator):
    def fit(self): ...
