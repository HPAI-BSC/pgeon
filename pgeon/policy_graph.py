from collections import defaultdict
from enum import Enum, auto
from typing import Optional, DefaultDict, Tuple, Any, List, Union, Set
import csv
import pickle

import gymnasium as gym
import networkx as nx
import numpy as np
import tqdm

from pgeon.agent import Agent
from pgeon.discretizer import Discretizer


class PolicyGraph(nx.MultiDiGraph):

    ######################
    # CREATION/LOADING
    ######################

    def __init__(self,
                 environment: gym.Env,
                 discretizer: Discretizer
                 ):
        super().__init__()
        self.environment = environment
        self.discretizer = discretizer

        self._is_fit = False
        self._trajectories_of_last_fit: List[List[Any]] = []

    @staticmethod
    def from_pickle(path: str):
        path_includes_pickle = path[-7:] == '.pickle'
        with open(f'{path}{"" if path_includes_pickle else ".pickle"}', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def from_nodes_and_edges(path_nodes: str,
                             path_edges: str,
                             environment: gym.Env,
                             discretizer: Discretizer):
        pg = PolicyGraph(environment, discretizer)

        path_to_nodes_includes_csv = path_nodes[-4:] == '.csv'
        path_to_edges_includes_csv = path_edges[-4:] == '.csv'

        node_info = {}
        with open(f'{path_nodes}{"" if path_to_nodes_includes_csv else ".csv"}', 'r+') as f:
            csv_r = csv.reader(f)
            next(csv_r)

            for state_id, value, prob, freq in csv_r:
                state_prob = float(prob)
                state_freq = int(freq)

                node_info[int(state_id)] = {
                    'value': pg.discretizer.str_to_state(value),
                    'probability': state_prob,
                    'frequency': state_freq
                }
                pg.add_node(node_info[int(state_id)]['value'],
                            probability=state_prob,
                            frequency=state_freq)

        with open(f'{path_edges}{"" if path_to_edges_includes_csv else ".csv"}', 'r+') as f:
            csv_r = csv.reader(f)
            next(csv_r)

            for node_from, node_to, action, prob, freq in csv_r:
                node_from = int(node_from)
                node_to = int(node_to)
                # TODO Get discretizer to process the action id correctly;
                #  we cannot assume the action will always be an int
                action = int(action)
                prob = float(prob)
                freq = int(freq)

                pg.add_edge(node_info[node_from]['value'], node_info[node_to]['value'], key=action,
                            frequency=freq, probability=prob)

        return pg

    @staticmethod
    def from_nodes_and_trajectories(path_nodes: str,
                                    path_trajectories: str,
                                    environment: gym.Env,
                                    discretizer: Discretizer):
        pg = PolicyGraph(environment, discretizer)

        path_to_nodes_includes_csv = path_nodes[-4:] == '.csv'
        path_to_trajs_includes_csv = path_trajectories[-4:] == '.csv'

        node_info = {}
        with open(f'{path_nodes}{"" if path_to_nodes_includes_csv else ".csv"}', 'r+') as f:
            csv_r = csv.reader(f)
            next(csv_r)

            for state_id, value, prob, freq in csv_r:
                state_prob = float(prob)
                state_freq = int(freq)

                node_info[int(state_id)] = {
                    'value': pg.discretizer.str_to_state(value),
                    'probability': state_prob,
                    'frequency': state_freq
                }

        with open(f'{path_trajectories}{"" if path_to_trajs_includes_csv else ".csv"}', 'r+') as f:
            csv_r = csv.reader(f)

            for csv_trajectory in csv_r:
                trajectory = []
                for elem_position, element in enumerate(csv_trajectory):
                    # Process state
                    if elem_position % 2 == 0:
                        trajectory.append(node_info[int(element)]['value'])
                    # Process action
                    else:
                        trajectory.append(int(element))

                pg._update_with_trajectory(trajectory)
                pg._trajectories_of_last_fit.append(trajectory)

        return pg

    ######################
    # FITTING
    ######################

    def _run_episode(self,
                     agent: Agent,
                     max_steps: int = None,
                     seed: int = None
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

    def _update_with_trajectory(self,
                                trajectory: List[Any]
                                ):

        # Only even numbers are states
        states_in_trajectory = [trajectory[i] for i in range(len(trajectory)) if i % 2 == 0]
        all_new_states_in_trajectory = {state for state in set(states_in_trajectory) if not self.has_node(state)}
        self.add_nodes_from(all_new_states_in_trajectory, frequency=0)

        state_frequencies = {s: states_in_trajectory.count(s) for s in set(states_in_trajectory)}
        for state in state_frequencies:
            self.nodes[state]['frequency'] += state_frequencies[state]

        pointer = 0
        while (pointer + 1) < len(trajectory):
            state_from, action, state_to = trajectory[pointer:pointer + 3]
            if not self.has_edge(state_from, state_to, key=action):
                self.add_edge(state_from, state_to, key=action, frequency=0)
            self[state_from][state_to][action]['frequency'] += 1
            pointer += 2

    def _normalize(self):
        weights = nx.get_node_attributes(self, 'frequency')
        total_frequency = sum([weights[state] for state in weights])
        nx.set_node_attributes(self, {state: weights[state] / total_frequency for state in weights}, 'probability')

        for node in self.nodes:
            total_frequency = sum([self.get_edge_data(node, dest_node, action)['frequency']
                                   for dest_node in self[node]
                                   for action in self.get_edge_data(node, dest_node)])
            for dest_node in self[node]:
                for action in self.get_edge_data(node, dest_node):
                    self[node][dest_node][action]['probability'] = \
                        self[node][dest_node][action]['frequency'] / total_frequency

    def fit(self,
            agent,
            num_episodes: int = 10,
            max_steps: int = None,
            update: bool = False
            ):

        if not update:
            self.clear()
            self._trajectories_of_last_fit = []
            self._is_fit = False

        progress_bar = tqdm.tqdm(range(num_episodes))
        progress_bar.set_description('Fitting PG...')
        for ep in progress_bar:
            trajectory_result: List[Any] = self._run_episode(agent, max_steps=max_steps, seed=ep)
            self._update_with_trajectory(trajectory_result)
            self._trajectories_of_last_fit.append(trajectory_result)

        self._normalize()

        self._is_fit = True

        return self

    ######################
    # SERIALIZATION
    ######################

    def _gram(self) -> str:
        graph_string = ''

        node_info = {
            node: {'id': i, 'value': self.discretizer.state_to_str(node),
                   'probability': self.nodes[node]['probability'],
                   'frequency': self.nodes[node]['frequency']}
            for i, node in enumerate(self.nodes)
        }
        # Get all unique actions in the PG
        action_info = {
            action: {'id': i, 'value': str(action)}
            for i, action in enumerate(set(action for _, _, action in self.edges))
        }

        for _, info in node_info.items():
            graph_string += f"\nCREATE (s{info['id']}:State " + "{" + f'\n  uid: "{info["id"]}",\n  value: "{info["value"]}",\n  probability: {info["probability"]}, \n  frequency:{info["frequency"]}' + "\n});"
        for _, action in action_info.items():
            graph_string += f"\nCREATE (a{action['id']}:Action " + "{" + f'\n  uid: "{action["id"]}",\n  value:{action["value"]}' + "\n});"

        for edge in self.edges:
            n_from, n_to, action = edge
            # TODO The identifier of an edge may need to be unique. Check and rework the action part of this if needed.
            graph_string += f"\nMATCH (s{node_info[n_from]['id']}:State) WHERE {node_info[n_from]['id']}.uid = \"{node_info[n_from]['id']}\" MATCH ({node_info[n_to]['id']}:State) WHERE {node_info[n_to]['id']}.uid = \"{node_info[n_to]['id']}\" CREATE (s{node_info[n_from]['id']})-[:action " + "{" + f"aid: \"{action_info[action]['id']}\", probability:{self[n_from][n_to][action]['probability']}, frequency:{self[n_from][n_to][action]['frequency']}" + "}" + f"]->(s{node_info[n_to]['id']});"

        return graph_string

    def _save_gram(self,
                   path: str
                   ):

        path_includes_gram = path[-5:] == '.gram'
        with open(f'{path}{"" if path_includes_gram else ".gram"}', 'w+') as f:
            f.write(self._gram())

    def _save_pickle(self,
                     path: str
                     ):

        path_includes_pickle = path[-7:] == '.pickle'
        with open(f'{path}{"" if path_includes_pickle else ".pickle"}', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_csv(self,
                  path_nodes: str,
                  path_edges: str,
                  path_trajectories: str
                  ):

        path_to_nodes_includes_csv = path_nodes[-4:] == '.csv'
        path_to_edges_includes_csv = path_edges[-4:] == '.csv'
        path_to_trajs_includes_csv = path_trajectories[-4:] == '.csv'

        node_ids = {}
        with open(f'{path_nodes}{"" if path_to_nodes_includes_csv else ".csv"}', 'w+') as f:
            csv_w = csv.writer(f)
            csv_w.writerow(['id', 'value', 'p(s)', 'frequency'])
            for elem_position, node in enumerate(self.nodes):
                node_ids[node] = elem_position
                csv_w.writerow([elem_position, self.discretizer.state_to_str(node),
                                self.nodes[node]['probability'], self.nodes[node]['frequency']])

        with open(f'{path_edges}{"" if path_to_edges_includes_csv else ".csv"}', 'w+') as f:
            csv_w = csv.writer(f)
            csv_w.writerow(['from', 'to', 'action', 'p(s)', 'frequency'])
            for edge in self.edges:
                state_from, state_to, action = edge
                csv_w.writerow([node_ids[state_from], node_ids[state_to], action,
                                self[state_from][state_to][action]['probability'],
                                self[state_from][state_to][action]['frequency']])

        with open(f'{path_trajectories}{"" if path_to_trajs_includes_csv else ".csv"}', 'w+') as f:
            csv_w = csv.writer(f)

            for trajectory in self._trajectories_of_last_fit:

                csv_trajectory = []
                for elem_position, element in enumerate(trajectory):
                    # Process state
                    if elem_position % 2 == 0:
                        csv_trajectory.append(node_ids[element])
                    # Process action
                    else:
                        csv_trajectory.append(element)

                csv_w.writerow(csv_trajectory)

    # gram format doesn't save the trajectories
    def save(self,
             format: str,
             path: Union[str, List[str]])\
            :
        if not self._is_fit:
            raise Exception('Policy Graph cannot be saved before fitting!')

        if format not in ['pickle', 'csv', 'gram']:
            raise NotImplementedError('format must be one of pickle, csv or gram')

        if format == 'csv':
            assert len(path) == 3, \
                "When saving in CSV format, path must be a list of 3 elements (nodes, edges, trajectories)!"
            self._save_csv(*path)
        elif format == 'gram':
            assert isinstance(path, str), "When saving in gram format, path must be a string!"
            self._save_gram(path)
        elif format == 'pickle':
            assert isinstance(path, str), "When saving in pickle format, path must be a string!"
            self._save_pickle(path)
        else:
            raise NotImplementedError


class PGBasedPolicyMode(Enum):
    GREEDY = auto()
    STOCHASTIC = auto()


class PGBasedPolicyNodeNotFoundMode(Enum):
    RANDOM_UNIFORM = auto()
    FIND_SIMILAR_NODES = auto()


class PGBasedPolicy(Agent):
    def __init__(self,
                 policy_graph: PolicyGraph,
                 mode: PGBasedPolicyMode,
                 node_not_found_mode: PGBasedPolicyNodeNotFoundMode = PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM
                 ):
        self.pg = policy_graph
        assert mode in [PGBasedPolicyMode.GREEDY, PGBasedPolicyMode.STOCHASTIC], \
            'mode must be a member of the PGBasedPolicyMode enum!'
        self.mode = mode
        assert node_not_found_mode in [PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM,
                                       PGBasedPolicyNodeNotFoundMode.FIND_SIMILAR_NODES], \
            'node_not_found_mode must be a member of the PGBasedPolicyNodeNotFoundMode enum!'
        self.node_not_found_mode = node_not_found_mode

        self.all_possible_actions = self._get_all_possible_actions()

    def _get_all_possible_actions(self) -> Set[Any]:
        all_possible_actions = set()

        for node_from in self.pg:
            for node_to in self.pg[node_from]:
                for action in self.pg[node_from][node_to]:
                    all_possible_actions.add(action)

        return all_possible_actions

    def _get_action_probability_dist(self,
                                     predicate
                                     ) -> List[Tuple[int, float]]:
        # Precondition: self.pg.has_node(predicate) and len(self.pg[predicate]) > 0:

        action_weights = defaultdict(lambda: 0)
        for dest_node in self.pg[predicate]:
            for action in self.pg[predicate][dest_node]:
                action_weights[action] += self.pg[predicate][dest_node][action]['probability']

        action_weights = [(a, action_weights[a]) for a in action_weights]
        return action_weights

    def _is_predicate_in_pg_and_usable(self, predicate) -> bool:
        return self.pg.has_node(predicate) and len(self.pg[predicate]) > 0

    def _get_nearest_predicate(self,
                               predicate
                               ):
        nearest_state_generator = self.pg.discretizer.nearest_state(predicate)
        new_predicate = predicate
        while not self._is_predicate_in_pg_and_usable(new_predicate):
            new_predicate = next(nearest_state_generator)

        return new_predicate

    def _get_action(self,
                    action_weights: List[Tuple[int, float]]
                    ) -> int:
        if self.mode == PGBasedPolicyMode.GREEDY:
            sorted_probs: List[Tuple[int, float]] = sorted(action_weights, key=lambda x: x[1], reverse=True)
            return sorted_probs[0][0]
        elif self.mode == PGBasedPolicyMode.STOCHASTIC:
            return np.random.choice([a for a, _ in action_weights], p=[w for _, w in action_weights])
        else:
            raise NotImplementedError

    def act(self,
            state
            ) -> Any:
        predicate = self.pg.discretizer.discretize(state)

        if self.pg.has_node(predicate) and len(self.pg[predicate]) > 0:
            action_prob_dist = self._get_action_probability_dist(predicate)
        else:
            if self.node_not_found_mode == PGBasedPolicyNodeNotFoundMode.RANDOM_UNIFORM:
                action_prob_dist = [(a, 1 / len(self.all_possible_actions)) for a in self.all_possible_actions]
            elif self.node_not_found_mode == PGBasedPolicyNodeNotFoundMode.FIND_SIMILAR_NODES:
                nearest_predicate = self._get_nearest_predicate(predicate)
                action_prob_dist = self._get_action_probability_dist(nearest_predicate)
            else:
                raise NotImplementedError

        return self._get_action(action_prob_dist)
# class PolicyGraph2(nx.MultiDiGraph):
#     def __init__(self, **attr):
#         super().__init__(**attr)
#
#     def _run_episode(self,
#                      agent: Agent,
#                      environment,
#                      discretizer: Discretizer,
#                      max_steps_per_episode: Optional[int] = None,
#                      seed: Optional[int] = None) -> DefaultDict[Tuple[Any, Any, Any], int]:
#
#         transition_frequencies = defaultdict(int)
#
#         observation = environment.reset(seed)
#         done = False
#
#         step_counter = 0
#         while not done:
#             if max_steps_per_episode is not None and step_counter >= step_counter:
#                 break
#
#             action = agent.act(observation)
#
#             previous_obs = observation
#             observation, _, done, _ = environment.step(action)
#
#             transition_frequencies[
#                 (discretizer.discretize(previous_obs), action, discretizer.discretize(observation))
#             ] += 1
#
#             step_counter += 1
#
#         # Adds a transition to track the final state as visited
#         transition_frequencies[(discretizer.discretize(observation), None, None)] += 1
#
#         return transition_frequencies
#
#     def _update_with_transition_frequencies(self,
#                                             transition_frequencies: DefaultDict[Tuple[Any, Any, Any], int]):
#
#         # Create or update the nodes for all visited states
#         for state_from, action, state_to in transition_frequencies:
#             if not self.has_node(state_from):
#                 self.add_node(state_from, frequency=0)
#
#             self.nodes[state_from]['frequency'] += transition_frequencies[(state_from, action, state_to)]
#
#         # Update edges
#         for state_from, action, state_to in transition_frequencies:
#             if action is not None and state_to is not None:
#                 if not self.has_edge(state_from, state_to, key=action):
#                     self.add_edge(state_from, state_to, key=action, frequency=0)
#                 else:
#                     self[state_from][state_to][action]['frequency'] += transition_frequencies[(state_from, action, state_to)]
#
#     def _normalize(self):
#         weights = nx.get_node_attributes(self, 'frequency')
#         total_frequency = sum([weights[k] for k in weights])
#         nx.set_node_attributes(self, {k: weights[k] / total_frequency for k in weights}, 'probability')
#
#         for node in self.nodes:
#             total_frequency = sum([self.get_edge_data(node, dest_node, action)['frequency']
#                                    for dest_node in self[node]
#                                    for action in self.get_edge_data(node, dest_node)])
#             for dest_node in self[node]:
#                 for action in self.get_edge_data(node, dest_node):
#                     self[node][dest_node][action]['probability'] = \
#                         self[node][dest_node][action]['frequency'] / total_frequency
#
#     def fit(self,
#             agent: Agent,
#             environment: Environment,
#             discretizer: Discretizer,
#             num_episodes: int = 1000,
#             max_steps_per_episode: Optional[int] = None,
#             update: bool = False,
#             verbose: bool = False):
#
#         if update:
#             self.clear()
#
#         for episode_i in range(num_episodes):
#             if verbose:
#                 print(f"Episode {episode_i+1}/{num_episodes}")
#
#             transition_frequencies = self._run_episode(agent, environment, discretizer, max_steps_per_episode)
#             self._update_with_transition_frequencies(transition_frequencies)
#
#         self._normalize()
#
#         return self
#
#     def serialize(self,
#                   path_to_nodes: str,
#                   path_to_edges: str):
#         path_to_nodes_includes_csv = path_to_nodes[-4:] == '.csv'
#         path_to_edges_includes_csv = path_to_edges[-4:] == '.csv'
#
#         node_info = {
#             node: {'id': i, 'value': str(node), 'p(s)': self.nodes[node]['probability']}
#             for i, node in enumerate(self.nodes)
#         }
#         edge_info = []
#         for edge in self.edges:
#             n_from, n_to, action = edge
#             edge_info.append({
#                 'from': node_info[n_from]['id'],
#                 'to': node_info[n_to]['id'],
#                 'action': action,
#                 "p(s'a|s)": self[n_from][n_to][action]['probability']
#             })
#
#         with open(f'{path_to_nodes}{"" if path_to_nodes_includes_csv else ".csv"}', 'w+') as f:
#             writer = csv.writer(f)
#             writer.writerow(['id', 'value', 'p(s)'])
#             for node in node_info:
#                 writer.writerow([node_info[node]['id'], node_info[node]['value'], node_info[node]['p(s)']])
#         with open(f'{path_to_edges}{"" if path_to_edges_includes_csv else ".csv"}', 'w+') as f:
#             writer = csv.writer(f)
#             writer.writerow(['from', 'to', 'action', "p(s'a|s)"])
#             for edge in edge_info:
#                 writer.writerow([edge['from'], edge['to'], edge['action'], edge["p(s'a|s)"]])