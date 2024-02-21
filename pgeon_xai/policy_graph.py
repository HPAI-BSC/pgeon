from collections import defaultdict
from enum import Enum, auto
from typing import Optional, DefaultDict, Tuple, Any, List, Union, Set
import csv
import pickle
import random

import gymnasium as gym
import networkx as nx
import numpy as np
import tqdm

from pgeon_xai.agent import Agent
from pgeon_xai.discretizer import Discretizer


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
                            frequency=freq, probability=prob, action=action)

        pg._is_fit = True
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

        pg._is_fit = True
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
                self.add_edge(state_from, state_to, key=action, frequency=0, action=action)
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
    # EXPLANATIONS
    ######################
    def get_nearest_predicate(self, input_predicate: Tuple[Enum], verbose=False):
        """ Returns the nearest predicate on the PG. If already exists, then we return the same predicate. If not,
        then tries to change the predicate to find a similar state (Maximum change: 1 value).
        If we don't find a similar state, then we return None

        :param input_predicate: Existent or non-existent predicate in the PG
        :return: Nearest predicate
        :param verbose: Prints additional information
        """
        # Predicate exists in the MDP
        if self.has_node(input_predicate):
            if verbose:
                print('NEAREST PREDICATE of existing predicate:', input_predicate)
            return input_predicate
        else:
            if verbose:
                print('NEAREST PREDICATE of NON existing predicate:', input_predicate)

            predicate_space = self.discretizer.get_predicate_space()
            # TODO: Implement distance function
            new_pred = random.choice(predicate_space)
            if verbose:
                print('\tNEAREST PREDICATE in PG:', new_pred)
            return new_pred

    def get_possible_actions(self, predicate):
        """ Given a predicate, get the possible actions and it's probabilities

        3 cases:

        - Predicate not in PG but similar predicate found in PG: Return actions of the similar predicate
        - Predicate not in PG and no similar predicate found in PG: Return all actions same probability
        - Predicate in MDP: Return actions of the predicate in PG

        :param predicate: Existing or not existing predicate
        :return: Action probabilities of a given state
        """
        result = defaultdict(float)

        # Predicate not in PG
        if predicate not in self.nodes():
            # Nearest predicate not found -> Random action
            if predicate is None:
                result = {action: 1 / len(self.discretizer.all_actions()) for action in self.discretizer.all_actions()}
                return sorted(result.items(), key=lambda x: x[1], reverse=True)

            predicate = self.get_nearest_predicate(predicate)
            if predicate is None:
                result = {a: 1 / len(self.discretizer.all_actions()) for a in self.discretizer.all_actions()}
                return list(result.items())

        # Out edges with actions [(u, v, a), ...]
        possible_actions = [(u, data['action'], v, data['probability'])
                            for u, v, data in self.out_edges(predicate, data=True)]
        """
        for node in self.pg.nodes():
            possible_actions = [(u, data['action'], v, data['weight'])
                                for u, v, data in self.pg.out_edges(node, data=True)]
            s = sum([w for _,_,_,w in possible_actions])
            assert  s < 1.001 and s > 0.99, f'Error {s}'
        """
        # Drop duplicated edges
        possible_actions = list(set(possible_actions))
        # Predicate has at least 1 out edge.
        if len(possible_actions) > 0:
            for _, action, v, weight in possible_actions:
                result[self.discretizer.all_actions()[action]] += weight
            return sorted(result.items(), key=lambda x: x[1], reverse=True)
        # Predicate does not have out edges. Then return all the actions with same probability
        else:
            result = {a: 1 / len(self.discretizer.all_actions()) for a in self.discretizer.all_actions()}
            return list(result.items())

    def question1(self, predicate, verbose=False):
        possible_actions = self.get_possible_actions(predicate)
        if verbose:
            print('I will take one of these actions:')
            for action, prob in possible_actions:
                print('\t->', action.name, '\tProb:', round(prob * 100, 2), '%')
        return possible_actions

    def get_when_perform_action(self, action):
        """ When do you perform action

        :param action: Valid action
        :return: Set of states that has an out edge with the given action
        """
        # Nodes where 'action' it's a possible action
        # All the nodes that has the same action (It has repeated nodes)
        all_nodes = [u for u, v, a in self.edges(data='action') if a == action]
        # Drop all the repeated nodes
        all_nodes = list(set(all_nodes))

        # Nodes where 'action' it's the most probable action
        all_edges = [list(self.out_edges(u, data=True)) for u in all_nodes]

        all_best_actions = [
            sorted([(u, data['action'], data['probability']) for u, v, data in edges], key=lambda x: x[2], reverse=True)[0]
            for edges in all_edges]
        best_nodes = [u for u, a, w in all_best_actions if a == action]

        all_nodes.sort()
        best_nodes.sort()
        return all_nodes, best_nodes

    def question2(self, action, verbose=False):
        """
        Answers the question: When do you perform action X?
        """
        if verbose:
            print('*********************************')
            print('* When do you perform action X?')
            print('*********************************')

        all_nodes, best_nodes = self.get_when_perform_action(action)
        if verbose:
            print(f"Most probable in {len(best_nodes)} states:")
        for i in range(len(all_nodes)):
            if i < len(best_nodes) and verbose:
                print(f"\t-> {best_nodes[i]}")
        # TODO: Extract common factors of resulting states
        return best_nodes

    def get_most_probable_option(self, predicate, greedy=False, verbose=False):
        if greedy:
            nearest_predicate = self.get_nearest_predicate(predicate, verbose=verbose)
            possible_actions = self.get_possible_actions(nearest_predicate)

            # Possible actions always will have 1 element since  for each state we only save the best action
            return possible_actions[0][0]
        else:
            nearest_predicate = self.get_nearest_predicate(predicate, verbose=verbose)
            possible_actions = self.get_possible_actions(nearest_predicate)
            possible_actions = sorted(possible_actions, key=lambda x: x[1])
            return possible_actions[-1][0]

    def substract_predicates(self, origin, destination):
        """
        Subtracts 2 predicates, getting only the values that are different

        :param origin: Origin predicate
        :type origin: Union[str, list]
        :param destination: Destination predicate
        :return dict: Dict with the different values
        """
        if type(origin) is str:
            origin = origin.split('-')
        if type(destination) is str:
            destination = destination.split('-')

        result = {}
        for value1, value2 in zip(origin, destination):
            if value1 != value2:
                result[value1.predicate] = (value1, value2)
        return result

    def nearby_predicates(self, state, greedy=False, verbose=False):
        """
        Gets nearby states from state

        :param verbose:
        :param greedy:
        :param state: State
        :return: List of [Action, destination_state, difference]
        """
        outs = self.out_edges(state, data=True)
        outs = [(u, d['action'], v, d['probability']) for u, v, d in outs]

        result = [(self.get_most_probable_option(v, greedy=greedy, verbose=verbose),
                   v,
                   self.substract_predicates(u, v)
                   ) for u, a, v, w in outs]

        result = sorted(result, key=lambda x: x[1])
        return result

    def question3(self, predicate, action, greedy=False, verbose=False):
        """
        Answers the question: Why do you perform action X in state Y?
        """
        if verbose:
            print('***********************************************')
            print('* Why did not you perform X action in Y state?')
            print('***********************************************')

        if greedy:
            mode = PGBasedPolicyMode.GREEDY
        else:
            mode = PGBasedPolicyMode.STOCHASTIC
        pg_policy = PGBasedPolicy(self, mode)
        best_action = pg_policy.act_upon_discretized_state(predicate)
        result = self.nearby_predicates(predicate)
        explanations = []

        if verbose:
            print('I would have chosen:', best_action)
            print(f"I would have chosen {action} under the following conditions:")
        for a, v, diff in result:
            # Only if performs the input action
            if a == action:
                if verbose:
                    print(f"Hypothetical state: {v}")
                    for predicate_key,  predicate_value in diff.items():
                        print(f"   Actual: {predicate_key} = {predicate_value[0]} -> Counterfactual: {predicate_key} = {predicate_value[1]}")
                explanations.append(diff)
        if len(explanations) == 0 and verbose:
            print("\tI don't know where I would have ended up")
        return explanations

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
            graph_string += f"\nCREATE (s{info['id']}:State " + "{" + f'\n  uid: "s{info["id"]}",\n  value: "{info["value"]}",\n  probability: {info["probability"]}, \n  frequency:{info["frequency"]}' + "\n});"
        for _, action in action_info.items():
            graph_string += f"\nCREATE (a{action['id']}:Action " + "{" + f'\n  uid: "a{action["id"]}",\n  value:{action["value"]}' + "\n});"

        for edge in self.edges:
            n_from, n_to, action = edge
            # TODO The identifier of an edge may need to be unique. Check and rework the action part of this if needed.
            graph_string += f"\nMATCH (s{node_info[n_from]['id']}:State) WHERE s{node_info[n_from]['id']}.uid = \"s{node_info[n_from]['id']}\" MATCH (s{node_info[n_to]['id']}:State) WHERE s{node_info[n_to]['id']}.uid = \"s{node_info[n_to]['id']}\" CREATE (s{node_info[n_from]['id']})-[:a{action_info[action]['id']} " + "{" + f"probability:{self[n_from][n_to][action]['probability']}, frequency:{self[n_from][n_to][action]['frequency']}" + "}" + f"]->(s{node_info[n_to]['id']});"

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
             path: Union[str, List[str]]) \
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

    def act_upon_discretized_state(self, predicate):
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

    def act(self,
            state
            ) -> Any:
        predicate = self.pg.discretizer.discretize(state)
        return self.act_upon_discretized_state(predicate)

