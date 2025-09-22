from collections import deque

import pgeon

from .predicates import (
    Action2Nearest,
    Agent,
    Direction,
    Held,
    Object,
    Orientations,
    PotState,
)


class Discretizer(pgeon.Discretizer):

    def __init__(
        self,
        pov: Agent,
    ):
        """
        :param env: Environment
        """
        self.pov_agent_id: int = 0 if pov == Agent.PLAYER else 1

    def all_actions(self):
        return [
            Action2Nearest.TOP,
            Action2Nearest.BOTTOM,
            Action2Nearest.LEFT,
            Action2Nearest.RIGHT,
            Action2Nearest.STAY,
            Action2Nearest.INTERACT,
        ]

    def discretize(self, state):
        pass

    def state_to_str(self, state) -> str:
        pass

    def str_to_state(self, state: str):
        pass

    def nearest_state(self, state):
        pass

    def _get_possible_orientations(self, position):
        """
        Returns which is the correct orientation to interact well with position.

        :param position: Position with which we want to interact.
        """
        possible_orientations = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for ori in possible_orientations:
            next_pos = (position[0] + ori[0], position[1] + ori[1])
            if next_pos in self.valid_pos:
                return (-ori[0], -ori[1])

        raise Exception("Position {} is not accessible to interact".format(position))

    def _get_basic_sources(self):
        """
        Gets the basic sources

        :returns: Dictionary with all the basic sources of the layout
        """

        sources = {
            "onion": self.gridWorld.get_onion_dispenser_locations(),
            "tomato": self.gridWorld.get_tomato_dispenser_locations(),
            "dish": self.gridWorld.get_dish_dispenser_locations(),
            "service": self.gridWorld.get_serving_locations(),
            "pot": self.gridWorld.get_pot_locations(),
        }

        # Bar zones, where the players can leave things on
        sources = {obj: list_pos for obj, list_pos in sources.items()}
        for obj, list_pos in sources.items():
            sources[obj] = [
                (pos, self._get_possible_orientations(pos)) for pos in list_pos
            ]

        return sources

    def _get_temporary_sources(self, obs):
        if obs is not None:
            unowned_objects = obs.unowned_objects_by_type
            # Temporary sources (coordination)
            temporary_sources = {
                obj: [
                    (obj_i.position, self._get_possible_orientations(obj_i.position))
                    for obj_i in list_obj
                ]
                for obj, list_obj in unowned_objects.items()
            }
            # Union between self.sources and temporary_sources
            for obj, l in self.sources.items():
                if obj in temporary_sources:
                    temporary_sources[obj] += l
                else:
                    temporary_sources[obj] = l
            return temporary_sources
        else:
            return None

    @staticmethod
    def _bfs(grid, start, goal, clear):
        """
        Shortest Path using BFS algorithm.

        :returns: Shortest path between position start to goal
        :param grid: Matrix layout
        :param start: Start position
        :param goal: Goal position
        :param clear: Types of cell where the agent can go through.
        """

        queue = deque([[start]])
        seen = {start}
        width = len(grid[0])
        height = len(grid)
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if goal[0] == x and goal[1] == y:
                return path
            for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                # Valid position to visit later
                if (
                    0 <= x2 < width
                    and 0 <= y2 < height
                    and grid[y2][x2] in clear
                    and (x2, y2) not in seen
                ):
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))

    @staticmethod
    def _get_move(start_pos, end_pos):
        if start_pos[0] < end_pos[0]:
            return Action2Nearest.RIGHT
        if start_pos[0] > end_pos[0]:
            return Action2Nearest.LEFT
        if start_pos[1] < end_pos[1]:
            return Action2Nearest.BOTTOM
        if start_pos[1] > end_pos[1]:
            return Action2Nearest.TOP
        else:
            return Action2Nearest.STAY

    @staticmethod
    def _get_moves(positions, end_ori: Orientations, agent_ori: Orientations):
        """
        Converts a list of positions in a list of actions.

        :returns: List of actions.
        :param positions: List of positions
        :param ori: Final orientation
        :param agent_ori: Initial agent orientation
        """

        if positions == None:
            return None
        actions = []
        # Convert each pair of position to an action
        for i in range(len(positions) - 1):
            actions.append(Discretizer._get_move(positions[i], positions[i + 1]))

        # If needed, add one step to face well the source
        # print('actions', actions, 'ori_source', ori.name, 'ori_agent', agent_ori)
        if (
            len(actions) >= 1
            and actions[-1].name != Orientations(end_ori).name
            or len(actions) == 0
            and end_ori != agent_ori
        ):
            actions.append(Action2Nearest[end_ori.name.upper()])

        # At the end, add the interaction action
        actions.append(Action2Nearest.INTERACT)
        return actions

    def _get_held_predicate(self, obs, player_id):
        """
        Gets the predicate 'Held'

        :returns: Dictionary with the predicate
        :param obs: Current Observation
        """

        if obs is None:
            return {"held": Held.NOTHING}
        player = obs.players[player_id]
        if player.has_object():
            held = player.get_object().name
            if held == "onion":
                return Held.ONION
            elif held == "tomato":
                return Held.TOMATO
            elif held == "dish":
                return Held.DISH
            elif held == "soup":
                return Held.SOUP
        else:
            return Held.NOTHING

    def _get_pot_state_predicate(self, obs, pot: Object):
        """
        Gets the predicate 'pot_state'

        :returns: Dictionary with the predicate
        :param obs: Current Observation
        """
        if obs is None:
            return PotState.NOT_STARTED

        pos, ori = self.sources["pot"][0 if pot == Object.POT0 else 1]

        unowned_objects = obs.unowned_objects_by_type
        if "soup" in unowned_objects:
            soups = unowned_objects["soup"]
            for soup in soups:
                if soup.position == pos:
                    if soup.is_idle:
                        return PotState.PREPARING
                    if not soup.is_ready:
                        return PotState.COOKING
                    if soup.is_ready:
                        return PotState.FINISHED
        return PotState.NOT_STARTED

        # if obs is None:
        #     return {f'pot {pos} state': PotState.NOT_STARTED for pos, ori in self.sources['pot']}
        # unowned_objects = obs.unowned_objects_by_type
        # oven = {f'pot {pos} state': PotState.NOT_STARTED for pos, ori in self.sources['pot']}
        # oven_positions = [p for p, o in self.sources['pot']]
        # if 'soup' in unowned_objects:
        #     soups = unowned_objects['soup']
        #     for soup in soups:
        #         if soup.position in oven_positions:
        #             num_onions = soup.state[1]
        #             pot_time = soup.state[2]
        #             # Fi
        #             if num_onions == 3 and pot_time == 20:
        #                 oven[f'pot {soup.position} state'] = PotState.FINISHED
        #             # Co
        #             elif num_onions == 3:
        #                 oven[f'pot {soup.position} state'] = PotState.COOKING
        #             # Wa
        #             else:
        #                 oven[f'pot {soup.position} state'] = PotState.PREPARING
        #
        # return oven

    def _get_action_to_nearest_object(self, pos, ori: Orientations, obj, sources):
        """
        Gets the next action to perform in order to be closer to the object.

        :returns: Next action
        :param pos: Actual agent position
        :param ori: Actual agent orientation
        :param obj: Object that the agent want to achieve
        :param sources: Current layout sources.
        """

        # Assert that exist some source for our object
        assert (
            obj in sources
        ), f"Error: Can't find object '{obj}' in sources. Existent objects are {list(sources.keys())}"
        clear = [" "]
        # Get all possible source positions
        obj_pos_list = [
            ((p[0] - orient[0], p[1] - orient[1]), Orientations(orient))
            for p, orient in sources[obj]
        ]
        # Get path to each source position
        paths = [
            self._get_moves(
                self._bfs(self.gridWorld.terrain_mtx, pos, goal=obj_pos, clear=clear),
                orient,
                ori,
            )
            for obj_pos, orient in obj_pos_list
        ]
        # Remove all None paths
        paths = [path for path in paths if path != None]
        # If the agent can't take nothing then stay
        if len(paths) == 0:
            # print(obj, 'Shortest Path: Not existing path')
            return Action2Nearest.STAY
        # Take the shortest path and return the first action
        lengths = list(map(len, paths))
        min_index = lengths.index(min(lengths))
        shortest_path = paths[min_index]
        # print(obj, '--', pos, ori.name, 'Shortest Path:', shortest_path)
        return shortest_path[0]

    def _get_action_to_nearest_pot(self, pos, ori: Orientations, obj, sources, id):
        """
        Gets the next action to perform in order to be closer to the nearest pot.

        :returns: Next action
        :param pos: Actual agent position
        :param ori: Actual agent orientation
        :param obj: Object that the agent want to achieve
        :param sources: Current layout sources
        :param id: Pot id
        """

        # Assert that exist some source for our object
        assert (
            obj in sources
        ), f"Error: Can't find object '{obj}' in sources. Existent objects are {list(sources.keys())}"
        clear = [" "]
        pot = sources[obj][id]
        # Get all possible source positions
        obj_pos_list = [
            ((pot[0][0] - pot[1][0], pot[0][1] - pot[1][1]), Orientations(pot[1]))
        ]
        # Get path to each source position
        paths = [
            self._get_moves(
                self._bfs(self.gridWorld.terrain_mtx, pos, goal=obj_pos, clear=clear),
                orient,
                ori,
            )
            for obj_pos, orient in obj_pos_list
        ]
        # Remove all None paths
        paths = [path for path in paths if path != None]
        # If the agent can't take nothing then stay
        if len(paths) == 0:
            # print(obj, 'Shortest Path: Not existing path')
            return Action2Nearest.STAY
        # Take the shortest path and return the first action
        lengths = list(map(len, paths))
        min_index = lengths.index(min(lengths))
        shortest_path = paths[min_index]
        # print(obj, '--', pos, ori.name, 'Shortest Path:', shortest_path)
        return shortest_path[0]

    def _get_action_to_nearest_soup(self, pos, ori: Orientations, obj, sources):
        """
        Gets the next action to perform in order to be closer to the nearest soup.

        :returns: Next action
        :param pos: Actual agent position
        :param ori: Actual agent orientation
        :param obj: Object that the agent want to achieve
        :param sources: Current layout sources
        """

        # Assert that exist some source for our object
        if obj not in sources:
            return Action2Nearest.STAY
        clear = [" "]
        # Get all possible source positions
        pot_pos_list = [p for p, o in self.sources["pot"]]
        obj_pos_list = [
            ((p[0] - orient[0], p[1] - orient[1]), Orientations(orient))
            for p, orient in sources[obj]
            if p not in pot_pos_list
        ]

        # Get path to each source position
        paths = [
            self._get_moves(
                self._bfs(self.gridWorld.terrain_mtx, pos, goal=obj_pos, clear=clear),
                orient,
                ori,
            )
            for obj_pos, orient in obj_pos_list
        ]
        # Remove all None paths
        paths = [path for path in paths if path != None]
        # If the agent can't take nothing then stay
        if len(paths) == 0:
            # print(obj, 'Shortest Path: Not existing path')
            return Action2Nearest.STAY
        # Take the shortest path and return the first action
        lengths = list(map(len, paths))
        min_index = lengths.index(min(lengths))
        shortest_path = paths[min_index]
        # print(obj, '--', pos, ori.name, 'Shortest Path:', shortest_path)
        return shortest_path[0]

    def _get_object_pos_predicate(self, obs, temporary_sources, player_id, item: str):
        """Computes next action to get the nearest of the object as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        if obs is None:
            return Action2Nearest.STAY
        else:
            player = obs.players[player_id]
            return self._get_action_to_nearest_object(
                player.position,
                Orientations(player.orientation),
                item,
                temporary_sources,
            )

    def _get_pot_pos_predicate(self, obs, temporary_sources, pot: Object, player_id):
        """Computes next action to get the nearest pot as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """

        if obs is None:
            return Action2Nearest.STAY

        pos, _ = self.sources["pot"][0 if pot == Object.POT0 else 1]

        player = obs.players[player_id]
        return self._get_action_to_nearest_pot(
            player.position,
            Orientations(player.orientation),
            "pot",
            temporary_sources,
            0 if pot == Object.POT0 else 1,
        )
        # pos_predicate = Action2Nearest.STAY
        #
        #
        #
        # for i in range(len(self.sources['pot'])):
        #     if obs is None:
        #         pos_predicate[f'pot {self.sources["pot"][i][0]} pos'] = Action.STAY
        #     else:
        #         player: PlayerState = obs.players[player_id]
        #         pos_predicate[f'pot {self.sources["pot"][i][0]} pos'] = \
        #             self._get_action_to_nearest_pot(player.position,
        #                                             Orientations(player.orientation),
        #                                             'pot',
        #                                             temporary_sources,
        #                                             i)
        # return pos_predicate

    def _get_soup_pos_predicate(self, obs, temporary_sources, player_id):
        """Computes next action to get the nearest soup as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        if obs is None:
            return Action2Nearest.STAY
        else:
            player = obs.players[player_id]
            return self._get_action_to_nearest_soup(
                player.position,
                Orientations(player.orientation),
                "soup",
                temporary_sources,
            )

    def _get_partner_zone(self, obs, player_id, partner_id):
        """Computes in which zone is the partner located:

        :return: A CardinalDirection
        """
        player_pos = obs.players[player_id].position
        partner_pos = obs.players[partner_id].position

        if player_pos[0] == partner_pos[0] and player_pos[1] > partner_pos[1]:
            return Direction.NORTH
        elif player_pos[0] == partner_pos[0] and player_pos[1] < partner_pos[1]:
            return Direction.SOUTH
        elif player_pos[0] > partner_pos[0] and player_pos[1] == partner_pos[1]:
            return Direction.WEST
        elif player_pos[0] < partner_pos[0] and player_pos[1] == partner_pos[1]:
            return Direction.EAST

        elif player_pos[0] > partner_pos[0] and player_pos[1] > partner_pos[1]:
            return Direction.NORTHWEST
        elif player_pos[0] < partner_pos[0] and player_pos[1] < partner_pos[1]:
            return Direction.SOUTHEAST
        elif player_pos[0] > partner_pos[0] and player_pos[1] < partner_pos[1]:
            return Direction.SOUTHWEST
        else:
            return Direction.NORTHEAST

    def _get_partner_zone_predicate(self, obs, player_id, partner_id):
        """Computes next action to get the nearest onion as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        if obs is None:
            return Direction.NORTH
        else:
            return self._get_partner_zone(obs, player_id, partner_id)
