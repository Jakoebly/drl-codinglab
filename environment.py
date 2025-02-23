# actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

# positions in grid:
# - (0,0) is upper left corner
# - first index is vertical (increasing from top to bottom)
# - second index is horizontal (increasing from left to right)

# if new item appears in a cell into which the agent moves/at which the agent stays in the same time step,
# it is not picked up (if agent wants to pick it up, it has to stay in the cell in the next time step)

import random
from typing import Tuple
import pandas as pd
from copy import deepcopy
from itertools import compress
import numpy as np
from queue import Queue
from collections import deque
import yaml
from src.renderer.renderer import Renderer
import src.settings as settings
import src.experiments as exp


class Environment(object):
    def __init__(self, variant, data_dir, render=False):
        self.variant = variant
        self.vertical_cell_count = 5
        self.horizontal_cell_count = 5
        self.vertical_idx_target = 2
        self.horizontal_idx_target = 0
        self.action_size = 5
        self.target_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.episode_steps = 200
        self.max_response_time = 15 if self.variant == 2 else 10
        self.reward = 25 if self.variant == 2 else 15
        self.data_dir = data_dir

        self.training_episodes = pd.read_csv(self.data_dir + f"/variant_{self.variant}/training_episodes.csv")
        self.training_episodes = self.training_episodes.training_episodes.tolist()
        self.validation_episodes = pd.read_csv(self.data_dir + f"/variant_{self.variant}/validation_episodes.csv")
        self.validation_episodes = self.validation_episodes.validation_episodes.tolist()
        self.test_episodes = pd.read_csv(self.data_dir + f"/variant_{self.variant}/test_episodes.csv")
        self.test_episodes = self.test_episodes.test_episodes.tolist()

        self.remaining_training_episodes = deepcopy(self.training_episodes)
        self.validation_episode_counter = 0

        if self.variant == 0 or self.variant == 2:
            self.agent_capacity = 1
        else:
            self.agent_capacity = 3

        if self.variant == 0 or self.variant == 1:
            self.eligible_cells = [(0,0), (0,1), (0,2), (0,3), (0,4),
                                   (1,0), (1,1), (1,2), (1,3), (1,4),
                                   (2,0), (2,1), (2,2), (2,3), (2,4),
                                   (3,0), (3,1), (3,2), (3,3), (3,4),
                                   (4,0), (4,1), (4,2), (4,3), (4,4)]
        else:
            self.eligible_cells = [(0,0),        (0,2), (0,3), (0,4),
                                   (1,0),        (1,2),        (1,4),
                                   (2,0),        (2,2),        (2,4),
                                   (3,0), (3,1), (3,2),        (3,4),
                                   (4,0), (4,1), (4,2),        (4,4)]

        # FOR RENDERING ONLY:
        self.render_enabled = render
        if self.render_enabled:
            self.renderer = Renderer(grid_size =(self.vertical_cell_count, self.horizontal_cell_count), eligible_cells=self.eligible_cells, max_response_time=self.max_response_time)
        
        self.current_episode_target_count = 0  # Counts number of items dropped off at target cell

        # TODO:EXPERIMENTS
        # Set experiment variables
        exp.set_exp_vars(self)

        self.obs_config = {}

    # Initializing new episode (specify if training, validation, or testing via the mode argument)
    def reset(self, mode, structure="DNN"):
        modes = ["training", "validation", "testing"]
        if mode not in modes:
            raise ValueError("Invalid mode. Expected one of: %s" % modes)

        structures = ["DNN", "CNN"]
        if structure not in structures:
            raise ValueError("Invalid NN structure. Expected one of: %s" % structures)
        self.nn_structure = structure

        self.step_count = 0
        self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.agent_load = 0  # number of items loaded (0 or 1, except for first extension, where it can be 0,1,2,3)
        self.item_locs = []
        self.item_times = []
        # self.item_distances = []

        if mode == "testing":
            episode = self.test_episodes[0]
            self.test_episodes.remove(episode)
        elif mode == "validation":
            episode = self.validation_episodes[self.validation_episode_counter]
            self.validation_episode_counter = (self.validation_episode_counter + 1) % 100
        else:
            if not self.remaining_training_episodes:
                self.remaining_training_episodes = deepcopy(self.training_episodes)
            episode = random.choice(self.remaining_training_episodes)
            self.remaining_training_episodes.remove(episode)
        self.data = pd.read_csv(self.data_dir + f"/variant_{self.variant}/episode_data/episode_{episode:03d}.csv",
                                index_col=0)

        # Reset target count for render
        self.current_episode_target_count = 0

        return self.get_new_obs()

    # take one environment step based on the action act
    def step(self, act):
        self.step_count += 1
        rew = 0
        # done signal (1 if episode ends, 0 if not)
        if self.step_count == self.episode_steps:
            done = 1
        else:
            done = 0

        # agent movement
        new_loc = self.agent_loc
        if act != 0:
            if act == 1:  # up
                new_loc = (self.agent_loc[0] - 1, self.agent_loc[1])
            elif act == 2:  # right
                new_loc = (self.agent_loc[0], self.agent_loc[1] + 1)
            elif act == 3:  # down
                new_loc = (self.agent_loc[0] + 1, self.agent_loc[1])
            elif act == 4:  # left
                new_loc = (self.agent_loc[0], self.agent_loc[1] - 1)

            if new_loc in self.eligible_cells:
                self.agent_loc = new_loc
                rew += -1

        # item pick-up
        time = 0
        if (self.agent_load < self.agent_capacity) and (self.agent_loc in self.item_locs):
            self.agent_load += 1
            idx = self.item_locs.index(self.agent_loc)
            time = self.item_times[idx]
            self.item_locs.pop(idx)
            self.item_times.pop(idx)
            rew += self.reward / 2

        # item drop-off
        if self.agent_loc == self.target_loc:
            rew += self.agent_load * self.reward / 2
            # FOR RENDER
            self.current_episode_target_count += self.agent_load
            self.agent_load = 0

        # track how long ago items appeared
        self.item_times = [i + 1 for i in self.item_times]

        # remove items for which max response time is reached
        mask = [i < self.max_response_time for i in self.item_times]
        self.item_locs = list(compress(self.item_locs, mask))
        self.item_times = list(compress(self.item_times, mask))

        # add items which appear in the current time step
        new_items = self.data[self.data.step == self.step_count]
        new_items = list(zip(new_items.vertical_idx, new_items.horizontal_idx))
        new_items = [i for i in new_items if i not in self.item_locs]  # not more than one item per cell
        num_new_items = len(new_items)

        self.item_locs += new_items
        self.item_times += [0] * len(new_items)

        # get new observation
        next_obs = self.get_new_obs()

        if self.render_enabled:
            self.update_render()

        # TODO:EXPERIMENTS
        # Update experiment variables
        exp.update_exp_vars(self, act, new_loc, new_items, num_new_items, time)

        return rew, next_obs, done

    def update_render(self):
        """Update the render."""

        self.renderer.render(
            agent_loc=self.agent_loc,
            target_loc=self.target_loc,
            item_locs=self.item_locs,
            item_times=self.item_times,
            agent_load=self.agent_load,
            target_count=self.current_episode_target_count
        )

    def get_new_obs(self):
        if self.nn_structure == "DNN":
            return self.get_dnn_obs()
        if self.nn_structure == "CNN":
            return self.get_cnn_obs()

    def get_state_size(self, structure="DNN"):
        if structure == "DNN":
            return self.reset(mode="training", structure=structure).shape[0]
        if structure == "CNN":
            # (Height, Width, Channels) i.e. (5, 5, x)
            # state = self.reset(mode = "training", structure = structure).shape
            # return state
            grid_state, scalar_state = self.reset(mode="training", structure=structure)

            return grid_state.shape, scalar_state.shape

    def manhatten_dist(self, pos: Tuple[int, int], tgt: Tuple[int, int]) -> Tuple[int, int]:
        if self.variant == 2:
            visited = set()
            q = Queue()
            q.put((pos, 0))
            while not q.empty():
                current, dist = q.get()
                if current == tgt:
                    return dist
                if current in visited or current not in self.eligible_cells:
                    continue
                visited.add(current)

                # Add neighbors
                neighbors = [
                    (current[0] - 1, current[1]), (current[0] + 1, current[1]),
                    (current[0], current[1] - 1), (current[0], current[1] + 1)
                ]
                for neighbor in neighbors:
                    if neighbor not in visited:
                        q.put((neighbor, dist + 1))

            return 0
        else:
            return abs(pos[0] - tgt[0]) + abs(pos[1] - tgt[1])

    def is_reachable(self, item_loc: Tuple[int, int], remaining_time: int) -> bool:
        """
        Check if an item is reachable before its time runs out and within the remaining episode time.

        Parameters:
            item_loc (Tuple[int, int]): The location of the item.
            remaining_time (int): Remaining time for the item.

        Returns:
            bool: True if reachable, False otherwise.
        """
        distance = self.manhatten_dist(self.agent_loc, item_loc)
        remaining_episode_time = self.episode_steps - self.step_count
        return distance <= min(remaining_time, remaining_episode_time)

    def shortest_path(self, start: Tuple[int, int]) -> Tuple[dict, dict]:
        """
        Computes shortest path distances and directions from the start position
        to all other positions in the grid, considering blocked cells.

        Returns:
            distances (dict): A dictionary with shortest path distances to each cell.
            directions (dict): A dictionary with directions to follow the shortest path.
        """
        distances = {start: 0}
        directions = {}  # Will hold the first step direction for each cell
        queue = deque([start])
        visited = set([start])

        while queue:
            current = queue.popleft()
            cur_distance = distances[current]

            # Explore neighbors
            for action, (dy, dx) in zip(["up", "down", "left", "right"], [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                neighbor = (current[0] + dy, current[1] + dx)

                if (
                        neighbor in self.eligible_cells and  # Not a blocked cell
                        neighbor not in visited and  # Not already visited
                        0 <= neighbor[0] < self.vertical_cell_count and
                        0 <= neighbor[1] < self.horizontal_cell_count
                ):
                    # Update distance only if the neighbor hasn't been visited
                    if neighbor not in distances:
                        distances[neighbor] = cur_distance + 1
                        # Store the direction for the first step from the start position
                        if current == start:
                            directions[neighbor] = action
                        else:
                            directions[neighbor] = directions[current]
                        queue.append(neighbor)
                    visited.add(neighbor)

        return distances, directions

    def get_dnn_obs(self) -> np.ndarray:
        """Returns the 2D encoded observations of the environment."""
        observation_tuple_size = 8
        pad_value = 0
        cumulative_items_time = 0

        if self.variant == 2:
            distances, directions = self.shortest_path(self.agent_loc)

        # Free capacity of agent (normalized)
        free_capacity_agent = (self.agent_capacity - self.agent_load) / self.agent_capacity

        # Agent Position
        position_agent = [self.agent_loc[0], self.agent_loc[1]]

        # Remaining Episode Time (normalized)
        remaining_episode_time = (self.episode_steps - self.step_count) / self.episode_steps

        # Manhatten Distance to Target (normalized)
        if self.variant == 2:
            distance_to_target = [distances.get(self.target_loc, 0) / (self.vertical_cell_count * 2)]
        else:
            distance_to_target = [
                self.manhatten_dist(self.agent_loc, self.target_loc) / (self.vertical_cell_count * 2)
            ]

        # Vector Distance to Target - 2D
        if self.variant == 2:
            if self.target_loc in directions:
                vector_to_target = {
                    "up": [-1, 0],
                    "down": [1, 0],
                    "left": [0, -1],
                    "right": [0, 1]
                }.get(directions[self.target_loc], [0, 0])
            else:
                vector_to_target = [0, 0]

        else:
            vector_to_target = [
                self.target_loc[0] - self.agent_loc[0],
                self.target_loc[1] - self.agent_loc[1],
            ]

        # Vector Distance to Walls
        if self.variant == 2:
            vector_to_walls = [
                min(
                    (
                        self.agent_loc[0] - row
                        for row in range(self.agent_loc[0], -1, -1)
                        if (row, self.agent_loc[1]) not in self.eligible_cells
                    ),
                    default=self.agent_loc[0],
                ),  # Up
                min(
                    (
                        row - self.agent_loc[0]
                        for row in range(self.agent_loc[0], self.vertical_cell_count)
                        if (row, self.agent_loc[1]) not in self.eligible_cells
                    ),
                    default=self.vertical_cell_count - self.agent_loc[0] - 1,
                ),  # Down
                min(
                    (
                        self.agent_loc[1] - col
                        for col in range(self.agent_loc[1], -1, -1)
                        if (self.agent_loc[0], col) not in self.eligible_cells
                    ),
                    default=self.agent_loc[1],
                ),  # Left
                min(
                    (
                        col - self.agent_loc[1]
                        for col in range(self.agent_loc[1], self.horizontal_cell_count)
                        if (self.agent_loc[0], col) not in self.eligible_cells
                    ),
                    default=self.horizontal_cell_count - self.agent_loc[1] - 1,
                ),  # Right
            ]

            # Apply overrides based on the agent's current position
            if self.agent_loc in [(0, 0), (1, 0), (2, 0), (1, 2), (2, 2), (3, 2), (4, 2)]:
                vector_to_walls[3] = 0  # "Right"
            if self.agent_loc in [(0, 2), (1, 2), (2, 2), (1, 4), (2, 4), (3, 4), (4, 4)]:
                vector_to_walls[2] = 0  # "Left"
            if self.agent_loc == (3, 1):
                vector_to_walls[0] = 0  # "Up"
            if self.agent_loc == (0, 3):
                vector_to_walls[1] = 0  # "Down"

            # New Overrides
            if self.agent_loc in [(3, 0), (4, 0)]:
                vector_to_walls[3] = 2  # "Right"
            if self.agent_loc in [(3, 1), (4, 1)]:
                vector_to_walls[3] = 1  # "Right"
            if self.agent_loc == (4, 1):
                vector_to_walls[0] = 1  # "Up"
            if self.agent_loc == (0, 4):
                vector_to_walls[2] = 2  # "Left"
            if self.agent_loc == (0, 3):
                vector_to_walls[2] = 1  # "Left"

        else:
            # Standard vector_to_walls calculation
            vector_to_walls = [
                self.agent_loc[0],
                self.vertical_cell_count - 1 - self.agent_loc[0],
                self.agent_loc[1],
                self.horizontal_cell_count - 1 - self.agent_loc[1],
            ]

        # Initialize item-based observations
        position_items = []
        remaining_time_items = []
        prioritization_ratio_items_1 = []
        prioritization_ratio_items_2 = []
        prioritization_ratio_items_3 = []
        vector_agent_to_items = []
        distances_target_to_items = []
        proximity_scores = []
        manhatten_distance_agent_to_items = []
        manhatten_distance_target_to_items = []
        filtered_item_locs = []
        filtered_item_times = []
        closest_item = [pad_value, pad_value]
        vector_closest = [pad_value, pad_value]

        # Filter reachable items and limit to observation_tuple_size
        for loc, time in zip(self.item_locs, self.item_times):
            if len(filtered_item_locs) >= observation_tuple_size:
                break
            if self.is_reachable(loc, self.max_response_time - time):
                filtered_item_locs.append(loc)
                filtered_item_times.append(time)

        # Closest item
        closest_idx = -1
        if filtered_item_locs:
            if self.variant == 2:
                item_distances = [distances.get(item, 0) for item in filtered_item_locs]
            else:
                item_distances = [self.manhatten_dist(self.agent_loc, item) for item in filtered_item_locs]
            closest_idx = np.argmin(item_distances)
            closest_item = [filtered_item_locs[closest_idx][0], filtered_item_locs[closest_idx][1]]

        # Distance to Closest Item
        if closest_idx != -1:
            if self.variant == 2:
                jaja = closest_item
                closest_item_tuple = tuple(jaja)
                if closest_item_tuple in directions:
                    vector_closest = {
                        "up": [-1, 0],
                        "down": [1, 0],
                        "left": [0, -1],
                        "right": [0, 1]
                    }.get(directions[closest_item_tuple], [0, 0])
                else:
                    vector_closest = [0, 0]
            else:
                vector_closest = [
                    self.agent_loc[0] - filtered_item_locs[closest_idx][0],
                    self.agent_loc[1] - filtered_item_locs[closest_idx][1],
                ]

        # Process filtered items for observations
        # for loc, time in zip(filtered_item_locs[:observation_tuple_size], filtered_item_times[:observation_tuple_size]):
        for loc, time in zip(filtered_item_locs, filtered_item_times):
            position_items.extend([loc[0], loc[1]])

            # Normalized item time
            remaining_time = (self.max_response_time - time) / self.max_response_time
            remaining_time_items.append(remaining_time)
            cumulative_items_time += remaining_time

            if self.variant == 2:
                d_agent = distances.get(loc, 0)
                d_target = distances.get(self.target_loc, 0)

            else:
                d_agent = self.manhatten_dist(self.agent_loc, loc)
                d_target = self.manhatten_dist(loc, self.target_loc)

            manhatten_distance_agent_to_items.append(d_agent / (self.vertical_cell_count * 2))
            manhatten_distance_target_to_items.append(d_target / (self.vertical_cell_count * 2))

            prioritization_ratio_1 = ((1 / (d_target + 1)) + (1 / (d_agent + 1))) * (1 / remaining_time)
            prioritization_ratio_2 = (self.max_response_time - remaining_time) / (d_agent + d_target)
            prioritization_ratio_3 = (self.max_response_time - (d_agent + d_target)) / remaining_time
            prioritization_ratio_items_1.append(prioritization_ratio_1)
            prioritization_ratio_items_2.append(prioritization_ratio_2)
            prioritization_ratio_items_3.append(prioritization_ratio_3)

            # Vector direction from agent to items
            if self.variant == 2:
                if loc in directions:
                    direction_to_item = {
                        "up": [-1, 0],
                        "down": [1, 0],
                        "left": [0, -1],
                        "right": [0, 1]
                    }.get(directions[loc], [0, 0])
                else:
                    direction_to_item = (0, 0)
                    vector_agent_to_items.extend(direction_to_item)
            else:
                vector_agent_to_items.extend([
                    self.agent_loc[0] - loc[0],
                    self.agent_loc[1] - loc[1],
                ])

            distances_target_to_items.extend([
                self.target_loc[0] - loc[0],
                self.target_loc[1] - loc[1],
            ])

            # Proximity/density score
            proximity_score = 1
            for other_loc in filtered_item_locs:
                if loc != other_loc:
                    if self.variant == 2:
                        distance_to_other = distances.get(other_loc, 0) / (self.vertical_cell_count * 2)
                    else:
                        distance_to_other = self.manhatten_dist(loc, other_loc) / (self.vertical_cell_count * 2)
                    proximity_score *= distance_to_other
            proximity_scores.append(proximity_score)

        # Pad 2d observations
        while len(position_items) < observation_tuple_size * 2:
            position_items.extend([pad_value, pad_value])
        while len(remaining_time_items) < observation_tuple_size:
            remaining_time_items.append(pad_value)
        while len(vector_agent_to_items) < observation_tuple_size * 2:
            vector_agent_to_items.extend([pad_value, pad_value])
        while len(distances_target_to_items) < observation_tuple_size:
            distances_target_to_items.append(pad_value)
        while len(prioritization_ratio_items_1) < observation_tuple_size:
            prioritization_ratio_items_1.append(pad_value)
        while len(prioritization_ratio_items_2) < observation_tuple_size:
            prioritization_ratio_items_2.append(pad_value)
        while len(prioritization_ratio_items_3) < observation_tuple_size:
            prioritization_ratio_items_3.append(pad_value)
        while len(proximity_scores) < observation_tuple_size:
            proximity_scores.append(pad_value)
        while len(manhatten_distance_agent_to_items) < observation_tuple_size:
            manhatten_distance_agent_to_items.append(pad_value)
        while len(manhatten_distance_target_to_items) < observation_tuple_size:
            manhatten_distance_target_to_items.append(pad_value)

        # Cummulative Timeout Risk (normalized)
        if len(filtered_item_locs) != 0:
            cumulative_timeout_risk = cumulative_items_time / len(filtered_item_locs)
        else:
            cumulative_timeout_risk = 1

        obs = np.concatenate([
            position_agent if self.obs_config.get("position_agent", True) else [],  # (x,y) 2
            distance_to_target if self.obs_config.get("distance_to_target", True) else [],  # z 1
            vector_to_target if self.obs_config.get("vector_to_target", True) else [],  # (x,y) 2
            vector_to_walls if self.obs_config.get("vector_to_walls", True) else [],  # (a,b,c,d) 4
            [free_capacity_agent] if self.obs_config.get("free_capacity_agent", True) else [],  # z (normalized) 1
            [cumulative_timeout_risk] if self.obs_config.get("cumulative_timeout_risk", True) else [],
            # z (normalized) 1

            position_items if self.obs_config.get("position_items", True) else [],  # ((x,y), (x,y),...) 16
            vector_agent_to_items if self.obs_config.get("vector_agent_to_items", True) else [],
            # ((x,y), (x,y),...) 16
            remaining_time_items if self.obs_config.get("remaining_time_items", True) else [],  # z (normalized) 8
            manhatten_distance_target_to_items if self.obs_config.get("manhatten_distance_target_to_items",
                                                                      True) else [],  # z (normalized) 8
            manhatten_distance_agent_to_items if self.obs_config.get("manhatten_distance_agent_to_items", True) else [],
            # z (normalized) 8
            prioritization_ratio_items_1 if self.obs_config.get("prioritization_ratio_items", 1) == 1 else [],
            prioritization_ratio_items_2 if self.obs_config.get("prioritization_ratio_items", 2) == 2 else [],
            prioritization_ratio_items_3 if self.obs_config.get("prioritization_ratio_items", 3) == 3 else [],

            closest_item if self.obs_config.get("closest_item", True) else [],  # 2
            vector_closest if self.obs_config.get("vector_closest", True) else [],  # 2

            proximity_scores if self.obs_config.get("proximity_scores", True) else [],  # 8
            [remaining_episode_time] if self.obs_config.get("remaining_episode_time", True) else [],
            # z (normalized) #1
        ])

        return obs

    def get_cnn_obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the state of the environment as a 3D grid with multiple channels."""
        grid_shape = (self.vertical_cell_count, self.horizontal_cell_count)

        if self.variant == 2:
            distances, directions = self.shortest_path(self.agent_loc)

        filtered_item_locs = []
        filtered_item_times = []
        # Filter reachable items and limit to observation_tuple_size
        for loc, time in zip(self.item_locs, self.item_times):
            if self.is_reachable(loc, self.max_response_time - time):
                filtered_item_locs.append(loc)
                filtered_item_times.append(time)

        # Channel 1: Agent position
        position_agent = np.zeros(grid_shape)
        position_agent[self.agent_loc] = 1

        # Channel 2: Item positions
        position_items = np.zeros(grid_shape)
        for loc in filtered_item_locs:
            position_items[loc] = 1

        # Channel 3: Remaining time of items
        remaining_time_items = np.zeros(grid_shape)
        for loc, time in zip(filtered_item_locs, filtered_item_times):
            remaining_time = (self.max_response_time - time) / self.max_response_time
            remaining_time_items[loc] = remaining_time

        # Channel 4: Prioritization ratio 1 of items
        prioritization_ratio_items_1 = np.zeros(grid_shape)
        prioritization_ratio_items_2 = np.zeros(grid_shape)
        prioritization_ratio_items_3 = np.zeros(grid_shape)
        for loc, time in zip(filtered_item_locs, filtered_item_times):
            if self.variant == 2:
                d_agent = distances.get(loc, 0)
                d_target = distances.get(self.target_loc, 0)
            else:
                d_agent = self.manhatten_dist(self.agent_loc, loc)
                d_target = self.manhatten_dist(loc, self.target_loc)

            remaining_time_after_collection = self.max_response_time - time - d_agent
            if remaining_time_after_collection > 0:
                prioritization_ratio_items_1[loc] = (
                        (1 / (d_target + 1)) + (1 / (d_agent + 1)) * (1 / (remaining_time + 1))
                )
                prioritization_ratio_items_2[loc] = (self.max_response_time - remaining_time) / (d_agent + d_target)
                prioritization_ratio_items_3[loc] = (self.max_response_time - (d_agent + d_target)) / remaining_time

        # Channel 5: Manhattan distance from agent to items
        manhatten_distance_agent_to_items = np.zeros(grid_shape)
        for loc in filtered_item_locs:
            if self.variant == 2:
                manhatten_distance_agent_to_items[loc] = distances.get(loc, 0)
            else:
                manhatten_distance_agent_to_items[loc] = self.manhatten_dist(self.agent_loc, loc)

        # Channel 6: Manhattan distance from items to target
        manhatten_distance_target_to_items = np.zeros(grid_shape)
        for loc in filtered_item_locs:
            if self.variant == 2:
                manhatten_distance_target_to_items[loc] = distances.get(self.target_loc, 0)
            else:
                manhatten_distance_target_to_items[loc] = self.manhatten_dist(loc, self.target_loc)

        # Channel 7: Proximity scores for items
        proximity_scores = np.zeros(grid_shape)
        for loc in filtered_item_locs:
            score = 1
            for other_loc in filtered_item_locs:
                if loc != other_loc:
                    if self.variant == 2:
                        distance_to_other = distances.get(other_loc, 0) / (self.vertical_cell_count * 2)
                    else:
                        distance_to_other = self.manhatten_dist(loc, other_loc) / (self.vertical_cell_count * 2)
                    score *= distance_to_other
            proximity_scores[loc] = score

        # Channel 8: Closest item
        closest_item = np.zeros(grid_shape)
        closest_idx = -1
        if filtered_item_locs:
            if self.variant == 2:
                item_distances = [distances.get(item, 0) for item in filtered_item_locs]
            else:
                item_distances = [self.manhatten_dist(self.agent_loc, loc) for loc in filtered_item_locs]
            closest_idx = np.argmin(item_distances)
            closest_item_loc = filtered_item_locs[closest_idx]
            closest_item[closest_item_loc] = 1

        # --- SCALAR VALUES ---
        # free capacity
        free_capacity_agent = (self.agent_capacity - self.agent_load) / self.agent_capacity

        # cumulative timeout risk
        cumulative_timeout_risk = sum(
            [(self.max_response_time - time) / self.max_response_time for time in filtered_item_times]
        ) / (len(filtered_item_times) or 1)

        # distance to target
        if self.variant == 2:
            distance_to_target = distances.get(self.target_loc, 0) / (self.vertical_cell_count * 2)
        else:
            distance_to_target = self.manhatten_dist(self.agent_loc, self.target_loc) / (self.vertical_cell_count * 2)

        # vector to walls
        if self.variant == 2:
            vector_to_walls = [
                min(
                    (
                        self.agent_loc[0] - row
                        for row in range(self.agent_loc[0], -1, -1)
                        if (row, self.agent_loc[1]) not in self.eligible_cells
                    ),
                    default=self.agent_loc[0],
                ) / 4,  # Up
                min(
                    (
                        row - self.agent_loc[0]
                        for row in range(self.agent_loc[0], self.vertical_cell_count)
                        if (row, self.agent_loc[1]) not in self.eligible_cells
                    ),
                    default=self.vertical_cell_count - self.agent_loc[0] - 1,
                ) / 4,  # Down
                min(
                    (
                        self.agent_loc[1] - col
                        for col in range(self.agent_loc[1], -1, -1)
                        if (self.agent_loc[0], col) not in self.eligible_cells
                    ),
                    default=self.agent_loc[1],
                ) / 4,  # Left
                min(
                    (
                        col - self.agent_loc[1]
                        for col in range(self.agent_loc[1], self.horizontal_cell_count)
                        if (self.agent_loc[0], col) not in self.eligible_cells
                    ),
                    default=self.horizontal_cell_count - self.agent_loc[1] - 1,
                ) / 4,  # Right
            ]

            # Apply overrides based on the agent's current position
            if self.agent_loc in [(0, 0), (1, 0), (2, 0), (1, 2), (2, 2), (3, 2), (4, 2)]:
                vector_to_walls[3] = 0  # "Right"
            if self.agent_loc in [(0, 2), (1, 2), (2, 2), (1, 4), (2, 4), (3, 4), (4, 4)]:
                vector_to_walls[2] = 0  # "Left"
            if self.agent_loc == (3, 1):
                vector_to_walls[0] = 0  # "Up"
            if self.agent_loc == (0, 3):
                vector_to_walls[1] = 0  # "Down"

            # New Overrides
            if self.agent_loc in [(3, 0), (4, 0)]:
                vector_to_walls[3] = 2  # "Right"
            if self.agent_loc in [(3, 1), (4, 1)]:
                vector_to_walls[3] = 1  # "Right"
            if self.agent_loc == (4, 1):
                vector_to_walls[0] = 1  # "Up"
            if self.agent_loc == (0, 4):
                vector_to_walls[2] = 2  # "Left"
            if self.agent_loc == (0, 3):
                vector_to_walls[2] = 1  # "Left"

        else:
            # Standard vector_to_walls calculation
            vector_to_walls = [
                self.agent_loc[0] / 4,
                self.vertical_cell_count - 1 - self.agent_loc[0] / 4,
                self.agent_loc[1] / 4,
                self.horizontal_cell_count - 1 - self.agent_loc[1] / 4,
            ]

        # vector to closest item
        vector_closest = [0, 0]
        if filtered_item_locs and closest_idx != -1:
            if self.variant == 2:
                closest_item_tuple = tuple(closest_item_loc)
                if closest_item_tuple in directions:
                    vector_closest = {
                        "up": [-1, 0],
                        "down": [1, 0],
                        "left": [0, -1],
                        "right": [0, 1]
                    }.get(directions[closest_item_tuple], [0, 0])
                else:
                    vector_closest = [0, 0]

            else:
                vector_closest = [
                    (closest_item_loc[0] - self.agent_loc[0]) / self.vertical_cell_count,
                    (closest_item_loc[1] - self.agent_loc[1]) / self.horizontal_cell_count,
                ]

        # vector to target
        vector_to_target = [0, 0]
        if self.variant == 2:
            if self.target_loc in directions:
                vector_to_target = {
                    "up": [-1, 0],
                    "down": [1, 0],
                    "left": [0, -1],
                    "right": [0, 1]
                }.get(directions[self.target_loc], [0, 0])
            else:
                vector_to_target = [0, 0]
        else:
            vector_to_target = [
                (self.target_loc[0] - self.agent_loc[0]) / self.vertical_cell_count,
                (self.target_loc[1] - self.agent_loc[1]) / self.horizontal_cell_count,
            ]

        # Remaining Episode Time (normalized)
        remaining_episode_time = (self.episode_steps - self.step_count) / self.episode_steps

        channels = []
        if self.obs_config.get("position_agent", True):
            channels.append(position_agent)
        if self.obs_config.get("position_items", True):
            channels.append(position_items)
        if self.obs_config.get("remaining_time_items", True):
            channels.append(remaining_time_items)
        if self.obs_config.get("prioritization_ratio_items", 1) == 1:
            channels.append(prioritization_ratio_items_1)
        if self.obs_config.get("prioritization_ratio_items", 2) == 2:
            channels.append(prioritization_ratio_items_2)
        if self.obs_config.get("prioritization_ratio_items", 3) == 3:
            channels.append(prioritization_ratio_items_3)
        if self.obs_config.get("manhatten_distance_agent_to_items", True):
            channels.append(manhatten_distance_agent_to_items)
        if self.obs_config.get("manhatten_distance_target_to_items", True):
            channels.append(manhatten_distance_target_to_items)
        if self.obs_config.get("proximity_scores", True):
            channels.append(proximity_scores)
        if self.obs_config.get("closest_item", True):
            channels.append(closest_item)

        # Stack channels
        grid_state = np.stack(channels, axis=-1)

        scalar_values = []

        if self.obs_config.get("free_capacity_agent", True):
            scalar_values.append([free_capacity_agent])

        if self.obs_config.get("cumulative_timeout_risk", True):
            scalar_values.append([cumulative_timeout_risk])

        if self.obs_config.get("distance_to_target", True):
            scalar_values.append([distance_to_target])

        if self.obs_config.get("remaining_episode_time", True):
            scalar_values.append([remaining_episode_time])

        if self.obs_config.get("vector_to_walls", True):
            scalar_values.append(vector_to_walls)

        if self.obs_config.get("vector_closest", True):
            scalar_values.append(vector_closest)

        if self.obs_config.get("vector_to_target", True):
            scalar_values.append(vector_to_target)

        scalar_state = np.concatenate(scalar_values).astype(float)

        return grid_state, scalar_state
