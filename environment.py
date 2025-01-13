# actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

# positions in grid:
# - (0,0) is upper left corner
# - first index is vertical (increasing from top to bottom)
# - second index is horizontal (increasing from left to right)

# if new item appears in a cell into which the agent moves/at which the agent stays in the same time step,
# it is not picked up (if agent wants to pick it up, it has to stay in the cell in the next time step)

import random
from typing import List, Tuple
import pandas as pd
from copy import deepcopy
from itertools import compress
import numpy as np

from renderer.renderer import Renderer

class Environment(object):
    def __init__(self, variant, data_dir, render = False):
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

    # Initializing new episode (specify if training, validation, or testing via the mode argument)
    def reset(self, mode, structure = "DNN"):
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
        #self.item_distances = []

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
        self.data = pd.read_csv(self.data_dir + f"/variant_{self.variant}/episode_data/episode_{episode:03d}.csv", index_col=0)

        # return self.get_new_obs()
        return self.get_obs_ppo_test()

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
        if (self.agent_load < self.agent_capacity) and (self.agent_loc in self.item_locs):
            self.agent_load += 1
            idx = self.item_locs.index(self.agent_loc)
            self.item_locs.pop(idx)
            self.item_times.pop(idx)
            rew += self.reward / 2

        # item drop-off
        if self.agent_loc == self.target_loc:
            rew += self.agent_load * self.reward / 2
            #FOR RENDER
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
        self.item_locs += new_items
        self.item_times += [0] * len(new_items)

        # get new observation
        #next_obs = self.get_new_obs()
        next_obs = self.get_obs_ppo_test()

        if self.render_enabled:
            self.update_render()

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
            #state = self.reset(mode = "training", structure = structure).shape
            # return state
            grid_state, scalar_state = self.reset(mode="training", structure=structure)

            return grid_state.shape, scalar_state.shape
            
        
    def manhatten_dist(self, pos: Tuple[int, int], tgt: Tuple[int, int]) -> Tuple[int, int]:
        return abs(pos[0] - tgt[0]) + abs(pos[1] - tgt[1])
    
    def is_reachable(self, item_loc: Tuple[int, int], remaining_time: int) -> bool:
        """
        Check if an item is reachable before its time runs out.
        
        Parameters:
            item_loc (Tuple[int, int]): The location of the item.
            remaining_time (int): Remaining time for the item.
            
        Returns:
            bool: True if reachable, False otherwise.
        """
        distance = self.manhatten_dist(self.agent_loc, item_loc)
        return distance <= remaining_time
    
    def get_dnn_obs(self, two_dim=True) -> np.ndarray:
        """
        Returns the default observation of the environment, which is 2D encoded.
        """
        if two_dim:
            return self.get_dnn_obs_2d()
        else:
            return self.get_dnn_obs_one_hot()
    
    
    def get_dnn_obs_2d(self) -> np.ndarray:
        """Returns the 2D encoded observations of the environment."""
        grid_size = self.vertical_cell_count * self.horizontal_cell_count

        # Free Capacity
        free_capacity = [self.agent_capacity - self.agent_load]

        # Agent Position (2D Flattened)
        agent_position_2d = [self.agent_loc[0], self.agent_loc[1]]

        # Initialize observations
        item_positions_2d = []
        remaining_item_times_2d = []
        prioritization_ratios = np.zeros(grid_size)
        distances_all_items_2d = []
        closest_item_2d = [-1, -1]
        cumulative_timeout_risk = 0

        filtered_item_locs = []
        filtered_item_times = []

        # Filter reachable items
        for loc, time in zip(self.item_locs, self.item_times):
            if self.is_reachable(loc, self.max_response_time - time):
                filtered_item_locs.append(loc)
                filtered_item_times.append(time)

        # Process filtered items for observations
        closest_idx = -1
        if filtered_item_locs:
            distances = [self.manhatten_dist(self.agent_loc, item) for item in filtered_item_locs]
            closest_idx = np.argmin(distances)
            closest_item_2d = [filtered_item_locs[closest_idx][0], filtered_item_locs[closest_idx][1]]

        # Process filtered items for observations
        for loc, time in zip(filtered_item_locs, filtered_item_times):
            item_positions_2d.extend([loc[0], loc[1]])

            remaining_time = (self.max_response_time - time) / self.max_response_time
            remaining_item_times_2d.append(remaining_time)

            cumulative_timeout_risk += remaining_time

            d_agent = self.manhatten_dist(self.agent_loc, loc)
            d_target = self.manhatten_dist(loc, self.target_loc)
       
            ratio = ((1 / (d_target + 1)) + (1 / (d_agent + 1)))
            prioritization_ratios[loc[0] * self.horizontal_cell_count + loc[1]] = ratio

            distances_all_items_2d.extend([
                abs(self.agent_loc[0] - loc[0]),
                abs(self.agent_loc[1] - loc[1]),
            ])

        # Pad _2d observations
        while len(item_positions_2d) < grid_size + 1:
            item_positions_2d.extend([-1, -1])
        while len(remaining_item_times_2d) < grid_size + 1:
            remaining_item_times_2d.append(-1)
        while len(distances_all_items_2d) < grid_size + 1:
            distances_all_items_2d.extend([-1, -1])

        # Distance to Target
        distance_to_target = [
            self.manhatten_dist(self.agent_loc, self.target_loc)
        ]

        # Distance to Closest Item
        distance_closest_item_2d = [-1, -1]
        if closest_idx != -1:
            distance_closest_item_2d = [
                abs(self.agent_loc[0] - filtered_item_locs[closest_idx][0]),
                abs(self.agent_loc[1] - filtered_item_locs[closest_idx][1]),
            ]

        # Distance to Walls
        distance_to_walls = [
            self.agent_loc[0],  # Up
            self.vertical_cell_count - 1 - self.agent_loc[0],  # Down
            self.agent_loc[1],  # Left
            self.horizontal_cell_count - 1 - self.agent_loc[1],  # Right
        ]

        obs = np.concatenate([
            agent_position_2d,
            distance_to_target,
            distance_to_walls,
            free_capacity,
            [cumulative_timeout_risk],
            
            item_positions_2d,
            distances_all_items_2d,
            remaining_item_times_2d,
            prioritization_ratios,

            closest_item_2d,
            distance_closest_item_2d,   
        ])

        return obs

    def get_dnn_obs_one_hot(self) -> np.ndarray:
        """Returns the one-hot encoded observations of the environment."""
        grid_size = self.vertical_cell_count * self.horizontal_cell_count

        # Free Capacity
        free_capacity = [self.agent_capacity - self.agent_load]

        # Agent Position (One-Hot Encoded)
        agent_position = np.zeros(grid_size)
        agent_index = self.agent_loc[0] * self.horizontal_cell_count + self.agent_loc[1]
        agent_position[agent_index] = 1

        # Initialize observations
        item_positions = np.zeros(grid_size)
        remaining_item_times = np.full(grid_size, -1.0)
        prioritization_ratios = np.zeros(grid_size)
        closest_item = np.zeros(grid_size)

        filtered_item_locs = []
        filtered_item_times = []

        # Filter reachable items
        for loc, time in zip(self.item_locs, self.item_times):
            if self.is_reachable(loc, self.max_response_time - time):
                filtered_item_locs.append(loc)
                filtered_item_times.append(time)
            
        # Process filtered items for observations
        closest_idx = -1
        if filtered_item_locs:
            distances = [self.manhatten_dist(self.agent_loc, item) for item in filtered_item_locs]
            closest_idx = np.argmin(distances)
            closest_item_index = filtered_item_locs[closest_idx][0] * self.horizontal_cell_count + filtered_item_locs[closest_idx][1]
            closest_item[closest_item_index] = 1

        # Process filtered items for observations
        for loc, time in zip(filtered_item_locs, filtered_item_times):
            item_index = loc[0] * self.horizontal_cell_count + loc[1]
            item_positions[item_index] = 1

            remaining_time = (self.max_response_time - time) / self.max_response_time
            remaining_item_times[item_index] = remaining_time

            d_agent = self.manhatten_dist(self.agent_loc, loc)
            d_target = self.manhatten_dist(loc, self.target_loc)
            remaining_time_after_collection = self.max_response_time - time - d_agent
            if remaining_time_after_collection > 0:
                ratio = ((1 / (d_target + 1)) + (1 / (d_agent + 1)))
                prioritization_ratios[item_index] = ratio

        # Distance to Closest Item
        distance_closest_item = [-1]
        if closest_idx != -1:
            distance_closest_item = [distances[closest_idx]]

        # Distance to Walls
        distance_to_walls = [
            self.agent_loc[0],  # Up
            self.vertical_cell_count - 1 - self.agent_loc[0],  # Down
            self.agent_loc[1],  # Left
            self.horizontal_cell_count - 1 - self.agent_loc[1],  # Right
        ]

        obs = np.concatenate([
            free_capacity,
            prioritization_ratios,
            agent_position,
            item_positions,
            remaining_item_times,
            distance_closest_item,
            distance_to_walls,
        ])

        return obs

    #CNN OBS
    def get_cnn_obs(self) -> np.ndarray:
        """Returns the state of the environment as a 3D grid with multiple channels."""
        grid_shape = (self.vertical_cell_count, self.horizontal_cell_count)
        cumulative_timeout_risk = 0

        # Channel 1: Agent position
        agent_channel = np.zeros(grid_shape)
        agent_channel[self.agent_loc] = 1

        # Channel 2: Item positions
        item_positions_channel = np.zeros(grid_shape)
        for loc in self.item_locs:
            item_positions_channel[loc] = 1

        # Channel 3: Remaining times of items
        remaining_times_channel = np.zeros(grid_shape)
        for loc, time in zip(self.item_locs, self.item_times):
            remaining_time = (self.max_response_time - time) / self.max_response_time
            remaining_times_channel[loc] = remaining_time
            cumulative_timeout_risk += remaining_time

        # Channel 5: Free capacity
        free_capacity_channel = np.full(grid_shape, self.agent_capacity - self.agent_load)

        # Channel 6: Manhattan distances to all grid cells
        distances_channel = np.zeros(grid_shape)
        for i in range(self.vertical_cell_count):
            for j in range(self.horizontal_cell_count):
                distances_channel[i, j] = self.manhatten_dist(self.agent_loc, (i, j))

        # Channel 7: Target location
        target_channel = np.zeros(grid_shape)
        target_channel[self.target_loc] = 1

        # Channel 8: Prioritization ratios
        prioritization_channel = np.zeros(grid_shape)
        for loc, time in zip(self.item_locs, self.item_times):
            d_agent = self.manhatten_dist(self.agent_loc, loc)
            d_target = self.manhatten_dist(loc, self.target_loc)
            remaining_time_after_collection = self.max_response_time - time - d_agent
            if remaining_time_after_collection > 0:
                prioritization_channel[loc] = ((1 / (d_target + 1)) + (1 / (d_agent + 1)))

        free_capacity = self.agent_capacity - self.agent_load

        # Distance to Target
        distance_to_target = [
            self.manhatten_dist(self.agent_loc, self.target_loc)
        ]

        state = np.stack(
            [agent_channel,
            prioritization_channel,
            item_positions_channel,
            distances_channel,
            remaining_times_channel,
            ],
            axis=-1,
        )

        scalar_state = np.array([free_capacity, distance_to_target[0], cumulative_timeout_risk])

        return state, scalar_state
    
    #GREEDY OBS
    def get_to_tgt(self, tgt):
        """Determines which action to take to get to tgt coordinates."""
        if self.agent_loc[0] > tgt[0]:
            act = 1
        elif self.agent_loc[0] < tgt[0]:
            act = 3
        elif self.agent_loc[1] > tgt[1]:
            act = 4
        elif self.agent_loc[1] < tgt[1]:
            act = 2
        else:
            act = 0

        return act

    def get_greedy_obs(self):
        """Greedy get_obs. Only works with base variant!"""

        # Case 1: Agent already carries an item
        if self.agent_load == 1:
            # Take direct route to drop-off
            act = self.get_to_tgt([2, 0])  # drop-off is at (2,0)

        # Case 2: Agent carries no item and no item has spawned on the grid => do nothing
        elif self.agent_load == 0 and len(self.item_locs) == 0:
            # Stay at current position
            act = 0

        # Case 3: Agent carries no item and item has spawned => Pick up closest item
        else:
            # Calculate distances to all items
            item_dist = [self.manhatten_dist(self.agent_loc, item) for item in self.item_locs]

            # Calculate time each item will still remain on the grid
            remaining_times = [(self.max_response_time - i) for i in self.item_times]

            # Make unreachable items unattractive
            for i, _ in enumerate(item_dist):
                if item_dist[i] > remaining_times[i]:  # if unreachable
                    item_dist[i] = 1e5  # set distance to unreachable items very high

            # Pick closest item
            closest_item_idx = item_dist.index(min(item_dist))
            closest_item = self.item_locs[closest_item_idx]

            # Determine best route to closest item
            act = self.get_to_tgt(closest_item)

        return np.array([act])


    def get_obs_ppo_test(self):
        grid_size = self.vertical_cell_count * self.horizontal_cell_count

        # Free Capacity
        free_capacity = [self.agent_capacity - self.agent_load]

        # Agent Position (One-Hot Encoded)
        agent_position = np.zeros(grid_size)
        agent_index = self.agent_loc[0] * self.horizontal_cell_count + self.agent_loc[1]
        agent_position[agent_index] = 1

        # Agent Position (2D Flattened)
        agent_position_2d = [self.agent_loc[0], self.agent_loc[1]]

        # Item Positions (One-Hot Encoded)
        item_positions = np.zeros(grid_size)
        item_positions_2d = []
        for loc in self.item_locs:
            item_index = loc[0] * self.horizontal_cell_count + loc[1]
            item_positions[item_index] = 1
            item_positions_2d.extend([loc[0], loc[1]])
        # Pad item_positions_2d if fewer items than grid cells
        while len(item_positions_2d) < grid_size * 2:
            item_positions_2d.extend([-1, -1])

        # Remaining Item Times
        remaining_item_times = np.full(grid_size, -1.0)
        # Remaining Item Times (Normalized, Ordered to Match item_positions_2d)
        remaining_item_times_2d = []
        for loc, time in zip(self.item_locs, self.item_times):
            item_index = loc[0] * self.horizontal_cell_count + loc[1]
            remaining_time = (self.max_response_time - time) / self.max_response_time
            remaining_item_times[item_index] = remaining_time
            remaining_item_times_2d.append(remaining_time)
        # Pad remaining_item_times_2d
        while len(remaining_item_times_2d) < grid_size:
            remaining_item_times_2d.append(-1)

        # Frequency of Past Items
        # past_items_vector = self.past_items.flatten()
        # frequency_past_items = past_items_vector / past_items_vector.sum() if past_items_vector.sum() > 0 else past_items_vector

        # Target Position (One-Hot Encoded)
        target_position = np.zeros(grid_size)
        target_index = self.target_loc[0] * self.horizontal_cell_count + self.target_loc[1]
        target_position[target_index] = 1

        # Target Position (2D Flattened)
        target_position_2d = [self.target_loc[0], self.target_loc[1]]

        # Closest Item (One-Hot Encoded)
        closest_item = np.zeros(grid_size)
        if self.item_locs:
            distances = [self.manhatten_dist(self.agent_loc, item) for item in self.item_locs]
            closest_idx = np.argmin(distances)
            closest_item_index = self.item_locs[closest_idx][0] * self.horizontal_cell_count + \
                                 self.item_locs[closest_idx][1]
            closest_item[closest_item_index] = 1
            closest_item_2d = [self.item_locs[closest_idx][0], self.item_locs[closest_idx][1]]
            distance_closest_item = [distances[closest_idx]]
            distance_closest_item_2d = [
                abs(self.agent_loc[0] - self.item_locs[closest_idx][0]),
                abs(self.agent_loc[1] - self.item_locs[closest_idx][1]),
            ]
        else:
            closest_item_2d = [-1, -1]
            distance_closest_item = [-1]
            distance_closest_item_2d = [-1, -1]

        distances_all_items = np.full(grid_size, -1, dtype=float)
        distances_all_items_2d = []
        for loc in self.item_locs:
            item_index = loc[0] * self.horizontal_cell_count + loc[1]
            distance = self.manhatten_dist(self.agent_loc, loc)
            distances_all_items[item_index] = self.manhatten_dist(self.agent_loc, loc)
            distances_all_items_2d.extend([
                abs(self.agent_loc[0] - loc[0]),
                abs(self.agent_loc[1] - loc[1]),
            ])
        # Pad distances_all_items_2d
        while len(distances_all_items_2d) < grid_size * 2:
            distances_all_items_2d.extend([-1, -1])

        # Distance to Walls
        distance_to_walls = [
            self.agent_loc[0],  # Up
            self.vertical_cell_count - 1 - self.agent_loc[0],  # Down
            self.agent_loc[1],  # Left
            self.horizontal_cell_count - 1 - self.agent_loc[1],  # Right
        ]

        prioritization_ratios = np.zeros(grid_size)
        for loc, time in zip(self.item_locs, self.item_times):
            d_agent = self.manhatten_dist(self.agent_loc, loc)
            d_target = self.manhatten_dist(loc, self.target_loc)
            remaining_time_after_collection = self.max_response_time - time - d_agent
            if remaining_time_after_collection > 0:
                # ratio = (1 / (remaining_time_after_collection + 1)) + (1 / (d_target + 1))
                # Own
                ratio = ((1 / (d_target + 1)) + (1 / (d_agent + 1)))
                prioritization_ratios[loc[0] * self.horizontal_cell_count + loc[1]] = ratio

        # Other features remain unchanged
        free_capacity = [self.agent_capacity - self.agent_load]
        agent_position_2d = [self.agent_loc[0], self.agent_loc[1]]
        item_positions_2d = [coord for loc in self.item_locs for coord in loc]
        remaining_item_times_2d = [(self.max_response_time - time) / self.max_response_time
                                   for time in self.item_times]

        # Pad features for consistency
        while len(item_positions_2d) < grid_size * 2:
            item_positions_2d.extend([-1, -1])
        while len(remaining_item_times_2d) < grid_size:
            remaining_item_times_2d.append(-1)

        # ADD: Distances to all cells
        obs = np.concatenate([
            free_capacity,  # Scalar
            prioritization_ratios,
            agent_position_2d,  # 2D Flattened e.g. [1 0]
            item_positions_2d,
            remaining_item_times_2d,
            closest_item_2d,
            distance_closest_item_2d,
            distances_all_items_2d,
            distance_to_walls  # Scalar e.g. [1 3 2 2]
        ])

        return obs
