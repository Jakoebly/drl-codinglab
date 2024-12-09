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

from Renderer.renderer import Renderer


def manhatten_dist(pos: Tuple[int, int], tgt: Tuple[int, int]) -> Tuple[int, int]:
    return abs(pos[0] - tgt[0]) + abs(pos[1] - tgt[1])

class Environment(object):
    def __init__(self, variant, data_dir, render=False):
        self.variant = variant
        self.vertical_cell_count = 5
        self.horizontal_cell_count = 5
        self.vertical_idx_target = 2
        self.horizontal_idx_target = 0
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

        # RENDERING ONLY:
        self.render_enabled = render

        if self.render_enabled:
            self.renderer = Renderer(grid_size=(self.vertical_cell_count, self.horizontal_cell_count), eligible_cells=self.eligible_cells, max_response_time=self.max_response_time)
        
        self.current_episode_target_count = 0  # Counts number of items dropped off at target cell

    # initialize a new episode (specify if training, validation, or testing via the mode argument)
    def reset(self, mode):
        modes = ["training", "validation", "testing"]
        if mode not in modes:
            raise ValueError("Invalid mode. Expected one of: %s" % modes)

        self.step_count = 0
        self.agent_loc = (self.vertical_idx_target, self.horizontal_idx_target)
        self.agent_load = 0  # number of items loaded (0 or 1, except for first extension, where it can be 0,1,2,3)
        self.item_locs = []
        self.item_times = []

        self.past_items = []
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
        
        #For CNN:
        self.past_items = np.zeros((self.vertical_cell_count, self.horizontal_cell_count), dtype=int)

        #For renderer:
        self.current_episode_target_count = 0
        self.update_render()
        #self.target_count = 0

        return self.get_obs()

    # take one environment step based on the action act
    def step(self, act):
        self.step_count += 1
        rew = 0

        # done signal (1 if episode ends, 0 if not)
        if self.step_count == self.episode_steps:
            done = 1
        else:
            done = 0

        #MP
        new_loc = self.agent_loc

        # agent movement
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
            #self.target_count += self.agent_load
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

        #FOR CNN
        for loc in new_items:
            self.past_items[loc] += 1

        # get new observation
        next_obs = self.get_obs()

        #For Render:
        self.update_render()

        return rew, next_obs, done

    def update_render(self):
        """Update the render only if enabled."""
        if self.render_enabled:
            self.renderer.render(
                agent_loc=self.agent_loc,
                target_loc=self.target_loc,
                item_locs=self.item_locs,
                item_times=self.item_times,
                agent_load=self.agent_load,
                target_count=self.current_episode_target_count
            )
    

    # TODO: PADDING
    # TODO: Item Position RELATIV not absolut
    # TODO: Manhatten distance to each item (new list)
    # TODO: Manhatten distance from item to target
    # TODO: Number of items present on field
    # TODO: Abstand zur nächsten wand
    # TODO: Past Items Vector
    # TODO: Delte items where distance > remaining time

    def get_action_len(self):
        """Returns the number of actions available."""
        return 5

    def get_new_obs(self) -> np.ndarray:
        """Returns the state of the environment as a flattened 1D array."""
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
        past_items_vector = self.past_items.flatten()
        frequency_past_items = past_items_vector / past_items_vector.sum() if past_items_vector.sum() > 0 else past_items_vector

        # Target Position (One-Hot Encoded)
        target_position = np.zeros(grid_size)
        target_index = self.target_loc[0] * self.horizontal_cell_count + self.target_loc[1]
        target_position[target_index] = 1

        # Target Position (2D Flattened)
        target_position_2d = [self.target_loc[0], self.target_loc[1]]

        # Closest Item (One-Hot Encoded)
        closest_item = np.zeros(grid_size)
        if self.item_locs:
            distances = [manhatten_dist(self.agent_loc, item) for item in self.item_locs]
            closest_idx = np.argmin(distances)
            closest_item_index = self.item_locs[closest_idx][0] * self.horizontal_cell_count + self.item_locs[closest_idx][1]
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
            distance = manhatten_dist(self.agent_loc, loc)
            distances_all_items[item_index] = manhatten_dist(self.agent_loc, loc)
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
            d_agent = manhatten_dist(self.agent_loc, loc)
            d_target = manhatten_dist(loc, self.target_loc)
            remaining_time_after_collection = self.max_response_time - time - d_agent
            if remaining_time_after_collection > 0:
                #ratio = (1 / (remaining_time_after_collection + 1)) + (1 / (d_target + 1))
                #Own
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

        #ADD: Distances to all cells
        obs = np.concatenate([
            free_capacity,  # Scalar
            prioritization_ratios,
            #agent_position,  # One-hot
            agent_position_2d,  # 2D Flattened e.g. [1 0]
            #item_positions,  # One-hot e.g. [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            item_positions_2d,  # 2D Flattened e.g. [ 0  1  3  3  0  2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1-1 -1]
            #remaining_item_times,  # Normalized times e.g. [-1.  -1.  -1.  -1.  -1.  -1.   0.6 -1.  -1.   0.4 -1.  -1.  -1.  -1.-1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1. ]
            remaining_item_times_2d,  # Times according to 2D Positions e.g. [ 0.3  0.4  0.9 -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1. -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1.  -1. ]
            #frequency_past_items,  # Past frequency
            #target_position,  # One-hot e.g. [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.0.]
            #target_position_2d,  # 2D Flattened e.g. [2 0]
            #closest_item,  # One-hot e.g. [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.0.]
            #closest_item_2d,  # 2D Flattened e.g. [4 2]
            #distance_closest_item,  # Scalar e.g. [2]
            #distance_closest_item_2d,  # 2D Flattened e.g. [0 2]
            #distances_all_items,  # Scalar e.g. [-1.  6. -1. -1.  3. -1. -1.  4. -1.  2. -1. -1. -1. -1. -1. -1. -1. -1.-1.  0. -1. -1. -1. -1. -1.]
            distances_all_items_2d,  # 2D Flattened e.g. [ 1  3  2  0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1-1 -1]
            distance_to_walls  # Scalar e.g. [1 3 2 2]
        ])

        return obs
    
    def get_state_len(self):
        """Returns the length of one state i.e., the flattened array returned by the get_obs() method."""
        return 157

   # GRIDOBS
    def get_grid_obs(self) -> List[float]:
        grid_shape = (self.vertical_cell_count, self.horizontal_cell_count)

        # Agent position
        agent_position = np.zeros(grid_shape)
        agent_position[self.agent_loc] = 1

        # Item positions
        item_positions = np.zeros(grid_shape)
        for loc in self.item_locs:
            item_positions[loc] = 1

        # Remaining times of items (normalized)
        remaining_times = np.zeros(grid_shape)
        for loc, time in zip(self.item_locs, self.item_times):
            remaining_times[loc] = (self.max_response_time - time) / self.max_response_time

        # TODO: evtl. weg weil eigentlich nicht relevant für die entscheidung
        # Past items (spawn counts at each position)
        past_items_vector = self.past_items_vector.flatten()

        total_spawns = past_items_vector.sum()
        if total_spawns > 0:
            past_items_vector = past_items_vector / total_spawns
        else:
            past_items_vector = np.zeros_like(past_items_vector)

        # Target location
        target_location = np.zeros(grid_shape)
        target_location[self.target_loc] = 1

        # Free capacity
        free_capacity = self.agent_capacity - self.agent_load

        # Manhattan distances to items and distance to closest item
        distance_to_closest_item = 1000
        remaining_time_closest_item = 1000
        distance_closest_item_to_target = 1000
        manhattan_distance_items = np.zeros(grid_shape)
        for loc in self.item_locs:
            manhattan_distance_items[loc] = manhatten_dist(self.agent_loc, loc)

            if manhattan_distance_items[loc] < distance_to_closest_item:
                distance_to_closest_item = manhattan_distance_items[loc]
                remaining_time_closest_item = remaining_times[loc]
                distance_closest_item_to_target = manhatten_dist(loc, self.target_loc)

        # TODO: weg lassen
        # Manhatten distance to target
        manhattan_distance_target = np.zeros(grid_shape)
        manhattan_distance_target[self.target_loc] = manhatten_dist(self.agent_loc, self.target_loc)


        # Distance to the closest wall
        distance_to_walls = [
            self.agent_loc[0],
            self.vertical_cell_count - 1 - self.agent_loc[0],
            self.agent_loc[1],
            self.horizontal_cell_count - 1 - self.agent_loc[1]
        ]

        # Current number of items on the field
        num_items = len(self.item_locs)

        state = np.concatenate([
            np.array(agent_position).flatten(), # Dim 25
            np.array(item_positions).flatten(), # Dim 25
            np.array(remaining_times).flatten(), # Dim 25
            np.array(target_location).flatten(), # Dim 25
            np.array(free_capacity).flatten(), # Dim 1
            np.array(manhattan_distance_items).flatten(), # Dim 25
            np.array(manhattan_distance_target).flatten(),  # Dim 25
            np.array(distance_to_closest_item).flatten(),  # Dim 1
            np.array(remaining_time_closest_item).flatten(),  # Dim 1
            np.array(distance_closest_item_to_target).flatten(),  # Dim 1
            np.array(distance_to_walls).flatten(), # Dim 4
            np.array(num_items).flatten(), # Dim 1
        ])

        return state

    #CNN OBS
    def get_obs(self) -> np.ndarray:
        """Returns the state of the environment as a 3D grid with multiple channels."""
        grid_shape = (self.vertical_cell_count, self.horizontal_cell_count)

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
            remaining_times_channel[loc] = (self.max_response_time - time) / self.max_response_time

        # Channel 4: Past items (spawn counts at each position)
        past_items_channel = np.copy(self.past_items)

        total_spawns = past_items_channel.sum()
        if total_spawns > 0:
            past_items_channel = past_items_channel / total_spawns
        else:
            past_items_channel = np.zeros_like(past_items_channel)


        # Channel 5: Free capacity
        free_capacity_channel = np.full(grid_shape, self.agent_capacity - self.agent_load)

        # Channel 6: Manhattan distances to all grid cells
        distances_channel = np.zeros(grid_shape)
        for i in range(self.vertical_cell_count):
            for j in range(self.horizontal_cell_count):
                distances_channel[i, j] = manhatten_dist(self.agent_loc, (i, j))

        # Channel 7: Target location
        target_channel = np.zeros(grid_shape)
        target_channel[self.target_loc] = 1

        # Channel 8: Prioritization ratios
        prioritization_channel = np.zeros(grid_shape)
        for loc, time in zip(self.item_locs, self.item_times):
            d_agent = manhatten_dist(self.agent_loc, loc)
            d_target = manhatten_dist(loc, self.target_loc)
            remaining_time_after_collection = self.max_response_time - time - d_agent
            if remaining_time_after_collection > 0:
                #NEW
                #prioritization_channel[loc] = (1 / (remaining_time_after_collection + 1)) + (1 / (d_target + 1))
                #OLD
                #prioritization_channel[loc] = ((remaining_time_after_collection / (d_target + 1)) + (1 / (d_agent + 1)))
                #Own
                prioritization_channel[loc] = ((1 / (d_target + 1)) + (1 / (d_agent + 1)))

        # Stack all channels into a 3D grid
        state = np.stack(
            [agent_channel,
            prioritization_channel,
            item_positions_channel,
            #distances_channel,
            remaining_times_channel,
            past_items_channel,
            #target_channel,
            free_capacity_channel
            ],
            axis=-1,
        )

        return state
    
    def get_state_shape(self):
        return (6, 5, 5)  # (Channels, Height, Width)

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

    #Greedy
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
            item_dist = [manhatten_dist(self.agent_loc, item) for item in self.item_locs]

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
