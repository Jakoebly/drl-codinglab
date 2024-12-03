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

        # fmt: off
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
        # fmt: on

        # RENDERING ONLY:
        if render:
            self.renderer = Renderer(grid_size=(self.vertical_cell_count, self.horizontal_cell_count), eligible_cells=self.eligible_cells, max_response_time=self.max_response_time)
        self.target_count = 0  # Counts number of items dropped off at target cell

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
        self.item_distances = []

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
        self.past_items_vector = np.zeros((self.vertical_cell_count, self.horizontal_cell_count), dtype=int)

        return self.get_obs()
        # return self.get_cnn_obs()

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
            self.target_count += self.agent_load
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
            self.past_items_vector[loc] += 1

        # get new observation
        next_obs = self.get_obs()
        # next_obs = self.get_cnn_obs()

        return rew, next_obs, done

    def render(self):
        """
        Call the renderer to visualize the environment.
        """
        if self.renderer:
            self.renderer.render(agent_loc=self.agent_loc, target_loc=self.target_loc, item_locs=self.item_locs, item_times=self.item_times, agent_load=self.agent_load, target_count=self.target_count)
        else:
            print("Renderer not initialized.")

    # DONE: implement function that gives the input features for the neural network(s)
    #       based on the current state of the environment

    # TODO: PADDING
    # TODO: Item Position RELATIV not absolut
    # TODO: Manhatten distance to each item (new list)
    # TODO: Manhatten distance from item to target
    # TODO: Number of items present on field
    # TODO: Abstand zur nächsten wand
    # TODO: Past Items Vector
    # TODO: Delte items where distance > remaining time

    # Updated old get obs which directly includes flattening
    """def get_obs2(self):
        agent_pos = self.agent_loc
        free_capacity = self.agent_capacity - self.agent_load
        item_locs = list.copy(self.item_locs)
        item_times = [(self.max_response_time - i) for i in list.copy(self.item_times)]

        # Zero Padding for fixed number of items (24)
        while len(item_locs) < 24:
            item_locs.append((-10000, -10000))
            item_times.append(0)

        obs = np.concatenate([np.array(agent_pos).flatten(), np.array([free_capacity]), np.array(item_locs).flatten(), np.array(item_times).flatten()])

        return obs  # len(obs) == 75"""

    def get_obs_grid(self) -> List[float]:
        grid_shape = (self.vertical_cell_count, self.horizontal_cell_count)

        # Agent position
        agent_position = np.zeros(grid_shape)
        agent_position[self.agent_loc] = 1  # Mark the agent's position on the grid

        # Item positions
        item_positions = np.zeros(grid_shape)
        for loc in self.item_locs:
            item_positions[loc] = 1  # Mark item positions on the grid

        # Remaining times of items (normalized)
        remaining_times = np.zeros(grid_shape)
        for loc, time in zip(self.item_locs, self.item_times):
            remaining_times[loc] = (self.max_response_time - time) / self.max_response_time

        # TODO: evtl. weg weil eigentlich nicht relevant für die entscheidung
        '''# Past items (spawn counts at each position)
        items = np.zeros(grid_shape)
        for loc in self.item_locs:
            items[loc] += 1
        past_items = np.copy(items)'''

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


    def get_obs(self) -> List[float]:
        """Returns the state of the environment as a 1-dimensional list of floats."""
        grid_size = self.vertical_cell_count * self.horizontal_cell_count

        # Agent Position (One-Hot Encoded)
        agent_position = np.zeros(grid_size)
        agent_index = self.agent_loc[0] * self.horizontal_cell_count + self.agent_loc[1]
        agent_position[agent_index] = 1

        # Agent Position (2D Tuple)
        agent_position_2d = tuple(self.agent_loc)

        # Target Position (One-Hot Encoded)
        target_position = np.zeros(grid_size)
        target_index = self.target_loc[0] * self.horizontal_cell_count + self.target_loc[1]
        target_position[target_index] = 1

        # Target Position (2D Tuple)
        target_position_2d = tuple(self.target_loc)

        # Item Positions (One-Hot Encoded)
        item_positions = np.zeros(grid_size)
        for loc in self.item_locs:
            item_index = loc[0] * self.horizontal_cell_count + loc[1]
            item_positions[item_index] = 1

        # Item Times (Flat Vector)
        item_times = np.zeros(grid_size)
        for loc, time in zip(self.item_locs, self.item_times):
            item_index = loc[0] * self.horizontal_cell_count + loc[1]
            item_times[item_index] = (self.max_response_time - time)

        # Past Items Vector (Flat Vector)
        past_items_vector = self.past_items_vector.flatten()  # Flatten the spawn count grid

        # Free Capacity (Scalar)
        free_capacity = self.agent_capacity - self.agent_load

        return np.concatenate(
            [
                agent_position,         # Agent position (one-hot) | Dim: grid_size
                #agent_position_2d,
                target_position,        # Target position (one-hot) | Dim: grid_size
                #target_position_2d,
                item_positions,         # Item positions (one-hot) | Dim: grid_size
                item_times,             # Remaining times of items | Dim: grid_size
                past_items_vector,      # Past items spawn counts (flat vector) | Dim: grid_size
                [free_capacity],        # Free capacity (scalar) | Dim: 1
            ]
        )

    #CNN OBS!!!
    def get_cnn_obs(self) -> np.ndarray:
        """Returns the state of the environment as a 3D grid with multiple channels."""
        grid_shape = (self.vertical_cell_count, self.horizontal_cell_count)

        # Channel 1: Agent position
        agent_channel = np.zeros(grid_shape)
        agent_channel[self.agent_loc] = 1  # Mark the agent's position on the grid

        # Channel 2: Item positions
        item_positions_channel = np.zeros(grid_shape)
        for loc in self.item_locs:
            item_positions_channel[loc] = 1  # Mark item positions on the grid

        # Channel 3: Remaining times of items
        remaining_times_channel = np.zeros(grid_shape)
        for loc, time in zip(self.item_locs, self.item_times):
            remaining_times_channel[loc] = (self.max_response_time - time)

        # Channel 4: Past items (spawn counts at each position)
        past_items_channel = np.copy(self.past_items_vector)  # Use the past_items_vector to track spawn counts

        # Channel 5: Free capacity
        free_capacity_channel = np.full(grid_shape, self.agent_capacity - self.agent_load)

        # Channel 6: Target location
        target_channel = np.zeros(grid_shape)
        target_channel[self.target_loc] = 1  # Mark the target position on the grid

        # Stack all channels into a 3D grid
        state = np.stack(
            [agent_channel,
             item_positions_channel,
             #remaining_times_channel,
             past_items_channel, free_capacity_channel, target_channel],
            axis=-1,
        )
        return state

    def get_to_tgt(self, tgt):
        """Determines which action to take to get to tgt coordinates."""
        if self.agent_loc[0] > tgt[0]:  # target is above agent
            act = 1  # go up
        elif self.agent_loc[0] < tgt[0]:  # target is below agent
            act = 3  # go down
        elif self.agent_loc[1] > tgt[1]:  # target is to the left of agent
            act = 4  # go left
        elif self.agent_loc[1] < tgt[1]:  # target is to the right of agent
            act = 2  # go right
        else:  # could not identify target position
            act = 0  # do nothing

        return act

    def get_obs_greedy(self):
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

    def get_state_len(self):
        """Returns the length of one state i.e., the flattened array returned by the get_obs() method."""
        return 126

    '''def get_state_len(self):
        """Dynamically calculate the length of one state."""
        obs = self.get_obs()
        return len(obs)'''

    def get_action_len(self):
        """Returns the number of actions available."""
        return 5
