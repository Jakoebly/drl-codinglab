# actions: 0 (nothing), 1 (up), 2 (right), 3 (down), 4 (left)

# positions in grid:
# - (0,0) is upper left corner
# - first index is vertical (increasing from top to bottom)
# - second index is horizontal (increasing from left to right)

# if new item appears in a cell into which the agent moves/at which the agent stays in the same time step,
# it is not picked up (if agent wants to pick it up, it has to stay in the cell in the next time step)

import random
import pandas as pd
from copy import deepcopy
from itertools import compress
import numpy as np

from Renderer.renderer import Renderer


class Environment(object):
    def __init__(self, variant, data_dir):
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
        self.renderer = Renderer(
            grid_size=(self.vertical_cell_count, self.horizontal_cell_count),
            eligible_cells=self.eligible_cells,
            max_response_time = self.max_response_time
        )
        self.target_count = 0 # Counts number of items dropped off at target cell

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

        # get new observation
        next_obs = self.get_obs()

        return rew, next_obs, done
    
    def render(self):
        """
        Call the renderer to visualize the environment.
        """
        if self.renderer:
            self.renderer.render(
                agent_loc=self.agent_loc,
                target_loc=self.target_loc,
                item_locs=self.item_locs,
                item_times=self.item_times,
                agent_load=self.agent_load,
                target_count=self.target_count
            )
        else:
            print("Renderer not initialized.")

    # DONE: implement function that gives the input features for the neural network(s)
    #       based on the current state of the environment
    """Returns current position of agent, free carry capacity of agent, and zero-padded item locations and times (24 total). This leads to a required 2 + 1 + 24 * (2 + 1) = 75 input neurons for a NN."""

    #TODO: PADDING
    #TODO: Item Position RELATIV not absolut
    #TODO: Manhatten distance to each item (new list)
    #TODO: Manhatten distance from item to target
    #TODO: Number of items present on field
    #TODO: Abstand zur nächsten wand
    #TODO: Past Items Vector
    #TODO: Delte items where distance > remaining time

    #TODO: DNN, CNN, RNN

    # This was our old get_obs()
    '''def get_obs(self):
        agent_pos = self.agent_loc
        free_capacity = self.agent_capacity - self.agent_load
        item_locs = list.copy(self.item_locs)
        item_times = [(self.max_response_time - i) for i in list.copy(self.item_times)]  # remaining time
        #item_distances = list.copy(self.item_distances)

        # Zero Padding for fixed number of neurons => item dummys
        while len(item_locs) < 24:
            item_locs.append((-1, -1))  # non-existent position => to mark dummy
            item_times.append(0)  # no remaining time => to mark dummy

        return [agent_pos, free_capacity, item_locs, item_times]'''

    # New get_obs()
    def get_obs(self):
        agent_pos = np.array(self.agent_loc) / np.array([self.vertical_cell_count, self.horizontal_cell_count])  # Normalize position
        free_capacity = np.array([self.agent_capacity - self.agent_load])  # Free capacity as a scalar

        item_locs = (
            np.array(self.item_locs) / np.array([self.vertical_cell_count, self.horizontal_cell_count])
            if self.item_locs else np.empty((0, 2))
        )
        item_times = np.array([self.max_response_time - i for i in self.item_times])  # Remaining times
        item_distances = np.linalg.norm(item_locs - np.array(self.agent_loc) / np.array([self.vertical_cell_count, self.horizontal_cell_count]), axis=1) if len(item_locs) > 0 else np.array([])
        target_distances = np.linalg.norm(item_locs - np.array(self.target_loc) / np.array([self.vertical_cell_count, self.horizontal_cell_count]), axis=1) if len(item_locs) > 0 else np.array([])

        max_items = 24  # Maximum number of items
        padding_value = -1  # Explicit padding value
        padded_item_locs = np.pad(item_locs, ((0, max_items - len(item_locs)), (0, 0)), constant_values=padding_value)
        padded_item_times = np.pad(item_times, (0, max_items - len(item_times)), constant_values=padding_value)
        padded_item_distances = np.pad(item_distances, (0, max_items - len(item_distances)), constant_values=padding_value)
        padded_target_distances = np.pad(target_distances, (0, max_items - len(target_distances)), constant_values=padding_value)

        valid_mask = np.array([1] * len(item_locs) + [0] * (max_items - len(item_locs)))

        # NO NEED FOR FLATTEN OBS - Check if this is correct
        observation = np.concatenate([
            agent_pos.flatten(),  # Flattened agent position
            free_capacity,  # Free capacity
            padded_item_locs.flatten(),  # Flattened item locations
            padded_item_times,  # Item times
            padded_item_distances,  # Distances to agent
            padded_target_distances,  # Distances to target
            valid_mask  # Validity mask
        ])
        
        return observation

    # TODO: Delete or intergate into get_obs()
    '''def flatten_obs(self, obs):
        """Flattens the list returned from the get_obs() method."""
        obs[1] = [obs[1]]
        obs[2] = np.array(obs[2]).flatten()
        return [elem for row in obs for elem in row]'''

    '''def get_state_len(self):
        """Returns the length of one state i.e., the tupel that is returned by the get_obs() method."""
        return 2 + 1 + 24 * (2 + 1)'''
    
    def get_state_len(self):
        """Returns the length of one state i.e., the flattened array returned by the get_obs() method."""
        # Agent position (2), free capacity (1)
        agent_features = 2 + 1
        
        # Items (24 max): locations (2 per item), times (1 per item), distances (2 per item)
        item_features = 24 * (2 + 1 + 1 + 1)  # locations + times + distances to agent + distances to target
        
        # Validity mask (24 binary values)
        mask_features = 24

        return agent_features + item_features + mask_features


    def get_action_len(self):
        """Returns the number of actions available."""
        return 5
