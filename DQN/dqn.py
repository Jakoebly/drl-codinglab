import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import random
from datetime import datetime
from Utils import utils
from environment import Environment
from collections import deque

import os
from typing import List, Tuple
import pandas as pd

class DQN_AGENT:
    def __init__(self, env: Environment,
                 gamma: float,
                 epsilon: float,
                 batch_size: int,
                 epsilon_min: float,
                 epsilon_decay: float,
                 learning_rate: float,
                 target_update_interval : int,
                 eval_interval: int,
                 mem_maxlen: int):
        
        self.env = env
        self.state_size = self.env.get_state_len()
        self.action_size = self.env.get_action_len()

        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate

        self.replay_buffer = deque(maxlen=mem_maxlen)
        self.actual_model = self._nn_model(input_size = self.state_size, output_size=self.action_size, output_act_fct = "softmax")
        self.target_model = self._nn_model(input_size = self.state_size, output_size=self.action_size, output_act_fct = "softmax")

        # Other attributes
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval

    def _nn_model(self, input_size: int, output_size: int, output_act_fct: str, load_old_model = False):
        """Returns a NN with an input layer the size of `input_size`, 2 hidden layers with a RELU act. fct., and an output layer with size `output_size` and act. fct. `output_act_fct`."""
        model = keras.Sequential(
            [
                # Two hidden layers with a RELU act. fct. and 64 nodes each
                layers.Dense(128, activation="relu", input_shape=(input_size,)),  # was: 64
                layers.Dense(128, activation="relu"),  # was: 64
                # Output layer with custom size and act. fct.
                layers.Dense(output_size, activation = output_act_fct),
            ]
        )
        return model
    
    
    '''def nn_model(self, state_size, action_size, load_old_model):
        """Define the neural network."""
        inputs = tf.keras.Input(shape=(state_size,))
        prev_layer = inputs
        for no_neurons in self.nn_architecture:
            prev_layer = tf.keras.layers.Dense(no_neurons, activation='relu')(prev_layer)

        outputs = tf.keras.layers.Dense(action_size)(prev_layer)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr), loss = 'mse')

        if load_old_model:
            utils.load_model_weights(WEIGHT_PATH, model)
        return model'''
    
    '''def _select_action(self, state: tf.Tensor, mode: str = "training"):
        if mode == "training":
            return self.select_randon_action()
        else:
            return self.select_greedy_action(state)'''
    
    def select_random_action(self):
        if np.random.uniform(0, 1) <= self.epsilon:
            act = random.randrange(self.action_size)
        return act
    
    def select_greedy_action(self, state):
        Q_values = self.model.predict(state)
        act = np.argmax(Q_values[0])
        return act
    
    # Here newly observed transitions are stored in the experience replay buffer
    def record(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def _sample_observation(self) -> Tuple[tf.Tensor]:
        """Randomly samples an observation from the replay buffer. Returns `(current_states, actions, rew, next_states, d)`, all of them being tensors."""
        # Randomly Sample from Replay Buffer
        batch = random.choices(self.replay_buffer, k = self.batch_size)
        # Group Features and Convert Them to Tensors
        current_states = tf.convert_to_tensor([batch[i][0] for i in range(len(batch))])
        actions = tf.convert_to_tensor([batch[i][1] for i in range(len(batch))])
        rew = tf.convert_to_tensor([batch[i][2] for i in range(len(batch))])
        next_states = tf.convert_to_tensor([batch[i][3] for i in range(len(batch))])
        d = tf.convert_to_tensor([batch[i][4] for i in range(len(batch))])

        return (current_states, actions, rew, next_states, d)

    def update_actual_weights(self):
        # Checking if the replay buffer is large enough to be sampled
        if len(self.replay_buffer) >= self.batch_size:
            current_states, actions, rewards, next_states, dones = self._sample_observation
            # This reshaping is necessary for the nn input!
            current_states = np.squeeze(current_states, axis=1)
            next_states = np.squeeze(next_states, axis=1)

            # Implementing the minibatch/replay buffer update algorithm as per the lecture
            Q_target = self.model.predict(current_states)
            Q_next = self.target_model.predict(next_states)
            max_Q_next = np.max(Q_next, axis=1)
            # Checking for terminal states with the "dones" flag
            targets = rewards + (1 - dones) * self.gamma * max_Q_next
            for i, action in enumerate(actions):
                Q_target[i][action] = targets[i]
            self.actual_model.fit(current_states, Q_target, epochs=1, verbose=0)

        # Function to update weights of target network - check whether there is enough samples for the minibatch
    def update_target_weights(self):  # t for time step at which target model will be updated
        #if t % self.target_model_time == 0 and len(self.memory) >= self.batch_size:
        self.target_model.set_weights(self.actual_model.get_weights())
        print("------ TARGET WEIGHTS UPDATED --------")

    # updates epsilon after each episode as long as it is larger than our defined minvalue
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_DQN(self, no_episodes: int, mode, visualize_results: bool = True) -> List[float]:
        rewards = deque(maxlen = no_episodes)

        for episode in range(no_episodes):
            # Console Message
            if (episode + 1) % 100 == 0:
                print("Episode: " + str(episode + 1) + " / " + str(no_episodes))

            current_state = self.env.flatten_obs(self.env.reset(mode))  # get starting position
            episode_rew = 0

            for step in range(self.env.episode_steps):
                # Choose and Perform Action
                action = self.select_random_action()  # sample action
                rew, next_state, d = self.env.step(action)  # observe outcome of action
                next_state = self.env.flatten_obs(next_state)  # flatten observation to 1d array

                episode_rew += rew  # increase total reward per episode

                #self.replay_buffer.append([current_state, action, rew, next_state, d]) # Store Observation in Replay Buffer for NN update
                self.record(current_state, action, rew, next_state, d)
                self.update_actual_weights()  # Updating the weights using a replay buffer

                if step % self.target_update_interval == 0:
                    self.update_target_weights()  # Updating the weights of the target network every t episodes

                # Continue With Next State
                current_state = next_state

                if d:
                    break

            # Update Internal Variables
            self.update_epsilon()
            rewards.append(episode_rew)
            print(f"Step: {episode + 1}/{no_episodes} | Score: {episode_rew:.2f} | Epsilon: {self.epsilon:.3f}")

            # Function that evaluates the agent's performance for every x episodes
            if (episode + 1) % self.eval_interval == 0:
                eval_reward = self.test_DQN(no_episodes = 50, mode = "evaluation", visualize_results=False)
                average_reward = np.mean(eval_reward)
                print(f"Average Reward of last 100 runs after episode {episode + 1}: {average_reward:.2f}")

                # Checking for the training termination criteria
                if average_reward >= 2.5:
                    print(f"Training terminated after {episode + 1} episodes as the target average reward was achieved.")
                    # Adjusting reward array length in case of early termination (easier to handle for visualizations)
                    rewards = rewards[:episode + 1]
                    break   # Stop training if the target reward is reached

            print("episode: {}/{} | score: {} | e: {:.3f}".format(episode + 1, no_episodes, rew, agent.epsilon))

        return rewards, self.replay_buffer


    def test_DQN(self, no_episodes: int, mode, eval: bool = False, visualize_results: bool = True) -> List[float]:
        rewards = deque(maxlen = no_episodes)
        # TODO: Define which weights to load
        if not eval:
            self.actual_model.load_weights("./checkpoints/sac-actor.weights.h5")
            self.target_model.load_weights("./checkpoints/sac-actor.weights.h5")

        for episode in range(no_episodes):
            # Console Message
            if (episode + 1) % 100 == 0:
                print("Episode: " + str(episode + 1) + " / " + str(no_episodes))

            current_state = self.env.flatten_obs(self.env.reset(mode))  # get starting position
            episode_rew = 0

            for step in range(self.env.episode_steps):
                # Choose and Perform Action
                action = self.select_greedy_action(state = tf.convert_to_tensor(current_state))
                rew, next_state, d = self.env.step(action)  # observe outcome of action
                next_state = self.env.flatten_obs(next_state)  # flatten observation to 1d array
                episode_rew += rew  # increase total reward per episode
                current_state = next_state

                if d:
                    break

            rewards.append(episode_rew)
            print(f"Step: {episode + 1}/{no_episodes} | Score: {episode_rew:.2f} | Epsilon: {self.epsilon:.3f}")

        return rewards

    def run(self, no_episodes: int, mode: str = "training", visualize_results: bool = True) -> List[float]:
        # Console Message
        print(f"Starting {mode}...")

        if mode == "training":
            rewards, replay_buffer = self.train_DQN(no_episodes = no_episodes, visualize_results=visualize_results)

        else:
            rewards = self.test_DQN(no_episodes = no_episodes, visualize_results=visualize_results)

        return rewards


if __name__ == "__main__":
    # Initialize Environment
    env = Environment(variant=0, data_dir="./data")

    # Initialize SAC Agent with Hyperparameters
    # fmt: off
    agent = DQN_AGENT(env=env,
                      gamma=0.99,  # was: 0.99
                      initial_alpha=0.2,  # was: 1
                      alpha_decay=1,  # was: 0.995
                      rho=0.01,  # was: 0.2
                      mem_maxlen=int(1e7),  # was: int(1e7)
                      batch_size=64)  # was: 20
    # fmt: on
    rewards = agent.run(no_episodes=800, mode="training", visualize_results=True)
