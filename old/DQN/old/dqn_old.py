import tensorflow as tf
from keras import layers
import numpy as np
import random
from environment import Environment
from collections import deque

from typing import List

from utils import utils as ut


class DQN_AGENT:
    def __init__(self, env: Environment,
                 gamma: float,
                 epsilon: float,
                 batch_size: int,
                 epsilon_min: float,
                 epsilon_decay: float,
                 learning_rate: float,
                 target_update_interval : int,
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
        self.actual_model = self._build_model()
        self.target_model = self._build_model()

        self.target_update_step = target_update_interval


    
    def _build_model(self):
        """Builds the Q-network."""
        model = tf.keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(self.action_size, activation="linear")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model
    
   
    def select_random_action(self, state):
        if np.random.uniform(0, 1) <= self.epsilon:
            act = random.randrange(self.action_size)
        else:
            act = self.select_greedy_action(state)
        return act
    
    def select_greedy_action(self, state):
        Q_values = self.actual_model.predict(state)
        act = np.argmax(Q_values[0])
        return act
    
    # Here newly observed transitions are stored in the experience replay buffer
    def record(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    

    def update_weights(self):
        # Checking if the replay buffer is large enough to be sampled
        if len(self.replay_buffer) >= self.batch_size:
            minibatch = random.sample(self.replay_buffer, self.batch_size)
            # Extracting the individual arguments
            current_states = np.array([transition[0] for transition in minibatch])
            actions = np.array([transition[1] for transition in minibatch])
            rewards = np.array([transition[2] for transition in minibatch])
            next_states = np.array([transition[3] for transition in minibatch])
            dones = np.array([transition[4] for transition in minibatch])

            # This reshaping is necessary for the nn input!
            current_states = np.squeeze(current_states, axis=1)
            next_states = np.squeeze(next_states, axis=1)

            # Implementing the minibatch/replay buffer update algorithm as per the lecture
            Q_target = self.actual_model.predict(current_states)
            Q_next = self.target_model.predict(next_states)
            max_Q_next = np.max(Q_next, axis=1)
            # Checking for terminal states with the "dones" flag
            targets = rewards + (1 - dones) * self.gamma * max_Q_next
            for i, action in enumerate(actions):
                Q_target[i][action] = targets[i]
                #Q_target[np.arange(self.batch_size), actions] = targets
            self.actual_model.fit(current_states, Q_target, epochs=1, verbose=0)

    # updates epsilon after each episode as long as it is larger than our defined minvalue
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Function to update weights of target network - check whether there is enough samples for the minibatch
    def update_target_weights(self):  # t for time step at which target model will be updated
        self.target_model.set_weights(self.actual_model.get_weights())
        
    def train_DQN(self, no_episodes: int) -> List[float]:
        train_rewards = deque(maxlen=no_episodes)

        for episode in range(no_episodes):
            #if (episode + 1) % 100 == 0:
                #inter_train_rew = np.mean(train_rewards[-100:])
                #print(f"--- Episode:  {str(episode + 1)} / {str(no_episodes)} --- Avg. Reward Last 100 Episodes: {inter_train_rew} --- Epsilon: {self.epsilon} ---" )

            current_state = self.env.reset("training")
            episode_rew = 0

            for step in range(self.env.episode_steps):
                # Choose and Perform Action
                #action = self._select_action(tf.convert_to_tensor(current_state))  # sample action
                action = self.select_random_action(state = current_state)
                rew, next_state, d = self.env.step(action)  # observe outcome of action
                #next_state = self.env.flatten_obs(next_state)  # flatten observation to 1d array

                # Store Observation in Replay Buffer
                # TODO: Differentiate between training and testing on abstract level
                #self.replay_buffer.append([current_state, action, rew, next_state, d])  # for NN update
                self.record(current_state, action, rew, next_state, d)
                episode_rew += rew  # increase total reward per episode

                if d:
                    break

                # Continue With Next State
                current_state = next_state

  
                self.update_weights()
                if step % self.target_update_step == 0:
                    self.update_target_weights()
                    print("Target Weights Updated")

                # Update Internal Variables
            self.update_epsilon()
            train_rewards.append(episode_rew)
            print(f"Episode {episode + 1}/{no_episodes}: Total Reward = {episode_rew}")

        return train_rewards


if __name__ == "__main__":
    # Initialize Environment
    env = Environment(variant = 0, data_dir = "drl-codinglab/data/")

    # Initialize DQN Agent with Hyperparameters
    # fmt: off
    agent = DQN_AGENT(env = env,
                      gamma = 0.95,
                      epsilon = 1,
                      batch_size = 128,
                      epsilon_min = 0.1,
                      epsilon_decay = 0.995,
                      learning_rate = 0.01,
                      target_update_interval = 100,
                      mem_maxlen = 2000)
    # fmt: on
    rewards = agent.train_DQN(no_episodes = 800)
    ut.rewards_plotting(rewards)
