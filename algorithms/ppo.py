import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


from typing import List, Tuple
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from keras import layers, optimizers
from environment import Environment
import matplotlib.pyplot as plt

# TODO:
# - inform what is meant with .to(device) and eventually adjust in code
# - different nn_architecture for Actor and Critic
# - different learning_rate for Actor and Critic

class PPOMemory:
    def __init__(self, batch_size, num_trajectories, num_steps_trajectory, state_size, num_actions, device):
        # Maybe switch shapes to (num_trajectories, num_steps_trajectory)
        self.states = torch.zeros((num_steps_trajectory, num_trajectories) + (state_size,)).to(device)
        self.actions = torch.zeros((num_steps_trajectory, num_trajectories) + (num_actions,)).to(device)
        self.logprobs = torch.zeros((num_steps_trajectory, num_trajectories)).to(device)
        self.rewards = torch.zeros((num_steps_trajectory, num_trajectories)).to(device)
        self.dones = torch.zeros((num_steps_trajectory, num_trajectories)).to(device)
        self.values = torch.zeros((num_steps_trajectory, num_trajectories)).to(device)

        self.next_states = []
        self.nextvalues = []
        self.advantages = []
        self.returns = []

        self.batch_size = batch_size

    def store(self, state, action, reward, next_state, done, value, next_value, log_probs):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.values.append(value)
        self.nextvalues.append(next_value)
        self.log_probs.append(log_probs)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.values.clear()
        self.nextvalues.clear()
        self.log_probs.clear()
        self.advantages.clear()
        self.returns.clear()

    def to_tensor(self):
        return {
            'states': tf.convert_to_tensor(self.states, dtype=tf.float32),
            'actions': tf.convert_to_tensor(self.actions, dtype=tf.int32),
            'rewards': tf.convert_to_tensor(self.rewards, dtype=tf.float32),
            'next_states': tf.convert_to_tensor(self.next_states, dtype=tf.float32),
            'dones': tf.convert_to_tensor(self.dones, dtype=tf.float32),
            'values': tf.convert_to_tensor(self.values, dtype=tf.float32),
            'next_values': tf.convert_to_tensor(self.nextvalues, dtype=tf.float32),
            'log_probs': tf.convert_to_tensor(self.log_probs, dtype=tf.float32),
            'advantages': tf.convert_to_tensor(self.advantages, dtype=tf.float32),
            'returns': tf.convert_to_tensor(self.returns, dtype=tf.float32)
        }

    def generate_batches(self, batch_size, minibatch_size, num_epochs):

        data = self.to_tensor()

        dataset_actor = tf.data.Dataset.from_tensor_slices({
            'states': data['states'],
            'actions': data['actions'],
            'log_probs': data['log_probs'],
            'advantages': data['advantages']
        })

        dataset_critic = tf.data.Dataset.from_tensor_slices({
            'states': data['states'],
            'returns': data['returns']
        })

        dataset_actor_shuffled = dataset_actor.shuffle(buffer_size=batch_size, reshuffle_each_iteration=True)
        dataset_actor_shuffled_batched = dataset_actor_shuffled.batch(minibatch_size).repeat(num_epochs)

        dataset_critic_shuffled = dataset_critic.shuffle(buffer_size=batch_size, reshuffle_each_iteration=True)
        dataset_critic_shuffled_batched = dataset_critic_shuffled.batch(minibatch_size).repeat(num_epochs)

        return dataset_actor_shuffled_batched, dataset_critic_shuffled_batched

class ActorNetwork(nn.Module):
    def __init__(self, nn_architecture, learning_rate, device):
        super().__init__()

        # Initialize an empty list to fill with layer info
        layers = []

        # Iteratively fill layers with layer info from the nn_architecture parameter
        for i in range(len(nn_architecture) - 1):

            # Load layer size and activation function
            in_size, activation = nn_architecture[i]
            out_size, _ = nn_architecture[i + 1]

            # Add layer to list layers using nn.Linear
            layers.append(nn.Linear(in_size, out_size))

            # Add activation function to list layers
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

        # Add last layer data separately since the for loop does not allow for this
        last_in_size, last_activation = nn_architecture[-1]
        layers.append(nn.Linear(last_in_size, last_in_size))

        if last_activation.lower() == "softmax":
            layers.append(nn.Softmax(dim=-1))
        elif last_activation.lower() == "sigmoid":
            layers.append(nn.Sigmoid())
        elif last_activation.lower() == "linear" or last_activation.lower() is None:
            pass
        else:
            raise ValueError(f"Unsupported activation function: {last_activation}")

        # Create network using the layer info
        self.actor = nn.Sequential(*layers).to(device)

        # Set optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Move model to device
        self.to(device)

    def forward(self, state):

        # Output softmax probs given a state
        dist = self.actor(state)

        # Create a Categorical distribution from the output
        dist = Categorical(dist)

        return dist


class CriticNetwork(nn.Module):
    def __init__(self, nn_architecture, learning_rate, device):
        super().__init__()

        # Initialize an empty list to fill with layer info
        layers = []

        # Iteratively fill layers with layer info from the nn_architecture parameter
        for i in range(len(nn_architecture) - 1):

            # Load layer size and activation function
            in_size, activation = nn_architecture[i]
            out_size, _ = nn_architecture[i + 1]

            # Add layer to list layers using nn.Linear
            layers.append(nn.Linear(in_size, out_size))

            # Add activation function to list layers
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

        # Add last layer data separately since the for loop does not allow for this
        last_in_size, last_activation = nn_architecture[-1]
        layers.append(nn.Linear(last_in_size, last_in_size))

        if last_activation.lower() == "softmax":
            layers.append(nn.Softmax(dim=-1))
        elif last_activation.lower() == "sigmoid":
            layers.append(nn.Sigmoid())
        elif last_activation.lower() == "linear" or last_activation.lower() is None:
            pass
        else:
            raise ValueError(f"Unsupported activation function: {last_activation}")

        # Create network using the layer info
        self.critic = nn.Sequential(*layers)

        # Set optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Move model to device
        self.to(device)

    def forward(self, state):

        # Output value given a state
        value = self.critic(state)

        return value


class PPO_Agent():
    def __init__(self, wandb_run, env, config, device):

        self.device = device
        self.wandb_run = wandb_run
        self.env = env

        # General hyperparams
        self.state_size = config["state_size"]
        self.num_actions = config["num_actions"]
        self.num_episodes_train = config["num_episodes_train"]
        self.num_episodes_validate = config["num_episodes_validate"]
        self.num_episodes_test = config["num_episodes_test"]
        self.num_trajectories = config["num_trajectories"]
        self.num_steps_trajectory = config["num_steps_trajectory"]
        self.num_epochs = config["num_epochs"]
        self.batch_size = self.num_trajectories * self.num_steps_trajectory
        self.minibatch_size = self.batch_size // config["num_minibatches"]

        # Hyperparameters
        self.nn_architecture = config["nn_architecture"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.lam = config["lam"]
        self.clip_ratio = config["clip_ratio"]
        self.entropy_bonus = config["entropy_bonus"]
        self.entropy_bonus_min = config["entropy_bonus_min"]
        self.entropy_bonus_decay = config["entropy_bonus_decay"]

        # Memory Initialization
        self.memory = PPOMemory(self.batch_size, self.num_trajectories, self.num_steps_trajectory, self.state_size, self.device)

        # Network Initialization
        self.actor = ActorNetwork(self.nn_architecture, self.learning_rate, self.device)
        self.critic = CriticNetwork(self.nn_architecture, self.learning_rate, self.device)

    def load_weights(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

    def save_weights(self, path):
        torch.save(self.actor.state_dict(), path)

    def get_action(self, state):

        probs = self.actor.predict(state.reshape(1, -1))
        print(f"Probability Output: {probs}")
        log_probs = tf.math.log(probs + 1e-8)
        # action = np.random.choice(self.num_actions, p=probs[0])
        action = np.argmax(probs)
        print(f"Selected Action: {action}")
        return action, log_probs[0, action]

    # TODO: überarbeiten
    def get_value(self, state):
        value = self.critic.predict(state.reshape(1, -1))
        return value[0][0]


    def run(self, mode, validate_after_training):

        rew_train = None
        rew_val = None
        rew_test = None

        if mode == "train":
            rew_train = self.simulate_environment(self.num_episodes_train, training=True, validation=False, testing=False)
            if validate_after_training:
                rew_val = self.simulate_environment(self.num_episodes_validate, training=False, validation=True, testing=False)
        elif mode == "validate":
            rew_val = self.simulate_environment(self.num_episodes_validate, training=False, validation=True, testing=False)
        elif mode == "test":
            rew_test = self.simulate_environment(self.num_episodes_test, training=False, validation=False, testing=False)

        return rew_train, rew_val, rew_test


    def simulate_environment(self, num_episodes, training=True, validation=False, testing=False):

        rewards = []

        for episode in range(5):

            if training:
                state = self.env.reset("training")
            if validation:
                state = self.env.reset("validation")
            if testing:
                state = self.env.reset("testing")

            reward_episode = 0
            done = False

            while not done:
                action = self.select_action(state, training)
                reward, next_state, done = self.env.step(action)

                if training:
                    self.record(state, action, reward, next_state, done)
                    self.train_step()

                state = next_state
                reward_episode += reward

                # if render:
                # self.env.render()  # Call the render method

            rewards.append(reward_episode)

            if training:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            self.wandb_run.log({"Reward per Episode": reward_episode})
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {reward_episode}")

            # TODO: Raus?
            '''if (episode + 1) % self.eval_interval == 0:
                eval_reward = self.test(episodes=self.evaluation_episodes, model=self.q_network, eval=True)
                average_eval_reward = np.mean(eval_reward)
                print(f"Average Evaluation Reward: {average_eval_reward:.2f}")

                if average_eval_reward >= self.evaluation_treshold:
                    print(
                        f"Training terminated after {episode + 1} episodes as the target average reward was achieved.")
                    rewards = rewards[:episode + 1]
                    break
                '''

        return rewards


    num_updates = self.num_episodes_train // self.num_trajectories




    state = torch.Tensor(self.env.reset(self.mode))



