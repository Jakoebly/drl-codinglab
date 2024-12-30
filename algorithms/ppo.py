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
# - adjust validate last act. function
# - if too slow, try to call memory.store only once after each trajectory and insert tensors

class PPOMemory:
    def __init__(self, batch_size, num_trajectories, num_steps_trajectory, state_size, device):

        self.states = torch.zeros((num_trajectories, num_steps_trajectory) + (state_size,)).to(device)
        self.actions = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)
        self.logprobs = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)
        self.values = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)
        self.rewards = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)
        self.next_states = torch.zeros((num_trajectories, num_steps_trajectory) + (state_size,)).to(device)
        self.dones = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)

        self.nextvalues = []
        self.advantages = []
        self.returns = []

        self.batch_size = batch_size


    def store(self, state, action, log_prob, value, reward, next_state, done, trajectory, step):
        self.states[trajectory][step] = state
        self.actions[trajectory][step] = action
        self.logprobs[trajectory][step] = log_prob
        self.values[trajectory][step] = value
        self.rewards[trajectory][step] = reward
        self.next_states[trajectory][step] = next_state
        self.dones[trajectory][step] = done


    def clear(self, num_trajectories, num_steps_trajectory, state_size):
        self.states = torch.zeros((num_trajectories, num_steps_trajectory) + (state_size,))
        self.actions = torch.zeros((num_trajectories, num_steps_trajectory))
        self.logprobs = torch.zeros((num_trajectories, num_steps_trajectory))
        self.values = torch.zeros((num_trajectories, num_steps_trajectory))
        self.rewards = torch.zeros((num_trajectories, num_steps_trajectory))
        self.next_states = torch.zeros((num_trajectories, num_steps_trajectory) + (state_size,))
        self.dones = torch.zeros((num_trajectories, num_steps_trajectory))


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
        layers.append(nn.Linear(last_in_size, 1))

        # if last_activation.lower() == "softmax":
        #     layers.append(nn.Softmax(dim=-1))
        # elif last_activation.lower() == "sigmoid":
        #     layers.append(nn.Sigmoid())
        # elif last_activation.lower() == "linear" or last_activation.lower() is None:
        #     pass
        # else:
        #     raise ValueError(f"Unsupported activation function: {last_activation}")

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
        self.memory = PPOMemory(self.batch_size, self.num_trajectories, self.num_steps_trajectory, self.state_size, self.num_actions, self.device)

        # Network Initialization
        self.actor = ActorNetwork(self.nn_architecture, self.learning_rate, self.device)
        self.critic = CriticNetwork(self.nn_architecture, self.learning_rate, self.device)

    def load_weights(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

    def save_weights(self, path):
        torch.save(self.actor.state_dict(), path)

    def get_action(self, state):

        dist = self.actor(state)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob


    def get_value(self, state):

        value = self.critic(state)

        return value


    def run(self, mode):

        rew_train = None
        rew_val = None
        rew_test = None

        if mode == "train":
            rew_train = self.simulate_environment(self.num_episodes_train, "training")
            rew_val = self.simulate_environment(self.num_episodes_validate, "validation")
        if mode == "validate":
            rew_val = self.simulate_environment(self.num_episodes_validate, "validation")
        elif mode == "test":
            rew_test = self.simulate_environment(self.num_episodes_test, "testing")

        return rew_train, rew_val, rew_test

    def simulate_environment(self, num_episodes, mode):

        rewards = []

        if mode == "training":
            training = True
        else:
            training = False

        # for episode in range(num_episodes):
        for episode in range(1):

            # Initialize reward to zero
            reward_episode = 0

            # Keep track of the number of trajectories per policy rollout
            trajectory = episode % (self.num_trajectories + 1)

            state = torch.tensor(self.env.reset(mode), dtype=torch.float32).to(self.device)

            # for step in range(self.num_steps_trajectory):
            for step in range(1):

                # Perform one environment step
                with torch.no_grad():
                    action, log_prob = self.get_action(state)
                    value = self.get_value(state).squeeze()
                reward, next_state, done = self.env.step(action)
                reward_episode += reward

                # Transform to tensors
                reward = torch.tensor([reward]).to(self.device)
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                done = torch.tensor([done]).to(self.device)

                if training:
                    self.memory.store(state, action, log_prob, value, reward, next_state, done, trajectory, step)
                    #self.train_step()

                state = next_state

                """    
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
"""
        return rewards


    # num_updates = self.num_episodes_train // self.num_trajectories







