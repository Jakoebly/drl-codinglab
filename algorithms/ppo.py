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
# - different nn_architecture for Actor and Critic and only one Network class
# - different learning_rate for Actor and Critic
# - adjust validate last act. function
# - if too slow, try to call memory.store only once after each trajectory and insert tensors
# - see how performance differs if actor and critic loss are separated
# - try out value loss clipping: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ point 9
# - vectorize implementation (i.e., collect data from multiple envs during rollout)
# - adjust hyperparams
# - set up all relevant parameters for wandb tracking
# - train step as separate function

# EXTENSIONS / ADJUSTMENTS
# - one total loss function (implemented)
# - advantage normalization (implemented)
# - value loss clipping (not implemented)
# - learning rate annealing (not implemented)
# - entropy coeffcient annealint (not implemented)
# - global gradient clipping (implemented)
# - early stopping (not implemented)


class PPOMemory:
    def __init__(self, batch_size, num_trajectories, num_steps_trajectory, state_size, device):

        self.states = torch.zeros((num_trajectories, num_steps_trajectory) + (state_size,)).to(device)
        self.actions = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)
        self.logprobs = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)
        self.values = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)
        self.rewards = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)
        self.next_states = torch.zeros((num_trajectories, num_steps_trajectory) + (state_size,)).to(device)
        self.dones = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)

        self.advantages = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)
        self.returns = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)


    def store(self, state, action, log_prob, value, reward, next_state, done, trajectory, step):
        self.states[trajectory][step] = state
        self.actions[trajectory][step] = action
        self.logprobs[trajectory][step] = log_prob
        self.values[trajectory][step] = value
        self.rewards[trajectory][step] = reward
        self.next_states[trajectory][step] = next_state
        self.dones[trajectory][step] = done


    def clear(self, num_trajectories, num_steps_trajectory, state_size, device):
        self.states = torch.zeros((num_trajectories, num_steps_trajectory) + (state_size,)).to(device)
        self.actions = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)
        self.logprobs = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)
        self.values = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)
        self.rewards = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)
        self.next_states = torch.zeros((num_trajectories, num_steps_trajectory) + (state_size,)).to(device)
        self.dones = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)

        self.advantages = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)
        self.returns = torch.zeros((num_trajectories, num_steps_trajectory)).to(device)

    def flatten(self, state_size):
        # Flatten data
        states_flat = self.states.reshape((-1,) + (state_size,))
        actions_flat = self.actions.reshape((-1,))
        logprobs_flat = self.logprobs.reshape((-1,))
        advantages_flat = self.advantages.reshape((-1))
        returns_flat = self.returns.reshape((-1,))
        values_flat = self.values.reshape((-1,))

        return states_flat, actions_flat, logprobs_flat, advantages_flat, returns_flat, values_flat

    def generate_batches(self, batch_size, minibatch_size):

        # Generate batch indices
        indices = np.arange(batch_size)
        batch_start_ind = np.arange(0, batch_size, minibatch_size)
        np.random.shuffle(indices)
        batches = [indices[i:i + minibatch_size] for i in batch_start_ind]

        return batches

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
            elif activation.lower() == "linear":
                pass
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
            elif activation.lower() == "linear":
                pass
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

        # Add last layer data separately since the for loop does not allow for this
        last_in_size, last_activation = nn_architecture[-1]
        layers.append(nn.Linear(last_in_size, last_in_size))

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
        self.num_minibatches = config["num_minibatches"]
        self.minibatch_size = self.batch_size // self.num_minibatches

        # Hyperparameters
        self.nn_architecture_actor = config["nn_architecture_actor"]
        self.nn_architecture_critic = config["nn_architecture_critic"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.lam = config["lam"]
        self.clip_ratio = config["clip_ratio"]
        self.entropy_coef = config["entropy_coef"]
        self.value_coef = config["value_coef"]
        self.entropy_coef_min = config["entropy_coef_min"]
        self.entropy_coef_decay = config["entropy_coef_decay"]
        self.max_grad_norm = config["max_grad_norm"]



        # Memory Initialization
        self.memory = PPOMemory(self.batch_size, self.num_trajectories, self.num_steps_trajectory, self.state_size, self.device)

        # Network Initialization
        self.actor = ActorNetwork(self.nn_architecture_actor, self.learning_rate, self.device)
        self.critic = CriticNetwork(self.nn_architecture_critic, self.learning_rate, self.device)

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

    def calc_advantage(self, values, next_states, rewards, dones, device):
        with torch.no_grad():
            # Initialize advantages and returns
            advantages = torch.zeros_like(values).to(device)
            returns = torch.zeros_like(values).to(device)

            # Last advantage accumulator
            advantage = torch.zeros(self.num_trajectories).to(device)


            # Iterate in reverse over time steps
            for t in reversed(range(self.num_steps_trajectory)):

                # Mask for terminal states
                mask = 1.0 - dones[:, t]
                if t == self.num_steps_trajectory - 1:
                    next_values = self.get_value(next_states[:, t])
                else:
                    next_values = values[:, t + 1]

                # Compute delta/TD-error
                delta = rewards[:, t] + self.gamma * next_values.view(-1) * mask - values[:, t]

                # Recursive GAE
                advantage = delta + self.gamma * self.lam * mask * advantage
                advantages[:, t] = advantage

                # Compute return
                returns[:, t] = advantage + values[:, t]

        return advantages, returns

    def run(self, mode, num_episodes):

        rewards = []

        if mode == "training":
            training = True
        else:
            training = False

        total_loss = 0
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0

        for episode in range(num_episodes):

            # Initialize reward to zero
            reward_episode = 0

            # Keep track of the number of trajectories per policy rollout
            trajectory = episode % self.num_trajectories

            state = torch.tensor(self.env.reset(mode), dtype=torch.float32).to(self.device)

            for step in range(self.num_steps_trajectory):

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

                state = next_state

                self.env.update_render()

            rewards.append(reward_episode)

            if training and trajectory == self.num_trajectories - 1:

                # Compute advantages and returns
                advantages, returns = self.calc_advantage(self.memory.values, self.memory.next_states, self.memory.rewards, self.memory.dones, self.device)
                self.memory.advantages = advantages
                self.memory.returns = returns

                # Flatten memory
                states, actions, old_logprobs, advantages, returns, values, = self.memory.flatten(self.state_size)

                for epoch in range(self.num_epochs):

                    # Generate batches
                    batches = self.memory.generate_batches(self.batch_size, self.minibatch_size)

                    for minibatch in batches:

                        # States, logprobs, action, returns for the current minibatch
                        states_mb = states[minibatch]
                        old_logprobs_mb = old_logprobs[minibatch]
                        actions_mb = actions[minibatch]
                        returns_mb = returns[minibatch]
                        values_mb = values[minibatch]

                        # Advantage normalization for the current minibatch
                        advantages_mb = advantages[minibatch]
                        advantages_norm_mb = (advantages_mb - advantages_mb.mean()) / (advantages_mb.std() + 1e-8)


                        # Compute prob_ratio
                        dist = self.actor(states_mb)
                        new_logprobs_mb = dist.log_prob(actions_mb)
                        prob_ratio = (new_logprobs_mb - old_logprobs_mb).exp()
                        clipped_prob_ratio = torch.clamp(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

                        # Compute entropy loss
                        entropy = dist.entropy()
                        entropy_loss = entropy.mean()

                        # Compute PPO objective function
                        actor_loss = torch.max(-advantages_norm_mb * prob_ratio, -advantages_norm_mb * clipped_prob_ratio).mean()

                        # Compute value loss
                        critic_values = self.critic(states_mb)
                        critic_loss = ((critic_values - returns_mb) ** 2).mean()

                        # Compute total loss
                        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_loss

                        # Perform updates
                        self.actor.optimizer.zero_grad()
                        self.critic.optimizer.zero_grad()
                        total_loss.backward()
                        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                        self.actor.optimizer.step()
                        self.critic.optimizer.step()

                y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

                self.wandb_run.log({"Losses/Explained Var": explained_var}, episode)

                self.memory.clear(self.num_trajectories, self.num_steps_trajectory, self.state_size, self.device)



            if self.wandb_run is not None:
                self.wandb_run.log({"Charts/Reward per Episode": reward_episode}, episode)

                self.wandb_run.log({"Losses/Total Loss": total_loss}, episode)
                self.wandb_run.log({"Losses/Actor Loss": actor_loss}, episode)
                self.wandb_run.log({"Losses/Critic Loss": critic_loss}, episode)
                self.wandb_run.log({"Losses/Entropy": entropy_loss}, episode)



            print(f"EPISODE: {episode + 1} / {num_episodes}, Total Reward: {reward_episode}")

        return rewards







