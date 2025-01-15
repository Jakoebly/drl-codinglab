import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.distributions.categorical import Categorical
from environment import Environment
from torch.utils.data import Dataset, DataLoader

from utils import neural_nets as nnets

# TODO:
# - vectorize implementation (i.e., collect data from multiple envs during rollout)

# DEBUG IDEAS
# - Self Imitation Learning for exploration
# - Adversarially Guided Actor-Critic (AGAC)
# - Reward Normalization
# - value loss clipping (not implemented)
# - learning rate annealing (not implemented)
# - entropy coeffcient annealing (not implemented)


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


class BuildNetwork(nn.Module):
    def __init__(self, nn_architecture, input_size, output_size, learning_rate, device):
        super().__init__()

        # Initialize an empty list to fill with layer info
        layers = []

        # Add input layer using nn.Linear
        in_size = input_size

        # Iteratively add hidden layers with layer info from the nn_architecture parameter
        for i in range(len(nn_architecture)):

            out_size, activation = nn_architecture[i]

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

            # Update input/output size and activation function
            in_size, _ = nn_architecture[i]


        # Add output layer using nn.Linear
        last_in_size, last_activation = nn_architecture[-1]
        layers.append(nn.Linear(last_in_size, output_size))

        # Create network using the layer info
        self.model = nn.Sequential(*layers)

        # Set optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Move model to device
        self.to(device)

    def forward(self, state):

        # Output unactivated outputs given a state
        output = self.model(state)

        return output

class PPO_Agent():
    def __init__(self, env, config, device):
        self.device = device
        self.env = env

        # Unpack config
        general = config["general"]
        nn_architecture_actor = config["nn_architecture_actor"]
        nn_architecture_critic = config["nn_architecture_critic"]
        hyperparameters = config["hyperparameters"]
        eval_settings = config["eval_settings"]

        # General
        self.state_size = general["state_size"]
        self.action_size = general["action_size"]
        self.num_trajectories = hyperparameters["num_trajectories"]
        self.num_steps_trajectory = general["num_steps_trajectory"]
        self.num_epochs = hyperparameters["num_epochs"]
        self.batch_size = self.num_trajectories * self.num_steps_trajectory
        self.num_minibatches = hyperparameters["num_minibatches"]
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.num_updates = general["num_episodes_train"] // self.num_trajectories

        # NN-Architecture
        self.nn_architecture_actor = nn_architecture_actor
        self.nn_structure_actor = nn_architecture_actor["structure"]
        self.nn_architecture_critic = nn_architecture_critic
        self.nn_structure_critic = nn_architecture_critic["structure"]

        # Hyperparameters
        self.learning_rate_actor = hyperparameters["learning_rate_actor"]
        self.learning_rate_critic = hyperparameters["learning_rate_critic"]
        self.gamma = hyperparameters["gamma"]
        self.lam = hyperparameters["lam"]
        self.clip_ratio = hyperparameters["clip_ratio"]
        self.value_coef = hyperparameters["value_coef"]
        self.entropy_coef = hyperparameters["entropy_coef"]
        self.max_grad_norm = hyperparameters["max_grad_norm"]
        self.target_kl = hyperparameters["target_kl"]
        self.warm_up_period = hyperparameters["warm_up_period"]

        # Memory Initialization
        self.memory = PPOMemory(self.batch_size, self.num_trajectories, self.num_steps_trajectory, self.state_size, self.device)

        # Network Initialization
        self.actor = nnets.build_network(self.nn_architecture_actor, self.state_size, self.action_size, self.learning_rate_actor).to(self.device)
        self.critic = nnets.build_network(self.nn_architecture_critic, self.state_size, 1, self.learning_rate_critic).to(self.device)


    def load_weights(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

    def save_weights(self, path):
        torch.save(self.actor.state_dict(), path)

    def update_learning_rate(self, update_ctr):

        learning_rate_decay = 1.0 - (update_ctr / (self.num_updates - 1))
        lr_actor_new = learning_rate_decay * self.learning_rate_actor
        lr_critic_new = learning_rate_decay * self.learning_rate_critic
        self.actor.optimizer.param_groups[0]['lr'] = lr_actor_new
        self.critic.optimizer.param_groups[0]['lr'] = lr_critic_new

    def update_entropy_coef(self, update_ctr):

        entropy_coef_decay = 1.0 - (update_ctr / (self.num_updates - 1))
        entropy_coef_new = entropy_coef_decay * self.entropy_coef
        self.entropy_coef = entropy_coef_new

    def get_action(self, state):

        # Output logits given a state
        logits = self.actor(state)

        # Create a Categorical distribution from the output
        dist = Categorical(logits=logits)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob

    def get_value(self, state):

        # Output value give a state
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
                    next_values = self.get_value(next_states[:, t]).view(-1)
                else:
                    next_values = values[:, t + 1]

                # Compute delta/TD-error
                delta = rewards[:, t] + self.gamma * next_values * mask - values[:, t]

                # Recursive GAE
                advantage = delta + self.gamma * self.lam * mask * advantage
                advantages[:, t] = advantage

                # Compute return
                returns[:, t] = advantage + values[:, t]

        return advantages, returns

    def compute_loss(self, states_mb, values_mb, actions_mb, old_logprobs_mb, advantages_norm_mb, returns_mb):

        # Compute prob_ratio
        logits = self.actor(states_mb)

        dist = Categorical(logits=logits)

        new_logprobs_mb = dist.log_prob(actions_mb)
        log_ratio = new_logprobs_mb - old_logprobs_mb
        prob_ratio = log_ratio.exp()
        clipped_prob_ratio = torch.clamp(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

        # Compute entropy loss
        entropy = dist.entropy()
        entropy_loss = entropy.mean()

        # Compute PPO objective function
        actor_loss = -(torch.min(advantages_norm_mb * prob_ratio, advantages_norm_mb * clipped_prob_ratio).mean() + self.entropy_coef * entropy_loss)

        # Compute critic loss
        critic_values = self.critic(states_mb)
        critic_loss_unclipped = (critic_values - returns_mb) ** 2
        values_clipped = values_mb + torch.clamp(critic_values - values_mb, -self.clip_ratio, self.clip_ratio)
        critic_loss_clipped = (values_clipped - returns_mb) ** 2
        critic_loss_max = torch.max(critic_loss_unclipped, critic_loss_clipped)
        critic_loss = 0.5 * critic_loss_max.mean()
        critic_loss = 0.5 * ((critic_values - returns_mb) ** 2).mean()

        # Compute total loss
        total_loss = actor_loss + self.value_coef * critic_loss

        # Compute approximate kl-divergence (http://joschu.net/blog/kl-approx.html)
        with torch.no_grad():
            approx_kl = ((prob_ratio - 1) - log_ratio).mean()

        return total_loss, actor_loss, critic_loss, entropy_loss, approx_kl

    def update_networks(self):

        # Compute advantages and returns
        advantages, returns = self.calc_advantage(self.memory.values, self.memory.next_states, self.memory.rewards, self.memory.dones, self.device)
        self.memory.advantages = advantages
        self.memory.returns = returns

        # Flatten memory
        states, actions, old_logprobs, advantages, returns, values, = self.memory.flatten(self.state_size)

        for epoch in range(self.num_epochs):

            # Generate batches
            batches = self.memory.generate_batches(self.batch_size, self.minibatch_size)

            minibatch_ctr = 0

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

                total_loss, actor_loss, critic_loss, entropy_loss, approx_kl = self.compute_loss(states_mb, values_mb, actions_mb, old_logprobs_mb, advantages_norm_mb, returns_mb)

                # Perform updates
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

            # if approx_kl > self.target_kl:
            #     print("exiting after epoch ", epoch, " out of ", self.num_epochs, " epochs")
            #     break

        return total_loss, actor_loss, critic_loss, entropy_loss

    def run(self, mode, num_episodes):

        rewards = []

        if mode == "training":
            training = True
        else:
            training = False

        total_loss, actor_loss, critic_loss, entropy_loss = 0, 0, 0, 0
        update_ctr = 0

        for episode in range(num_episodes):

            # Initialize reward to zero
            reward_episode = 0

            # Keep track of the number of trajectories per policy rollout
            trajectory = episode % self.num_trajectories

            # Reset environment
            state = torch.tensor(self.env.reset(mode), dtype=torch.float32).to(self.device)

            # Perform one trajectory
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

                # Keep track of rollout data in case of training
                if training:
                    self.memory.store(state, action, log_prob, value, reward, next_state, done, trajectory, step)
                # Update state for next step
                state = next_state

            # Append episodic reward
            rewards.append(reward_episode)
            average_reward = np.mean(rewards)

            print("Episode: ", episode, "Reward: ", reward_episode)

            # Perform update step with rollout data and clear memory afterwards for new rollout phase
            if training and trajectory == self.num_trajectories - 1:

               # if episode > self.warm_up_period:
               #     self.update_learning_rate(update_ctr)
               #     self.update_entropy_coef(update_ctr)
               #     update_ctr += 1

                total_loss, actor_loss, critic_loss, entropy_loss = self.update_networks()

                self.memory.clear(self.num_trajectories, self.num_steps_trajectory, self.state_size, self.device)

            # Log relevant data
            wandb.log({"Charts/Average Reward": average_reward}, episode)
            wandb.log({"Charts/Reward per Episode": reward_episode}, episode)
            wandb.log({"Losses/Total Loss": total_loss}, episode)
            wandb.log({"Losses/Actor Loss": actor_loss}, episode)
            wandb.log({"Losses/Critic Loss": critic_loss}, episode)
            wandb.log({"Losses/Entropy": entropy_loss}, episode)
            wandb.log({"Parameters/Actor Learning Rate": self.actor.optimizer.param_groups[0]['lr']}, episode)
            wandb.log({"Parameters/Critic Learning Rate": self.critic.optimizer.param_groups[0]['lr']}, episode)
            wandb.log({"Parameters/Entropy Coef": self.entropy_coef}, episode)

        print("Avg. Rewards: ", np.mean(rewards))

        return rewards
