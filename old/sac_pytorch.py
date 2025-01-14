import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from environment import Environment
from torch.utils.data import Dataset, DataLoader



class SACMemory:
    def __init__(self, mem_maxlen, state_size, num_actions):

        self.buffer_size = mem_maxlen
        self.current_buffer_size = 0

        self.states = np.zeros((self.buffer_size, state_size))
        self.actions = np.zeros(self.buffer_size)
        self.rewards = np.zeros(self.buffer_size)
        self.next_states = np.zeros((self.buffer_size, state_size))
        self.dones = np.zeros(self.buffer_size)


    def store(self, state, action, reward, next_state, done):
        index = self.current_buffer_size % self.buffer_size

        self.states[index] = np.array(state, dtype=np.float32)
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = np.array(next_state, dtype=np.float32)
        self.dones[index] = done

        self.current_buffer_size += 1

    def sample(self, batch_size):

        size = min(self.current_buffer_size, self.buffer_size)

        indices = np.random.choice(size, batch_size)

        states = torch.tensor(self.states[indices], dtype=torch.float32)
        actions = torch.tensor(self.actions[indices])
        rewards = torch.tensor(self.rewards[indices])
        next_states = torch.tensor(self.next_states[indices], dtype=torch.float32)
        dones = torch.tensor(self.dones[indices])

        return states, actions, rewards, next_states, dones

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

class SAC_Agent():
    def __init__(self, env, config, device):

        self.device = device
        self.env = env

        # General
        self.state_size = config["state_size"]
        self.num_actions = config["num_actions"]
        self.mem_maxlen = config["mem_maxlen"]
        self.num_episodes_train = config["num_episodes_train"]
        self.num_episodes_validate = config["num_episodes_validate"]
        self.num_episodes_test = config["num_episodes_test"]
        self.num_trajectories = config["num_trajectories"]
        self.num_steps_trajectory = config["num_steps_trajectory"]
        self.num_epochs = config["num_epochs"]
        self.batch_size = config["batch_size"]
        self.num_minibatches = config["num_minibatches"]
        self.minibatch_size = self.batch_size // self.num_minibatches

        # Hyperparameters
        self.nn_architecture_actor = config["nn_architecture_actor"]
        self.nn_architecture_critic1 = config["nn_architecture_critic2"]
        self.nn_architecture_critic2 = config["nn_architecture_critic1"]
        self.learning_rate_actor = config["learning_rate_actor"]
        self.learning_rate_critic1 = config["learning_rate_critic1"]
        self.learning_rate_critic2 = config["learning_rate_critic2"]
        self.gamma = config["gamma"]
        self.alpha = config["alpha"]
        self.rho = config["rho"]
        self.max_grad_norm = config["max_grad_norm"]

        # # Memory Initialization
        self.memory = SACMemory(self.mem_maxlen, self.state_size, self.num_actions)

        # Network Initialization
        self.actor = BuildNetwork(self.nn_architecture_actor, self.state_size, self.num_actions, self.learning_rate_actor, self.device)
        self.critic1 = BuildNetwork(self.nn_architecture_critic1, self.state_size, self.num_actions, self.learning_rate_critic1, self.device)
        self.critic2 = BuildNetwork(self.nn_architecture_critic2, self.state_size, self.num_actions, self.learning_rate_critic2, self.device)
        self.critic1_target = BuildNetwork(self.nn_architecture_critic1, self.state_size, self.num_actions, self.learning_rate_critic1, self.device)
        self.critic2_target = BuildNetwork(self.nn_architecture_critic2, self.state_size, self.num_actions, self.learning_rate_critic2, self.device)
        self.update_target_weights()

    def update_target_weights(self):
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_target.eval()
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_target.eval()

    def load_weights(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

    def save_weights(self, path):
        torch.save(self.actor.state_dict(), path)

    def select_action(self, state):

        # Output logits given a state
        logits = self.actor(state)

        # Create a Categorical distribution from the output
        dist = Categorical(logits=logits)

        action = dist.sample()

        probs = dist.probs

        log_probs = torch.log(probs)

        return action, probs, log_probs,

    def compute_critic_losses(self, states_b, actions_b, rewards_b, next_states_b, dones_b):

        with torch.no_grad():
            _, next_probs_b, next_logprobs_b = self.select_action(next_states_b)
            critic1_target = self.critic1_target(next_states_b)
            critic2_target = self.critic2_target(next_states_b)
            min_target = next_probs_b * (torch.min(critic1_target, critic2_target) - self.alpha * next_logprobs_b)

            min_target = min_target.sum(dim=-1).unsqueeze(-1)
            targets = rewards_b.unsqueeze(-1) + self.gamma * (1 - dones_b.unsqueeze(-1)) * min_target

        q_values1_b = self.critic1(states_b).gather(1, actions_b.unsqueeze(1).long())
        q_values2_b = self.critic2(states_b).gather(1, actions_b.unsqueeze(1).long())
        critic1_loss = F.mse_loss(q_values1_b.float(), targets.float())
        critic2_loss = F.mse_loss(q_values2_b.float(), targets.float())

        return critic1_loss, critic2_loss

    def compute_actor_loss(self, states_b):

        new_actions_b, new_probs_b, new_logprobs_b = self.select_action(states_b)
        critic_q_values1 = self.critic1(states_b)
        critic_q_values2 = self.critic2(states_b)
        min_critic_q_value = torch.min(critic_q_values1, critic_q_values2)
        actor_loss = new_probs_b * (self.alpha * new_logprobs_b - min_critic_q_value)
        actor_loss = actor_loss.sum(dim=1).mean()

        return actor_loss

    def update_networks(self):

        states_b, actions_b, rewards_b, next_states_b, dones_b = self.memory.sample(self.batch_size)

        for epoch in range(self.num_epochs):

            # Critics
            critic1_loss, critic2_loss = self.compute_critic_losses(states_b, actions_b, rewards_b, next_states_b, dones_b)
            self.critic1.optimizer.zero_grad()
            self.critic2.optimizer.zero_grad()
            critic1_loss.backward()
            critic2_loss.backward()
            nn.utils.clip_grad_norm_(self.critic1.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic2.parameters(), self.max_grad_norm)
            self.critic1.optimizer.step()
            self.critic2.optimizer.step()

            # Targets
            for params, target_params in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_params.data.copy_(self.rho * target_params.data + (1.0 - self.rho) * params.data)

            for params, target_params in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_params.data.copy_(self.rho * target_params.data + (1.0 - self.rho) * params.data)

            # Actor
            actor_loss = self.compute_actor_loss(states_b)
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor.optimizer.step()



    def run(self, mode, num_episodes):

        rewards = []

        if mode == "training":
            training = True
        else:
            training = False

        for episode in range(num_episodes):

            # Initialize reward to zero
            reward_episode = 0.0

            # Reset environment
            state = torch.tensor(self.env.reset(mode), dtype=torch.float32).to(self.device)

            # Perform one trajectory
            for step in range(self.num_steps_trajectory):

                # Perform one environment step
                with torch.no_grad():
                    action, _, _ = self.select_action(state)
                    reward, next_state, done = self.env.step(action)

                # reward = torch.tensor([reward]).to(self.device)
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                done = torch.tensor([done]).to(self.device)

                reward_episode += reward

                # Keep track of rollout data in case of training
                if training:
                    self.memory.store(state, action, reward, next_state, done)

                # Update state for next step
                state = next_state

            # Append episodic reward
            rewards.append(reward_episode)

            print("Episode: ", episode, "Reward: ", reward_episode)

            # Perform update step with rollout data and clear memory afterwards for new rollout phase
            if training:
                self.update_networks()

        print("Avg. Rewards: ", np.mean(rewards))

        return rewards


if __name__ == "__main__":

    algorithm = "SAC"
    variant = 0
    mode = "train"

    nn_architecture_actor = [(128, "relu"), (128, "relu"), (128, "relu")]
    nn_architecture_critic1 = [(128, "relu"), (128, "relu"), (128, "relu")]
    nn_architecture_critic2 = [(128, "relu"), (128, "relu"), (128, "relu")]
    learning_rate_actor = 0.0003
    learning_rate_critic1 = 0.0003
    learning_rate_critic2 = 0.0003
    gamma = 0.99
    lam = 0.95
    clip_ratio = 0.2
    value_coef = 0.5
    entropy_coef = 0.02
    state_size = 161
    num_actions = 5
    max_grad_norm = 0.5
    alpha = 0.2
    rho = 0.1

    num_episodes_train = 800
    num_episodes_validate = 100
    num_episodes_test = 100
    num_trajectories = 10
    num_steps_trajectory = 200
    num_minibatches = 32
    num_epochs = 10
    mem_maxlen = 1000000
    batch_size = 2000

    config = {
        "nn_architecture_actor": nn_architecture_actor,
        "nn_architecture_critic1": nn_architecture_critic1,
        "nn_architecture_critic2": nn_architecture_critic2,
        "learning_rate_actor": learning_rate_actor,
        "learning_rate_critic1": learning_rate_critic1,
        "learning_rate_critic2": learning_rate_critic2,
        "gamma": gamma,
        "lam": lam,
        "alpha": alpha,
        "rho": rho,
        "clip_ratio": clip_ratio,
        "value_coef": value_coef,
        "entropy_coef": entropy_coef,
        "state_size": state_size,
        "num_actions": num_actions,
        "max_grad_norm": max_grad_norm,
        "num_episodes_train": num_episodes_train,
        "num_episodes_validate": num_episodes_validate,
        "num_episodes_test": num_episodes_test,
        "num_trajectories": num_trajectories,
        "num_steps_trajectory": num_steps_trajectory,
        "num_minibatches": num_minibatches,
        "num_epochs": num_epochs,
        "mem_maxlen": mem_maxlen,
        "batch_size": batch_size
    }

    seed = 45

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = "./data"

    env = Environment(variant, DATA_DIR)

    agent = SAC_Agent(env, config, device)

    rew_train = None

    if mode == "train":

        rew_train = agent.run("training", config["num_episodes_train"])

    rew_val = agent.run("validation", config["num_episodes_validate"])







