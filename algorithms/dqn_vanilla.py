import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from environment import Environment
from utils import utils as ut
from config import dqn_extensions

class QNetwork(nn.Module):
    def __init__(self, nn_architecture):
        super().__init__()

        layers = []

        for i in range(len(nn_architecture) - 1):
            in_size, activation = nn_architecture[i]
            out_size, _ = nn_architecture[i + 1]

            layers.append(nn.Linear(in_size, out_size))

            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")


        last_in_size, last_activation = nn_architecture[-1]
        layers.append(nn.Linear(last_in_size, last_in_size))

        if last_activation.lower() == "softmax":
            layers.append(nn.Softmax(dim=1))
        elif last_activation.lower() == "sigmoid":
            layers.append(nn.Sigmoid())
        elif last_activation.lower() == "linear" or last_activation.lower() is None:
            pass
        else:
            raise ValueError(f"Unsupported activation function: {last_activation}")

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# TODO:
# - validate extension string and implement extensions based on this parameter
# - evtl. unterschiedliche nn_architectures für q and target network
# - evtl. training loop als for loop mit num_steps_trajectory anstatt mit while loop

# Base Extension
class BaseExtension:
    def __init__(self, agent):
        """Initialize with a reference to the agent."""
        self.agent = agent

    def select_action(self, state, training=True):
        """Default epsilon-greedy action selection."""
        if training and np.random.rand() < self.agent.epsilon:
            return np.random.randint(self.agent.num_actions)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.agent.q_network(state)
        return torch.argmax(q_values).item()

    def compute_targets(self, rewards, next_states, dones):
        """Default Q-value target computation."""
        q_targets_next = torch.max(self.agent.target_network(next_states), dim=1)[0]
        return rewards + (1 - dones) * self.agent.gamma * q_targets_next

    def record(self, state, action, reward, next_state, done):
        """Default replay buffer logic."""
        self.agent.buffer.append((state, action, reward, next_state, done))

    def sample_batch(self):
        """Default sample logic."""
        batch = random.sample(self.agent.buffer, self.agent.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
        )

# Double DQN Extension
class DoubleDQNExtension(BaseExtension):
    def compute_targets(self, rewards, next_states, dones):
        """Double DQN target computation."""
        next_actions = torch.argmax(self.agent.q_network(next_states), dim=1)
        q_targets_next = self.agent.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        return rewards + (1 - dones) * self.agent.gamma * q_targets_next

# Prioritized Replay Extension
class PrioritizedReplayExtension(BaseExtension):
    def __init__(self, agent):
        super().__init__(agent)
        self.prioritized_buffer = PrioritizedReplayBuffer(agent.buffer.maxlen)

    def record(self, state, action, reward, next_state, done):
        """Add transition with priority."""
        td_error = abs(reward + self.agent.gamma * torch.max(self.agent.q_network(torch.FloatTensor(next_state))).item() -
                       self.agent.q_network(torch.FloatTensor(state))[action].item())
        self.prioritized_buffer.add((state, action, reward, next_state, done), td_error)

    def sample_batch(self):
        """Sample from prioritized buffer."""
        return self.prioritized_buffer.sample(self.agent.batch_size)

# Extension Registry
EXTENSION_REGISTRY = {
    "double": DoubleDQNExtension,
    "priority": PrioritizedReplayExtension,
}

class DQN_Agent:
    def __init__(self, wandb_run, env, config):
        self.wandb_run = wandb_run
        self.env = env
        self.state_size = config["state_size"]
        self.num_actions = config["num_actions"]
        self.num_episodes_train = config["num_episodes_train"]
        self.num_episodes_validate = config["num_episodes_validate"]
        self.num_episodes_test = config["num_episodes_test"]

        # Hyperparams
        self.nn_architecture = config["nn_architecture"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.target_update_interval = config["target_update_interval"]
        self.buffer = deque(maxlen=config["buffer_size"])

        # Eval settings
        self.eval_interval = 100
        self.evaluation_treshold = 190
        self.evaluation_episodes = 100

        # Initialize networks
        self.q_network = QNetwork(self.nn_architecture)
        self.target_network = QNetwork(self.nn_architecture)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.update_target_weights()

        # Initialize extensions
        self.extensions = []
        for ext_name in config.get("extensions", []):
            ExtensionClass = EXTENSION_REGISTRY[ext_name]
            self.extensions.append(ExtensionClass(self))


        self.step_count = 0

    def load_weights(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()

    def save_weights(self, path):
        torch.save(self.q_network.state_dict(), path)

    def update_target_weights(self):
        """Sync target network weights with Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def select_action(self, state, training=True):
        """Select an action using epsilon-greedy policy."""

        if training and np.random.rand() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            action = torch.argmax(q_values).item()

        return action

    def record(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def sample_batch(self):
        """Sample a batch from the replay buffer."""
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
        )

    def train_step(self):
        """Perform one training step."""
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()
        q_targets = rewards + (1 - dones) * self.gamma * torch.max(self.target_network(next_states), dim=1)[0]
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = self.loss_fn(q_values, q_targets.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network if applicable
        self.step_count += 1
        if self.step_count % self.target_update_interval == 0:
            self.update_target_weights()


    '''def run(self, mode):
        rew_train = None
        rew_val = None
        rew_test = None

        rew = None

        if mode == "train":
            rew_train = self.simulate_environment(self.num_episodes_train, "training")
            rew_val = self.simulate_environment(self.num_episodes_validate, "validation")
        if mode == "validate":
            rew_val = self.simulate_environment(self.num_episodes_validate, "validation")
        elif mode == "test":
            rew_test = self.simulate_environment(self.num_episodes_test, "testing")

        return rew_train, rew_val, rew_test'''


    def run(self, mode, num_episodes):
        rewards = []
        if mode == "training":
            training = True
        else:
            training = False

        for episode in range(num_episodes):
            state = self.env.reset(mode)
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

            rewards.append(reward_episode)

            if training:
                self.update_epsilon()

            self.wandb_run.log({"Reward per Episode": reward_episode})
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {reward_episode}")

            if (episode + 1) % self.eval_interval == 0:
                eval_reward = self.simulate_environment(num_episodes=num_episodes, mode = "validation")
                average_eval_reward = np.mean(eval_reward)
                print(f"Average Evaluation Reward: {average_eval_reward:.2f}")

                if average_eval_reward >= self.evaluation_treshold:
                    print(
                        f"Training terminated after {episode + 1} episodes as the target average reward was achieved.")
                    rewards = rewards[:episode + 1]
                    break

        return rewards