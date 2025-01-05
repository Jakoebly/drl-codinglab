import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from utils import utils as ut


# Define the CNN-based Q-Network
class CNNQNetwork(nn.Module):
    def __init__(self, input_shape, action_size):
        super(CNNQNetwork, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1)  # Output: [32, H, W]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: [64, H, W]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: [128, H, W]
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * input_shape[1] * input_shape[2], 512)  # Flattened input
        self.fc2 = nn.Linear(512, action_size)  # Output is number of actions

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the DQN Agent
class CNNDQNAgent:
    def __init__(self, input_shape, action_size, gamma=0.99, lr=0.00005, batch_size=128, memory_size=50000, target_update=150):
        self.input_shape = input_shape
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update = target_update
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        # Initialize networks
        self.q_network = CNNQNetwork(input_shape, action_size)
        self.target_network = CNNQNetwork(input_shape, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Sync target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.update_count = 0

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert observations to proper tensors
        states = torch.FloatTensor(np.stack(states))  # Shape: [batch_size, channels, height, width]
        next_states = torch.FloatTensor(np.stack(next_states))  # Shape: [batch_size, channels, height, width]
        actions = torch.LongTensor(actions)  # Shape: [batch_size]
        rewards = torch.FloatTensor(rewards)  # Shape: [batch_size]
        dones = torch.FloatTensor(dones)  # Shape: [batch_size]

        # Compute target Q-values
        q_targets = rewards + (1 - dones) * self.gamma * torch.max(self.target_network(next_states), dim=1)[0]

        # Compute current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute loss
        loss = self.loss_fn(q_values, q_targets.detach())

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


# Main Training Loop
def train_cnn_dqn(env, episodes=800):
    env.reset(mode="training")  # Initialize the environment
    input_shape = env.get_obs().shape  # Get the shape of the CNN observation
    action_size = env.get_action_len()
    agent = CNNDQNAgent(input_shape, action_size)
    train_rewards = []
    for episode in range(episodes):
        state = env.reset(mode="training")
        state = env.get_obs()  # Use CNN observations
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            reward, next_state, done = env.step(action)
            next_state = env.get_obs()  # Use CNN observations
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
        train_rewards.append(total_reward)

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

    return train_rewards


# Usage
if __name__ == "__main__":
    from environment import Environment  # Assuming the provided environment script is named environment.py
    env = Environment(variant=0, data_dir="drl-codinglab/data/", render=False)
    t_rew = train_cnn_dqn(env)
    ut.rewards_plotting(t_rew)
