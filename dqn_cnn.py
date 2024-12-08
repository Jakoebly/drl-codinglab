import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from environment import Environment
from Utils import utils as ut

class QNetwork(nn.Module):
    def __init__(self, state_shape, action_size):
        """
        Initialize the Q-Network with CNN layers.
        
        Parameters:
        - state_shape: Tuple representing the shape of the state (C, H, W).
        - action_size: Number of possible actions.
        """
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(state_shape[0], 64, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) 
        
        self.fc_input_size = 128 * state_shape[1] * state_shape[2]
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
    
    def forward(self, x):
        """
        Forward pass of the Q-Network.
        
        Parameters:
        - x: Input tensor of shape (batch_size, C, H, W).
        
        Returns:
        - Q-values for each action.
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        #x = x.view(x.size(0), -1)  # Flatten
        x = x.contiguous()
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self,
                 env: Environment,
                 gamma,
                 epsilon,
                 epsilon_min,
                 epsilon_decay,
                 learning_rate,
                 batch_size,
                 target_update_interval,
                 eval_interval,
                 buffer_size):
        
        self.env = env
        self.state_shape = env.get_state_shape()  # Tuple (C, H, W)
        self.action_size = env.get_action_len()

        self.evaluation_treshold = 190
        self.evaluation_episodes = 50

        # Hyperparams
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval
        self.buffer = deque(maxlen=buffer_size)

        # Initialize networks
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Sync target network
        self.update_target_weights()
        self.step_count = 0

    def _build_model(self):
        """Builds the CNN-based Q-network."""
        return QNetwork(self.state_shape, self.action_size)

    def update_target_weights(self):
        """Sync target network weights with Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def select_action(self, state):
        """Select an action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return self.select_random_action()
        else:
            return self.select_greedy_action(state)
    
    def select_random_action(self):
        return np.random.randint(self.action_size)
    
    '''def select_greedy_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()'''
    
    def select_greedy_action(self, state):
        # Add batch dimension and convert to PyTorch tensor
        state = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
        
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()


    def record(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    '''def sample_batch(self):
        """Sample a batch from the replay buffer."""
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
        )'''
    
    def sample_batch(self):
        """Sample a batch from the replay buffer."""
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)).permute(0, 3, 1, 2),  # Ensure (batch_size, C, H, W)
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)).permute(0, 3, 1, 2),  # Same for next_states
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

    def train(self, episodes=800):
        train_rewards = []

        for episode in range(episodes):
            state = self.env.reset(mode="training")
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                reward, next_state, done = self.env.step(action)
                self.record(state, action, reward, next_state, done)
                self.train_step()
                state = next_state
                total_reward += reward

            train_rewards.append(total_reward)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

            if (episode + 1) % self.eval_interval == 0:
                eval_rewards = self.test(episodes=self.evaluation_episodes, model=self.q_network, evaluation=True)
                average_eval_reward = np.mean(eval_rewards)
                if average_eval_reward >= self.evaluation_treshold:
                    break

        return train_rewards, self.q_network, self.target_network
    
    def test(self, episodes, model, evaluation = False):
        test_rewards = []
        if evaluation:
            mode = "Evalation"
        else: mode = "Testing"

        self.test_model = model

        for episode in range(episodes):
            test_state = self.env.reset("testing")
            test_episode_reward = 0
            done = False

            while not done:
                action = self.select_greedy_action(test_state)
                reward, next_state, done = self.env.step(action)
                test_state = next_state
                test_episode_reward += reward

            test_rewards.append(test_episode_reward)

            print(f"{mode} Episode {episode + 1}/{episodes} | Episode Reward: {test_episode_reward}")

        return test_rewards
    
    def run(self, episodes, mode = "training", model = None, render = False):
        print(f"Starting {mode}...")

        if mode == "training":
            rewards, q_model, target_model = self.train(episodes)
            average_reward = np.mean(rewards)
            print(f"Average Training Reward: {average_reward}")
            ut.rewards_plotting(rewards)
            testing_rewards = self.test(100, q_model)
            average_reward = np.mean(testing_rewards)
            print(f"Average Testing Reward: {average_reward}")
            ut.rewards_plotting(testing_rewards)
            
        else:
            rewards = self.test(episodes, model)
            average_reward = np.mean(rewards)

        #print(f"Average {mode} reward: {average_reward} ")

        return rewards


if __name__ == "__main__":
    env = Environment(variant=2, data_dir="drl-codinglab/data/", render=False)

    agent = DQNAgent(env = env,
                    gamma = 0.91, #0.95
                    epsilon = 1.0,
                    epsilon_min = 0.1, #0.1
                    epsilon_decay = 0.99, #0.99
                    learning_rate = 0.000025, #0.000025
                    batch_size = 64, #64
                    target_update_interval = 200, #200
                    eval_interval = 2000,
                    buffer_size = 80000) #80000

    #rewards = agent.run(episodes = 800, mode = "training", render=False)
    #Old: 0.998, item position yes, distances no
    #ut.rewards_plotting(rewards)

    train_rewards, model,_ = agent.train(episodes=800)
    print(f"Average Training Reward: {np.mean(train_rewards)}")
    print(train_rewards)
    ut.rewards_plotting(train_rewards)
    

    test_rewards = agent.test(episodes=100, model = model)
    print(f"Average Testing Reward: {np.mean(test_rewards)}")
    print(test_rewards)
    ut.rewards_plotting(test_rewards)
