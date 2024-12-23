import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from environment import Environment
from Utils import utils as ut

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
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
        self.state_size = env.get_state_len()
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
        """Builds the Q-network."""
        return QNetwork(self.state_size, self.action_size)

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
    
    def select_greedy_action(self,state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def record(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.append((state, action, reward, next_state, done))

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
        #self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Update target network if applicable
        self.step_count += 1
        if self.step_count % self.target_update_interval == 0:
            self.update_target_weights()

    def train(self, episodes=800):
        train_rewards = []

        for episode in range(episodes):
            if (episode + 1) % 100 == 0:
                inter_train_rew = np.mean(train_rewards[-100:])
                print(f"--- Episode:  {str(episode + 1)} / {str(episodes)} --- Avg. Reward Last 100 Episodes: {inter_train_rew} --- Epsilon: {self.epsilon} ---" )
            
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
                eval_reward = self.test(episodes = self.evaluation_episodes, model = self.q_network, eval = True)
                average_eval_reward = np.mean(eval_reward)
                print(f"Average Evaluation Reward: {average_eval_reward:.2f}")

                if average_eval_reward >= self.evaluation_treshold:
                    print(f"Training terminated after {episode + 1} episodes as the target average reward was achieved.")
                    train_rewards = train_rewards[:episode + 1]
                    break

        return train_rewards, self.q_network, self.target_network
    
    def test(self, episodes, model, evaluation = False):
        test_rewards = []
        if evaluation:
            mode = "Evalation"
        else: mode = "Testing"

        self.test_model = model

        for episode in range(episodes):
            test_state = self.env.reset("validation")
            test_episode_reward = 0
            done = False

            while not done:
                action = self.select_greedy_action(test_state)
                reward, next_state, done = self.env.step(action)
                state = next_state
                test_episode_reward += reward

            test_rewards.append(test_episode_reward)

            print(f"{mode} Episode {episode + 1}/{episodes} | Episode Reward: {test_episode_reward}")

        return test_rewards


    def run(self, episodes, mode = "training", model = None, render = False):
        print(f"Starting {mode}...")

        if mode == "training":
            training_rewards, q_model, target_model = self.train(episodes)
            average_reward = np.mean(rewards)
            ut.rewards_plotting(rewards)
            testing_rewards = self.test(episodes, q_model)
            ut.rewards_plotting(testing_rewards)
            
        else:
            rewards = self.test(episodes, model)
            average_reward = np.mean(rewards)

        print(f"Average {mode} reward: {average_reward} ")

        return rewards

if __name__ == "__main__":
    env = Environment(variant=0, data_dir="drl-codinglab/data/", render=False)

    agent = DQNAgent(env = env,
                    gamma = 0.99,
                    epsilon = 1.0,
                    epsilon_min = 0.08,
                    epsilon_decay = 0.996, #0.998
                    learning_rate = 0.00002, #0.00005 #0.000025
                    batch_size = 128,
                    target_update_interval = 30, #150
                    eval_interval = 2000,
                    buffer_size = 50000) #80000
    
    rewards = agent.run(episodes = 800, mode = "training")
    #ut.rewards_plotting(rewards)