import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from environment import Environment
from utils import utils as ut



# TODO: was soll das _ immer vor den Funktionen?

class QNetwork(nn.Module):
    def __init__(self, state_size, num_actions, hidden_size=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class DQN_Agent:
    def __init__(self, wandb_run, env, config):

        self.wandb_run = wandb_run
        self.env = env
        self.state_size = config["state_size"]
        self.num_actions = config["num_actions"]
        self.num_episodes_train = config["num_episodes_train"]
        self.num_episodes_validate = config["num_episodes_validate"]
        self.num_episodes_test = config["num_episodes_test"]

        self.extension = config["extension"]

        self.evaluation_treshold = config["evaluation_treshold"]
        self.evaluation_episodes = config["evaluation_episodes"]

        # Hyperparams
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.target_update_interval = config["target_update_interval"]
        self.eval_interval = config["eval_interval"]
        self.buffer = deque(maxlen=config["buffer_size"])

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
        return QNetwork(self.state_size, self.num_actions)

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

        for episode in range(10):

            if training:
                state = self.env.reset("training")
            if validation:
                state = self.env.reset("validation")
            if testing:
                state = self.env.reset("testing")

            reward_episode = 0
            done = False

            # TODO: evtl. mit num_steps_trajectory
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

            # TODO: Raus hätt ich gesagt
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