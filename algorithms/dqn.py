import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import wandb
from typing import List, Tuple

from neural_nets import dqn_extensions, neural_nets
   
class DQN_Agent:
    def __init__(self, env, config, device):
        self.device = device
        self.env = env
        hyperparameters = config["hyperparameters"]
        eval_settings = config["eval_settings"]
        nn_architecture = config["nn_architecture"]
        general = config["general"]
        extensions = config.get("extensions", [])

        """NN Architecture"""
        self.nn_architecture = nn_architecture
        self.nn_structure = nn_architecture["structure"]
        self.state_size = general["state_size"]
        self.action_size = general["action_size"]

        """Hyperparameters"""
        self.gamma = hyperparameters["gamma"]
        self.epsilon = hyperparameters["epsilon"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.batch_size = hyperparameters["batch_size"]
        self.target_update_interval = hyperparameters["target_update_interval"]
        self.buffer = deque(maxlen = hyperparameters["buffer_size"])

        """Evaluation settings"""
        self.eval_interval = eval_settings["eval_interval"]
        self.evaluation_treshold = eval_settings["evaluation_treshold"]
        self.evaluation_episodes = eval_settings["evaluation_episodes"]

        """Initialize networks"""
        self.q_network = neural_nets.create_network(nn_architecture, self.state_size, self.action_size, self.learning_rate).to(self.device)
        self.target_network = neural_nets.create_network(nn_architecture, self.state_size, self.action_size, self.learning_rate).to(self.device)
        self.update_target_weights()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr = self.learning_rate)
        self.loss_fn = nn.MSELoss()
       
        """Initialize extensions"""
        self.extensions = []
        for ext_name in extensions:
            if ext_name not in dqn_extensions.EXTENSION_REGISTRY:
                raise ValueError(f"Unknown extension: {ext_name}")
            ExtensionClass = dqn_extensions.EXTENSION_REGISTRY[ext_name]
            self.extensions.append(ExtensionClass(self))
        
        self.step_count = 0

    def load_weights(self, path):
        """
        Load pre-trained weights into the Q-network.

        Parameters:
            path (str): Path to the file containing the saved weights.
        """
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()

    def save_weights(self, path):
        """
        Save the current Q-network weights to a file.

        Parameters:
            path (str): Path to save the weights file.
        """
        torch.save(self.q_network.state_dict(), path)

    def update_target_weights(self):
        """
        Synchronize the weights of the target network with the Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def compute_targets(self, rewards, next_states, dones):
        """
        Compute target Q-values for training using the Bellman equation.

        Parameters:
            rewards (torch.Tensor): Tensor of rewards from the environment.
            next_states (torch.Tensor): Tensor of next states.
            dones (torch.Tensor): Tensor indicating whether the episode is done (1 if done, 0 otherwise).

        Returns:
            torch.Tensor: Computed target Q-values.
        """
        q_targets_next = torch.max(self.target_network(next_states), dim=1)[0]
        q_targets = rewards + (1 - dones) * self.gamma * q_targets_next
        return q_targets
    
    def tensorize_state(self, state, batch = False):
        """
        Convert a state into a tensor suitable for model input.

        Parameters:
            state (np.ndarray): State or batch of states to tensorize.
            batch (bool): Whether the state represents a batch of states.

        Returns:
            torch.Tensor: Tensorized state(s).
        """
        if not batch:
            return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
        elif batch:
            return torch.FloatTensor(state).to(self.device)
   
    def select_action(self, state, training = True) -> int:
        """
        Select an action based on the epsilon-greedy policy.

        Parameters:
            state (np.ndarray): Current state of the environment.
            training (bool): Whether the agent is in training mode.

        Returns:
            int: The selected action.
        """
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state = self.tensorize_state(state)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def record(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Parameters:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state after taking the action.
            done (bool): Whether the episode has ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        """
        Decay the exploration rate (epsilon) based on the decay factor.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
       
    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions from the replay buffer.

        Returns:
            tuple: A batch containing tensors for states, actions, rewards, next states, dones, and ones.
        """
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)

        states_tensor = self.tensorize_state(states, batch = True)
        next_states_tensor = self.tensorize_state(next_states, batch = True)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        ones_tensor = torch.FloatTensor(torch.ones(len(states))).to(self.device)

        return (states_tensor,
                actions_tensor,
                rewards_tensor,
                next_states_tensor,
                dones_tensor,
                ones_tensor,
        )
    
    def compute_loss(self, q_values, q_targets, weights = 1):
        """
        Compute the Temporal Difference (TD) loss.

        Parameters:
            q_values (torch.Tensor): Q-values predicted by the Q-network for the sampled states and actions.
            q_targets (torch.Tensor): Target Q-values computed using the target network.
            weights (torch.Tensor, optional): Weights for the loss computation (default is 1).

        Returns:
            torch.Tensor: Computed loss value.
        """
        td_errors = q_targets.detach().to(self.device) - q_values.to(self.device)
        weights = weights.to(self.device)
        # Weight == 1 for vanilla implementation
        loss = (weights * td_errors ** 2).mean()
        #loss = self.q_network.loss_fn(q_values, q_targets.detach())
        return loss

    def update_network(self, loss):
        """
        Perform a gradient descent step to update the Q-network.

        Parameters:
            loss (torch.Tensor): The computed loss to backpropagate.
        """
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()

    def train_step(self):
        """
        Perform one training step: sample a batch, compute loss, and update the network.

        Returns:
            torch.Tensor: The computed loss for this training step.
        """
        if len(self.buffer) < self.batch_size:
            return

        batch = self.sample_batch()
        states, actions, rewards, next_states, dones, weights = batch
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        #q_targets = rewards
        q_targets = self.compute_targets(rewards, next_states, dones)
        '''for extension in self.extensions:
            if hasattr(extension, "compute_targets"):
                q_targets = extension.compute_targets(rewards, next_states, dones)'''
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        loss = self.compute_loss(q_values, q_targets, weights)
        self.update_network(loss)

        '''for extension in self.extensions:
            if hasattr(extension, "update_priorities"):
                extension.update_priorities(td_errors.detach().abs().numpy())
            elif isinstance(extension, dqn_extensions.NoisyDQNExtension):
                extension.reset_noise()'''

        if self.step_count % self.target_update_interval == 0:
            self.update_target_weights()
        self.step_count += 1

        return loss

    def run(self, mode, num_episodes, eval = False) -> List[float]:
        """
        Run the agent in the environment for a specified number of episodes.

        Parameters:
            mode (str): Mode of operation ("training", "testing" or "validation").
            num_episodes (int): Number of episodes to run.
            eval (bool): Whether this run is an evaluation (default is False).

        Returns:
            list: Total rewards collected in each episode.
        """
        rewards = []
        last_evaluation_reward = -100

        if mode == "training":
            training = True
        else:
            training = False

        for episode in range(num_episodes):
            state = self.env.reset(mode, structure = self.nn_structure)
            reward_episode = 0
            done = False

            while not done:
                action = self.select_action(state, training)
                #reward, next_state, done = self.env.step(action, structure = self.nn_structure)
                reward, next_state, done = self.env.step(action)

                if training:
                    self.record(state, action, reward, next_state, done)
                    loss = self.train_step()

                state = next_state
                reward_episode += reward
                
            rewards.append(reward_episode)

            if training:
                self.update_epsilon()

                wandb.log({"Losses/Total Loss": loss}, episode)
                wandb.log({"Hyperparameters/Epsilon": self.epsilon}, episode)

            if not eval:
                wandb.log({"Charts/Reward per Episode": reward_episode}, episode)

            print(f"{mode} episode {episode + 1} / {num_episodes}, total reward: {reward_episode}")

            """EVALUATION"""
            if training and (episode + 1) % self.eval_interval == 0 and episode < num_episodes - 1:
                print("evaluation started")
                eval_reward = self.run(mode = "validation", num_episodes = self.evaluation_episodes, eval = True)
                average_eval_reward = np.mean(eval_reward)

                wandb.log({"Charts/Evaluation Rewards": average_eval_reward}, episode)

                difference = average_eval_reward - last_evaluation_reward

                if difference < last_evaluation_reward * self.evaluation_treshold:
                    print(f"Training terminated after {episode + 1} episodes as training progress has terminated.")
                    rewards = rewards[:episode + 1]
                    break

                last_evaluation_reward = average_eval_reward
            
        print(f"--- {mode} finished - average reward after {num_episodes} episodes: {np.mean(rewards)} ---")

        return rewards