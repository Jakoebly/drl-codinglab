import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from environment import Environment
from utils import utils as ut
from dqn_extensions import (
    DoubleDQNExtension,
    PrioritizedReplayExtension,
    DuelingDQNExtension,
    MultiStepExtension,
    DistributionalDQNExtension,
    NoisyDQNExtension,
)
from nn_and_extensions.neural_nets import DNN, CNN

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
    
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha = 0.75):
        """Initialize the buffer."""
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.pos = 0

    def add(self, transition, priority):
        """Add a new experience to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        # Store the priority
        self.priorities.append(priority ** self.alpha)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of experiences with priorities."""
        if len(self.buffer) == 0:
            raise ValueError("Replay buffer is empty!")

        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probs[indices]) ** -beta
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
            torch.FloatTensor(weights),
            indices,
        )

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha

# Dueling Q-Network
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_units):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_units),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, action_size)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
    
class DistributionalQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_units, num_atoms, v_min, v_max):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, num_atoms)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_units),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_units, action_size * num_atoms)

    def forward(self, x):
        features = self.feature_layer(x)
        logits = self.output_layer(features)
        logits = logits.view(-1, self.num_atoms)
        probs = torch.softmax(logits, dim=-1)
        return probs
    
def categorical_projection(next_probs, rewards, dones, gamma, num_atoms, v_min, v_max):
    delta_z = (v_max - v_min) / (num_atoms - 1)
    support = torch.linspace(v_min, v_max, num_atoms)
    next_support = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * support

    next_support = next_support.clamp(v_min, v_max)
    b = (next_support - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    m = torch.zeros_like(next_probs)
    for i in range(num_atoms):
        m[:, l[:, i]] += next_probs[:, i] * (u[:, i] - b[:, i])
        m[:, u[:, i]] += next_probs[:, i] * (b[:, i] - l[:, i])

    return m

# Noisy Linear Layer
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        bound = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        self.weight_epsilon = epsilon_out.ger(epsilon_in)
        self.bias_epsilon = epsilon_out

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)

# Noisy Q-Network
class NoisyQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_units):
        super().__init__()
        self.feature_layer = nn.Sequential(
            NoisyLinear(state_size, hidden_units),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_units, hidden_units),
            nn.ReLU(),
            NoisyLinear(hidden_units, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_units, hidden_units),
            nn.ReLU(),
            NoisyLinear(hidden_units, action_size)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# TODO:
# - validate extension string and implement extensions based on this parameter

class BaseImplementation:
    def __init__(self, agent):
        """Initialize with a reference to the agent."""
        self.agent = agent

    def select_action(self, state, training = True):

        if training and np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            action = torch.argmax(q_values).item()

        return action
    
    def compute_targets(self, rewards, next_states, dones):
        """Default Q-value target computation."""
        q_targets_next = torch.max(self.agent.target_network(next_states), dim=1)[0]
        return rewards + (1 - dones) * self.agent.gamma * q_targets_next

    def record(self, state, action, reward, next_state, done, transition):
        """Default replay buffer logic."""
        if transition is None:
            transition = (state, action, reward, next_state, done)
        return transition
    
    def sample_batch(self):
        """Default sample logic."""
        batch = random.sample(self.agent.buffer, self.agent.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        weights = torch.ones(len(states))  # Default weights of 1
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
            torch.FloatTensor(weights),  # Always include weights
        )

class DoubleDQNExtension(BaseImplementation):
    """
    Implements Double Deep Q-Learning (Double DQN) to reduce overestimation bias in Q-value estimation.

    Key Differences from Base DQN:
    - **Decoupled Action Selection and Target Estimation**:
      Double DQN uses the online network (q_network) to select the action:
          next_action = argmax(Q_online(next_state))
      and the target network (target_network) to estimate the Q-value for this action:
          Q_target(next_state, next_action).
    - **Reduced Overestimation Bias**:
      By separating action selection and target estimation, Double DQN improves stability and 
      performance, especially in noisy environments.
    """
    def compute_targets(self, rewards, next_states, dones):
        """Double DQN target computation."""
        next_actions = torch.argmax(self.agent.q_network(next_states), dim=1)
        q_targets_next = self.agent.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        return rewards + (1 - dones) * self.agent.gamma * q_targets_next

class PrioritizedReplayExtension(BaseImplementation):
    """
    Implements Prioritized Experience Replay for Deep Q-Learning.

    This extension enhances the standard replay buffer by assigning a priority to each
    transition, allowing the agent to sample more significant experiences (based on TD error)
    more frequently. Key features include:
    
    - **Priority Sampling**: Transitions with higher TD errors are sampled with higher probability.
    - **Importance Sampling**: Adjusts the loss using weights to account for sampling bias.
    - **Dynamic Priority Updates**: Updates priorities in the buffer after learning from sampled transitions.

    Methods:
    - `record`: Adds transitions to the prioritized buffer with their initial TD error.
    - `sample_batch`: Samples transitions based on priority and provides importance sampling weights.
    - `update_priorities`: Updates the priorities of sampled transitions.
    """
    def __init__(self, agent):
        super().__init__(agent)
        self.prioritized_buffer = PrioritizedReplayBuffer(agent.buffer.maxlen)

    def record(self, state, action, reward, next_state, done, transition=None):
        if transition is None:
            transition = (state, action, reward, next_state, done)

        state, action, reward, next_state, done = transition
        q_value = self.agent.q_network(torch.FloatTensor(state))[action].item()
        q_next = 0 if done else torch.max(self.agent.target_network(torch.FloatTensor(next_state))).item()
        td_error = abs(reward + self.agent.gamma * q_next - q_value)

        self.prioritized_buffer.add(transition, td_error)
        return transition

    def sample_batch(self):
        """Sample from prioritized buffer."""
        batch = self.prioritized_buffer.sample(self.agent.batch_size)
        states, actions, rewards, next_states, dones, weights, indices = batch
        self.last_sampled_indices = indices  # Store for priority updates
        return states, actions, rewards, next_states, dones, weights
    
    def update_priorities(self, td_errors):
        """Update priorities based on new TD errors."""
        self.prioritized_buffer.update_priorities(self.last_sampled_indices, td_errors)

class DuelingDQNExtension(BaseImplementation):
    """
    Implements the Dueling Deep Q-Learning (Dueling DQN) extension.

    This extension replaces the standard Q-network with a dueling network architecture,
    which decomposes Q-value computation into:
    - **Value Stream**: Estimates the value of being in a state, independent of actions.
    - **Advantage Stream**: Estimates the advantage of each action relative to the state.

    The Q-values are computed as:
        Q(s, a) = V(s) + (A(s, a) - mean(A(s, a'))),
    where V(s) is the value and A(s, a) is the advantage.

    Benefits:
    - Improves value estimation in states where the action has little impact.
    - Leads to more stable learning in environments with large action spaces.

    Key Methods:
    - Initializes the dueling network architecture (`DuelingQNetwork`).
    - Replaces the agent's Q-network and target network with the dueling architecture.
    """
    def __init__(self, agent):
        super().__init__(agent)
        state_size = agent.state_size
        action_size = agent.num_actions
        hidden_units = 128  # Example hidden layer size, can be parameterized
        self.agent.q_network = DuelingQNetwork(state_size, action_size, hidden_units)
        self.agent.target_network = DuelingQNetwork(state_size, action_size, hidden_units)
        self.agent.update_target_weights()

# Multi-step Learning Extension
class MultiStepExtension(BaseImplementation):
    """
    Implements multi-step learning for Deep Q-Learning.

    Multi-step learning uses the cumulative reward over `n` steps instead of a single-step reward,
    allowing for better credit assignment and faster convergence.

    Key Features:
    - **n-Step Return**:
      Accumulates rewards over `n` steps:
          G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{n-1} r_{t+n-1}.
    - **Buffer Management**:
      Uses a deque to store transitions until `n` steps are collected.
    - **Replay Buffer Integration**:
      Adds multi-step transitions to the replay buffer, ensuring compatibility with prioritized replay.

    Benefits:
    - Improves stability and learning by incorporating longer-term rewards.
    - Enhances convergence speed by providing richer updates.

    Methods:
    - `record`: Accumulates transitions and computes n-step returns when the buffer is full.
    - `_get_n_step_transition`: Calculates cumulative reward and final transition state.
    """
    def __init__(self, agent, n_steps=3):
        super().__init__(agent)
        self.n_steps = n_steps
        self.n_step_buffer = deque(maxlen=n_steps)

    def record(self, state, action, reward, next_state, done, transition=None):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_steps:
            multi_step_reward, multi_step_state, multi_step_done = self._get_n_step_transition()
            transition = (multi_step_state[0], multi_step_state[1], multi_step_reward, multi_step_state[3], multi_step_done)
            super().record(state, action, reward, next_state, done, transition)
        return transition

    def _get_n_step_transition(self):
        """Compute n-step return and final transition."""
        multi_step_reward = 0
        for idx, (state, action, reward, next_state, done) in enumerate(self.n_step_buffer):
            multi_step_reward += (self.agent.gamma ** idx) * reward
            if done:
                return multi_step_reward, (state, action, reward, next_state, done), True
        return multi_step_reward, self.n_step_buffer[-1], False

    def compute_targets(self, rewards, next_states, dones):
        """Adjust targets for multi-step learning."""
        return super().compute_targets(rewards, next_states, dones)
    
class DistributionalDQNExtension(BaseImplementation):
    """
    Implements the Distributional Deep Q-Learning (C51) extension.

    This extension learns a probability distribution over possible Q-values for each state-action pair
    instead of a single scalar Q-value. The distribution is represented using discrete atoms across
    a value range `[v_min, v_max]`.

    Key Features:
    - **Distributional Output**:
      Outputs a probability distribution for each action over `num_atoms`.
    - **Categorical Projection**:
      Projects the target Q-value distribution onto the atom distribution using the Bellman equation.
    - **Custom Network Architecture**:
      Replaces the Q-network and target network with `DistributionalQNetwork`, which supports
      distributional outputs.

    Benefits:
    - Provides a richer representation of value uncertainty.
    - Can improve stability and performance in environments with stochastic rewards.

    Parameters:
    - `agent`: Reference to the agent, providing access to networks and hyperparameters.
    - `num_atoms` (default=51): Number of discrete points (atoms) in the value distribution.
    - `v_min` (default=-10): Minimum value of the distributional range.
    - `v_max` (default=10): Maximum value of the distributional range.

    Methods:
    - `compute_targets`: Computes the target distribution using categorical projection.
    """
    def __init__(self, agent):
        super().__init__(agent)
        state_size = agent.state_size
        action_size = agent.num_actions
        hidden_units = 128  # Example hidden layer size, can be parameterized
        num_atoms = 51
        v_min = -10
        v_max = 10

        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Replace networks with distributional versions
        self.agent.q_network = DistributionalQNetwork(state_size, action_size, hidden_units, num_atoms, v_min, v_max)
        self.agent.target_network = DistributionalQNetwork(state_size, action_size, hidden_units, num_atoms, v_min, v_max)
        self.agent.update_target_weights()

    def compute_targets(self, rewards, next_states, dones):
        """Compute target distribution using categorical projection."""
        with torch.no_grad():
            next_probs = self.agent.target_network(next_states)
            next_actions = torch.argmax(next_probs.sum(dim=-1), dim=1)
            next_probs = next_probs[torch.arange(next_probs.size(0)), next_actions]

        target_probs = categorical_projection(
            next_probs, rewards, dones, self.agent.gamma,
            self.num_atoms, self.v_min, self.v_max
        )
        return target_probs
    
# Noisy DQN Extension
class NoisyDQNExtension(BaseImplementation):
    """
    Implements Noisy Networks for Deep Q-Learning (Noisy DQN).

    This extension replaces explicit exploration mechanisms (like ε-greedy) with learnable
    noise added to the network weights. Noisy layers allow the agent to explore the environment
    by introducing stochasticity directly into the Q-network's computations.

    Key Features:
    - **Noisy Networks**:
      Replaces standard linear layers with `NoisyLinear`, which adds learnable noise to the weights.
    - **Exploration Without Epsilon**:
      Eliminates the need for ε-decay by using noisy layers for action selection.
    - **Resettable Noise**:
      Resets noise at the beginning of each training step, ensuring consistent exploration.

    Benefits:
    - Adaptive exploration that improves as the agent learns.
    - Removes the need to tune ε-decay parameters.

    Parameters:
    - `agent`: Reference to the agent, providing access to networks and hyperparameters.
    - `hidden_units` (default=128): Number of units in the hidden layers of the network.

    Methods:
    - `select_action`: Selects actions based on the noisy Q-network's output, without ε-greedy logic.
    - `reset_noise`: Resets the noise in all `NoisyLinear` layers at each training step.
    """
    def __init__(self, agent):
        super().__init__(agent)
        state_size = agent.state_size
        action_size = agent.num_actions
        hidden_units = 128  # Example hidden layer size, can be parameterized

        # Replace networks with noisy versions
        self.agent.q_network = NoisyQNetwork(state_size, action_size, hidden_units)
        self.agent.target_network = NoisyQNetwork(state_size, action_size, hidden_units)
        self.agent.update_target_weights()

    def select_action(self, state, training=True):
        """No epsilon-greedy needed; use noisy exploration."""
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.agent.q_network(state)
        return torch.argmax(q_values).item()

    def reset_noise(self):
        self.agent.q_network.reset_noise()
        self.agent.target_network.reset_noise()

EXTENSION_REGISTRY = {
    "double": DoubleDQNExtension,
    "priority": PrioritizedReplayExtension,
    "dueling": DuelingDQNExtension,
    "multi_step": MultiStepExtension,
    "distribution": DistributionalDQNExtension,
    "noisy": NoisyDQNExtension,
}

class DQN_Agent:
    def __init__(self, wandb_run, env, config, device):
        self.wandb_run = wandb_run
        self.device = device
        self.env = env

        # NN Architecture
        self.nn_structure = config["structure"]
        self.nn_architecture = config["architecture"]

        # Env settings
        self.state_size = env.get_state_size(structure = self.nn_structure)
        self.action_size = env.action_size

        '''self.state_size = config["state_size"]
        self.action_size = config["num_actions"]
        self.num_episodes_train = config["num_episodes_train"]
        self.num_episodes_validate = config["num_episodes_validate"]
        self.num_episodes_test = config["num_episodes_test"]'''

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
        self.eval_interval = config["eval_interval"]
        self.evaluation_treshold = config["evaluation_treshold"]
        self.evaluation_episodes = config["evaluation_episodes"]

        # Initialize networks
        self.q_network = QNetwork(self.nn_architecture)
        self.target_network = QNetwork(self.nn_architecture)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.update_target_weights()

        # Initialize extensions
        self.extensions = []
        for ext_name in config.get("extensions", []):
            if ext_name not in EXTENSION_REGISTRY:
                raise ValueError(f"Unknown extension: {ext_name}")
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
        """Delegate action selection through extensions."""
        action = state
        for extension in self.extensions:
            action = extension.select_action(state, training)
        return action

    def record(self, state, action, reward, next_state, done):
        """Delegate recording through extensions."""
        transition = None
        for extension in self.extensions:
            transition = extension.record(state, action, reward, next_state, done, transition)
        if transition is not None:
            self.buffer.append(transition)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def sample_batch(self):
        """Delegate batch sampling through extensions."""
        for extension in self.extensions:
            return extension.sample_batch()

    def train_step(self):
        """Perform one training step."""
        if len(self.buffer) < self.batch_size:
            return

        batch = self.sample_batch()
        states, actions, rewards, next_states, dones, weights = batch

        q_targets = rewards
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        for extension in self.extensions:
            q_targets = extension.compute_targets(rewards, next_states, dones)

        td_errors = q_targets.detach() - q_values
        # Weight == 1 for vanilla implementation!
        loss = (weights * td_errors ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #TODO Def check_exentsions()
        for extension in self.extensions:
            if hasattr(extension, "update_priorities"):
                extension.update_priorities(td_errors.detach().abs().numpy())
            if isinstance(extension, NoisyDQNExtension):
                extension.reset_noise()

        self.step_count += 1

        if self.step_count % self.target_update_interval == 0:
            self.update_target_weights()

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