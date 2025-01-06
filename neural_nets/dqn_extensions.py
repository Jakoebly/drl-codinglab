import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

EXTENSION_ORDER = [
    "double", 
    "priority", 
    "multi_step", 
    "dueling", 
    "distribution", 
    "noisy"
]

class DoubleDQNExtension:
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
    def __init__(self, agent):
        self.agent = agent

    def compute_targets(self, rewards, next_states, dones):
        """Double DQN target computation."""
        # Online network action selection
        next_actions = torch.argmax(self.agent.q_network(next_states), dim=1)
        # Target network Q-value estimation for action from online network
        q_targets_next = self.agent.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        # Computing target Q-values via Bellman equation
        q_targets = rewards + (1 - dones) * self.agent.gamma * q_targets_next

        return q_targets
    
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha = 0.75):
        """Initialize the buffer."""
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)  # Return the length of the buffer

    def add(self, transition, priority):
        """Add a new experience to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        # Storing priority
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
        states = np.array(states)
        next_states = np.array(next_states)

        states_tensor = torch.FloatTensor(states)
        next_states_tensor = torch.FloatTensor(next_states)
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        dones_tensor = torch.FloatTensor(dones)
        weights_tensor = torch.FloatTensor(weights)

        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            weights_tensor,
            indices,
        )

    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha

class PrioritizedReplayExtension:
    def __init__(self, agent):
        """
        Implements Prioritized Experience Replay.
        """
        self.agent = agent
        # Replace the default buffer with prioritized buffer
        self.agent.buffer = PrioritizedReplayBuffer(capacity = agent.buffer.maxlen)
        self.last_sampled_indices = None

    def sample_batch(self):
        """
        Sample a batch from the prioritized replay buffer.
        """
        batch = self.agent.buffer.sample(self.agent.batch_size)
        states, actions, rewards, next_states, dones, weights, indices = batch

        # Move tensors to the correct device
        states = states.to(self.agent.device)
        actions = actions.to(self.agent.device)
        rewards = rewards.to(self.agent.device)
        next_states = next_states.to(self.agent.device)
        dones = dones.to(self.agent.device)
        weights = weights.to(self.agent.device)

        self.last_sampled_indices = indices
        return states, actions, rewards, next_states, dones, weights

    def record(self, state, action, reward, next_state, done):
        """
        Add transitions to the prioritized buffer using TD error as priority.
        """
        state_tensor = self.agent.tensorize_state(state)
        next_state_tensor = self.agent.tensorize_state(next_state)

        rewards_tensor = torch.FloatTensor([reward]).to(self.agent.device)
        dones_tensor = torch.FloatTensor([done]).to(self.agent.device)
        q_next = self.agent.compute_targets(rewards_tensor, next_state_tensor, dones_tensor)

        q_value = self.agent.q_network(state_tensor)[0, action]
        td_error = self.compute_td_errors(q_value, q_next).item()

        transition = (state, action, reward, next_state, done)
        self.agent.buffer.add(transition, abs(td_error))

    def update_priorities(self, td_errors):
        """
        Update priorities for sampled transitions.
        """
        if self.last_sampled_indices is None:
            raise ValueError("No samples have been drawn from the buffer.")
        self.agent.buffer.update_priorities(self.last_sampled_indices, td_errors)

    def compute_td_errors(self, q_values, q_targets):
        """
        Compute the Temporal Difference (TD) errors.

        Parameters:
            q_values (torch.Tensor): Q-values predicted by the Q-network for the sampled states and actions.
            q_targets (torch.Tensor): Target Q-values computed using the target network.

        Returns:
            torch.Tensor: Computed TD errors.
        """
        td_errors = q_targets.detach() - q_values

        return td_errors

    def compute_loss(self, q_values, q_targets, weights):
        td_errors = self.compute_td_errors(q_values, q_targets)
        loss = (weights * td_errors ** 2).mean()

        return loss, td_errors

# Multi-step Learning Extension
class MultiStepExtension:
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
        self.agent = agent
        self.n_steps = n_steps
        self.n_step_buffer = deque(maxlen=n_steps)

    def record(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_steps:
            multi_step_reward, multi_step_state, multi_step_done = self._get_n_step_transition()
            transition = (multi_step_state[0], multi_step_state[1], multi_step_reward, multi_step_state[3], multi_step_done)

            for extension in self.agent.extensions:
                if extension != self and hasattr(extension, "record"):
                    extension.record(*transition)

    def _get_n_step_transition(self):
        """Compute n-step return and final transition."""
        multi_step_reward = 0
        for idx, (state, action, reward, next_state, done) in enumerate(self.n_step_buffer):
            multi_step_reward += (self.agent.gamma ** idx) * reward
            if done:
                return multi_step_reward, (state, action, reward, next_state, done), True
        return multi_step_reward, self.n_step_buffer[-1], False
    
    def compute_targets(self, rewards, next_states, dones):
        """
        Compute Q-value targets for multi-step learning.

        Parameters:
            rewards (torch.Tensor): Rewards received from the environment.
            next_states (torch.Tensor): Next states resulting from actions taken.
            dones (torch.Tensor): Binary tensor indicating terminal states (1 if done, 0 otherwise).

        Returns:
            torch.Tensor: Multi-step target Q-values.
        """
        # Use the agent's compute_targets method to get baseline or combined targets
        baseline_targets = self.agent.compute_targets(rewards, next_states, dones)

        # Modify targets for multi-step learning
        multi_step_targets = torch.zeros_like(rewards)
        for t in range(len(rewards)):
            for n in range(self.n_steps):
                if t + n < len(rewards):
                    multi_step_targets[t] += (self.agent.gamma ** n) * rewards[t + n]
                if dones[t + n]:
                    break

        # Combine multi-step targets with baseline targets for consistency
        return multi_step_targets + (1 - dones) * (self.agent.gamma ** self.n_steps) * baseline_targets


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
    
class DuelingDQNExtension:
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

class DistributionalDQNExtension:
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

# Noisy DQN Extension
class NoisyDQNExtension:
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