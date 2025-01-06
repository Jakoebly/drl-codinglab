import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from extensions import neural_nets

EXTENSION_ORDER = [
    "double", 
    "priority", 
    "distribution",
    "noisy",
    "dueling",
    "multi_step",
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
    def __init__(self, capacity, alpha = 0.6):
        """Initialize the buffer."""
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

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
        self.agent.buffer = PrioritizedReplayBuffer(capacity = agent.buffer.maxlen)
        self.last_sampled_indices = None

    def sample_batch(self):
        """
        Sample a batch from the prioritized replay buffer.
        """
        batch = self.agent.buffer.sample(self.agent.batch_size)
        states, actions, rewards, next_states, dones, weights, indices = batch

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
    
class NoisyDQNExtension:
    def __init__(self, agent):
        self.agent = agent
        nn_architecture = self.agent.nn_architecture
        state_shape = self.agent.state_size
        action_size = self.agent.action_size

        self.agent.q_network = neural_nets.NoisyAdapter(base_network=self.agent.q_network, action_size=action_size)
        self.agent.target_network = neural_nets.NoisyAdapter(base_network=self.agent.target_network, action_size=action_size)

        self.agent.update_target_weights()

    def select_action(self, state, training=True):
        """
        Select action using noisy exploration (no epsilon-greedy).

        Parameters:
            state (np.ndarray): Current state of the environment.
            training (bool): Whether the agent is in training mode.

        Returns:
            int: The selected action.
        """
        state_tensor = self.agent.tensorize_state(state)
        with torch.no_grad():
            q_values = self.agent.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def reset_noise(self):
        """
        Reset noise in both Q-network and target network.
        """
        self.agent.q_network.reset_noise()
        self.agent.target_network.reset_noise()

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
        self.agent = agent

        self.v_min = -10
        self.v_max = 10
        self.num_atoms = 51

        self.agent.v_min = self.v_min
        self.agent.v_max = self.v_max
        self.agent.num_atoms = self.num_atoms

        self.agent.q_network = neural_nets.convert_to_distributional_network(
            self.agent.q_network,
            action_size=self.agent.action_size,
            num_atoms=self.num_atoms,
        )

        self.agent.target_network = neural_nets.convert_to_distributional_network(
            self.agent.target_network,
            action_size=self.agent.action_size,
            num_atoms=self.num_atoms,
        )

        self.agent.update_target_weights()

    def compute_targets(self, rewards, next_states, dones):
        """
        Compute target distributions for the Distributional DQN extension.

        Accounts for Double DQN and Priority Replay extensions.

        Parameters:
            rewards (torch.Tensor): Rewards received for taking actions.
            next_states (torch.Tensor): Next states resulting from actions taken.
            dones (torch.Tensor): Binary tensor indicating terminal states.

        Returns:
            torch.Tensor: Projected target distribution.
        """
        with torch.no_grad():
            next_probs = self.agent.target_network(next_states)

            if any(isinstance(ext, DoubleDQNExtension) for ext in self.agent.extensions):
                next_actions = torch.argmax(self.agent.q_network(next_states).sum(dim=-1), dim=1)
            else:
                next_actions = torch.argmax(next_probs.sum(dim=-1), dim=1)

            batch_indices = torch.arange(next_probs.size(0), device=next_probs.device)
            selected_probs = next_probs[batch_indices, next_actions]

            target_probs = self.categorical_projection(
                selected_probs, rewards, dones, self.agent.gamma,
                self.num_atoms, self.v_min, self.v_max
            )

        return target_probs

    def categorical_projection(self, next_probs, rewards, dones, gamma, num_atoms, v_min, v_max):
        """
        Project the target distribution onto the support using the Bellman equation.

        Parameters:
            next_probs (torch.Tensor): Probabilities of the next state distribution.
            rewards (torch.Tensor): Rewards for the transitions.
            dones (torch.Tensor): Binary tensor indicating terminal states.
            gamma (float): Discount factor.
            num_atoms (int): Number of atoms in the distribution.
            v_min (float): Minimum value of the distribution.
            v_max (float): Maximum value of the distribution.

        Returns:
            torch.Tensor: Projected target distribution.
        """
        delta_z = (v_max - v_min) / (num_atoms - 1)
        support = torch.linspace(v_min, v_max, num_atoms).to(rewards.device)

        next_support = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * support
        next_support = next_support.clamp(v_min, v_max)

        b = (next_support - v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        m = torch.zeros_like(next_probs)
        for i in range(num_atoms):
            l_idx = torch.clamp(l[:, i], 0, num_atoms - 1)
            u_idx = torch.clamp(u[:, i], 0, num_atoms - 1)
            m.scatter_add_(1, l_idx.unsqueeze(1), next_probs[:, i:i+1] * (u[:, i] - b[:, i]).unsqueeze(1))
            m.scatter_add_(1, u_idx.unsqueeze(1), next_probs[:, i:i+1] * (b[:, i] - l[:, i]).unsqueeze(1))

        return m

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
        self.agent = agent
        nn_architecture = self.agent.nn_architecture
        state_shape = self.agent.state_size
        action_size = self.agent.action_size

        if nn_architecture["structure"] == "DNN":
            self.agent.q_network = neural_nets.DuelingQNetwork(state_shape, action_size, nn_architecture["num_neurons"])
            self.agent.target_network = neural_nets.DuelingQNetwork(state_shape, action_size, nn_architecture["num_neurons"])
        elif nn_architecture["structure"] == "CNN":
            self.agent.q_network = neural_nets.DuelingQNetworkCNN(nn_architecture, state_shape, action_size)
            self.agent.target_network = neural_nets.DuelingQNetworkCNN(nn_architecture, state_shape, action_size)
        else:
            raise ValueError(f"Unsupported structure: {nn_architecture['structure']}")

        self.agent.update_target_weights()

class MultiStepExtension:
    """
    Implements multi-step learning for Deep Q-Learning.

    Multi-step learning uses the cumulative reward over `n` steps instead of a single-step reward,
    allowing for better credit assignment and faster convergence.
    """
    def __init__(self, agent, n_steps=3):
        self.agent = agent
        self.n_steps = n_steps
        self.n_step_buffer = deque(maxlen=n_steps)

        self.previous_record = agent.record
        self.previous_compute_targets = agent.compute_targets

    def record(self, state, action, reward, next_state, done):
        """
        Record a transition with multi-step accumulation.

        Parameters:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state after taking the action.
            done (bool): Whether the episode has ended.
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) == self.n_steps:
            multi_step_reward, (s, a, _, ns, d), multi_step_done = self._get_n_step_transition()
            transition = (s, a, multi_step_reward, ns, multi_step_done)

            self.previous_record(*transition)

    def _get_n_step_transition(self):
        """
        Compute the n-step reward and final transition state.

        Returns:
            tuple: (multi-step reward, final transition, multi-step done flag)
        """
        multi_step_reward = 0
        for idx, (s, a, r, ns, d) in enumerate(self.n_step_buffer):
            multi_step_reward += (self.agent.gamma ** idx) * r
            if d:
                return multi_step_reward, (s, a, r, ns, d), True
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
        # Use the saved "current" compute_targets method
        baseline_targets = self.previous_compute_targets(rewards, next_states, dones)

        # Initialize multi-step targets
        multi_step_targets = torch.zeros_like(rewards)

        # Batch size
        num_samples = rewards.size(0)

        for t in range(num_samples):
            accumulated_reward = 0
            for n in range(self.n_steps):
                if t + n < num_samples:
                    accumulated_reward += (self.agent.gamma ** n) * rewards[t + n]
                    if dones[t + n]:  # Stop if terminal state is reached
                        break
            multi_step_targets[t] = accumulated_reward

        # Combine multi-step rewards with baseline targets
        return multi_step_targets + (1 - dones) * (self.agent.gamma ** self.n_steps) * baseline_targets


EXTENSION_REGISTRY = {
    "double": DoubleDQNExtension,
    "priority": PrioritizedReplayExtension,
    "dueling": DuelingDQNExtension,
    "multi_step": MultiStepExtension,
    "distribution": DistributionalDQNExtension,
    "noisy": NoisyDQNExtension,
}