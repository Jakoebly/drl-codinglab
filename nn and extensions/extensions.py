import torch
import torch.nn as nn
import torch.optim as optim

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
    
