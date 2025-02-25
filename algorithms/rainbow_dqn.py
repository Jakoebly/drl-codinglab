import numpy as np
import torch
from collections import deque
import random
import wandb
from typing import List, Tuple

from src import neural_nets as nnets
from algorithms import dqn_extensions as dqn_ext
from algorithms.dqn_extensions import PrioritizedReplayBuffer
from src import experiments as exp


class DQN_Agent:
    def __init__(self, env, config, device):
        self.device = device
        self.env = env

        # Unpack config
        general = config["general"]
        nn_architecture = config["nn_architecture"]
        hyperparameters = config["hyperparameters"]
        eval_settings = config["eval_settings"]
        extensions = config["extensions"]

        # Extensions
        self.double = extensions["double"]  # "double" in extensions
        self.prio_replay = extensions["priority"]
        self.noisy = extensions["noisy"]
        self.distribution = extensions["distribution"]
        self.dueling = extensions["dueling"]
        self.multi_step = extensions["multi_step"]

        # NN-Architecture
        self.nn_architecture = nn_architecture
        self.nn_structure = nn_architecture["structure"]
        self.state_size = general["state_size"]
        self.action_size = general["action_size"]

        # Hyperparameters
        self.gamma = hyperparameters["gamma"]
        self.epsilon = hyperparameters["epsilon"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.batch_size = hyperparameters["batch_size"]
        self.target_update_interval = hyperparameters["target_update_interval"]
        self.extension_params = config["extension_parameters"]

        if self.prio_replay:
            self.buffer = PrioritizedReplayBuffer(capacity=hyperparameters["buffer_size"], alpha=self.extension_params["alpha"], beta=self.extension_params["beta"])
        else:
            self.buffer = deque(maxlen=hyperparameters["buffer_size"])
        self.last_sampled_indices = None

        # Evaluation settings
        self.eval_interval = eval_settings["eval_interval"]
        self.evaluation_treshold = eval_settings["evaluation_treshold"]
        self.evaluation_episodes = eval_settings["evaluation_episodes"]

        # Network Initialization
        self.q_network = nnets.build_network(device, self.nn_architecture, self.state_size, self.action_size, self.learning_rate, extensions, self.extension_params).to(self.device)
        self.target_network = nnets.build_network(device, self.nn_architecture, self.state_size, self.action_size, self.learning_rate, extensions, self.extension_params).to(self.device)
        self.update_target_weights()

        # Assign CNN-specific methods if structure is CNN
        # if self.nn_structure == "CNN":
        #     self._override_with_cnn_behavior()

        # Global step counter for updating the target network
        self.global_step_count = 0

    def _override_with_cnn_behavior(self):
        """Override default methods with CNN-specific implementations."""
        self.to_tensor = lambda state, batch=False: dqn_ext.CNNBehavior.to_tensor(self, state, batch)
        self.select_action = lambda state, training=True: dqn_ext.CNNBehavior.select_action(self, state, training)
        self.sample_batch = lambda: dqn_ext.CNNBehavior.sample_batch(self)
        self.compute_targets = lambda rewards, next_states, dones: dqn_ext.CNNBehavior.compute_targets(self, rewards, next_states, dones)

    def to_tensor(self, state, batch=False):
        """
        Convert a state into a tensor suitable for model input.

        Parameters:
            state (np.ndarray): State or batch of states to tensorize.
            batch (bool): Whether the state represents a batch of states.

        Returns:
            torch.Tensor: Tensorized state(s).
        """

        if self.nn_structure == "DNN":
            if not batch:
                return torch.FloatTensor(state).unsqueeze(0).to(self.device)

            elif batch:
                return torch.FloatTensor(state).to(self.device)
        else:
            if batch:
                # Unpack batch of tuples into separate lists for grid and scalar states
                grid_states, scalar_states = zip(*state)
                grid_tensor = torch.FloatTensor(grid_states).to(self.device)
                scalar_tensor = torch.FloatTensor(scalar_states).to(self.device)
            else:
                # Unpack a single state
                grid_state, scalar_state = state
                grid_tensor = torch.FloatTensor(grid_state).unsqueeze(0).to(self.device)
                scalar_tensor = torch.FloatTensor(scalar_state).unsqueeze(0).to(self.device)

            return grid_tensor, scalar_tensor

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
        if self.distribution:
            with torch.no_grad():
                next_probs = self.target_network.create_dist(next_states)

                if self.double:
                    next_actions = torch.argmax(self.q_network(next_states), dim=1)
                else:
                    next_actions = torch.argmax(next_probs.sum(dim=-1), dim=1)
                batch_indices = torch.arange(next_probs.size(0), device=next_probs.device)
                selected_probs = next_probs[batch_indices, next_actions]

                q_targets = self.categorical_projection(
                    selected_probs,
                    rewards,
                    dones,
                    self.gamma,
                    self.extension_params["num_atoms"],
                    self.extension_params["v_min"],
                    self.extension_params["v_max"],
                )
        elif self.double:
            next_actions = torch.argmax(self.q_network(next_states), dim=1)
            q_targets_next = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            q_targets = rewards + (1 - dones) * self.gamma * q_targets_next
        else:
            q_targets_next = torch.max(self.target_network(next_states), dim=1)[0]
            q_targets = rewards + (1 - dones) * self.gamma * q_targets_next

        return q_targets

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
            m.scatter_add_(1, l_idx.unsqueeze(1), next_probs[:, i : i + 1] * (u[:, i] - b[:, i]).unsqueeze(1))
            m.scatter_add_(1, u_idx.unsqueeze(1), next_probs[:, i : i + 1] * (b[:, i] - l[:, i]).unsqueeze(1))

        return m

    def select_action(self, state, training=True) -> int:
        """
        Select an action based on the epsilon-greedy policy.

        Parameters:
            state (np.ndarray): Current state of the environment.
            training (bool): Whether the agent is in training mode.

        Returns:
            int: The selected action.
        """
        if training and np.random.rand() < self.epsilon and not self.noisy:
            action = np.random.randint(self.action_size)
        else:
            state = self.to_tensor(state)
            with torch.no_grad():
                q_values = self.q_network(state)
            action = torch.argmax(q_values).item()
        return action

    def reset_noise(self):
        """
        Reset noise in both Q-network and target network.
        """
        self.q_network.reset_noise()
        self.target_network.reset_noise()

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
        if self.prio_replay:
            state_tensor = self.to_tensor(state)
            next_state_tensor = self.to_tensor(next_state)

            rewards_tensor = torch.FloatTensor([reward]).to(self.device)
            dones_tensor = torch.FloatTensor([done]).to(self.device)
            q_next = self.compute_targets(rewards_tensor, next_state_tensor, dones_tensor)

            if self.distribution:
                q_network_output = self.q_network.create_dist(state_tensor)
                q_network_output.to(self.device)
                batch_indices = torch.arange(q_network_output.size(0)).to(self.device)
                q_value = q_network_output[batch_indices, action]
            else:
                q_value = self.q_network(state_tensor)[0, action]

            td_error = (-(q_next * q_value.log()).sum(dim=1)) if self.distribution else (q_next.detach() - q_value) ** 2

            transition = (state, action, reward, next_state, done)
            self.buffer.add(transition, td_error.item())
        else:
            self.buffer.append((state, action, reward, next_state, done))

    def update_priorities(self, td_errors):
        """
        Update priorities for sampled transitions.
        """
        if self.last_sampled_indices is None:
            raise ValueError("No samples have been drawn from the buffer.")
        self.buffer.update_priorities(self.last_sampled_indices, td_errors)

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
        if self.nn_structure == "CNN":
            batch = random.sample(self.buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            grid_states, scalar_states = zip(*states)
            next_grid_states, next_scalar_states = zip(*next_states)
            grid_states = np.array(grid_states)
            scalar_states = np.array(scalar_states)
            next_grid_states = np.array(next_grid_states)
            next_scalar_states = np.array(next_scalar_states)

            states_tensor = (
                torch.FloatTensor(grid_states).to(self.device),
                torch.FloatTensor(scalar_states).to(self.device),
            )
            next_states_tensor = (
                torch.FloatTensor(next_grid_states).to(self.device),
                torch.FloatTensor(next_scalar_states).to(self.device),
            )
            actions_tensor = torch.LongTensor(actions).to(self.device)
            rewards_tensor = torch.FloatTensor(rewards).to(self.device)
            dones_tensor = torch.FloatTensor(dones).to(self.device)

            return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
        else:
            # Sample from buffer
            if self.prio_replay:
                batch = self.buffer.sample(self.batch_size)
                states, actions, rewards, next_states, dones, weights, indices = batch
            else:
                batch = random.sample(self.buffer, self.batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = np.array(states)
                next_states = np.array(next_states)
                weights = None
                indices = None

            # Transform to tensors
            states = self.to_tensor(states, batch=True)
            next_states = self.to_tensor(next_states, batch=True)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            weights = torch.FloatTensor(weights).to(self.device) if (weights is not None) else None

            self.last_sampled_indices = indices
            return (states, actions, rewards, next_states, dones, weights)

    def compute_loss(self, q_values, q_targets, weights=None):
        """
        Compute the Temporal Difference (TD) loss.

        Parameters:
            q_values (torch.Tensor): Q-values predicted by the Q-network for the sampled states and actions.
            q_targets (torch.Tensor): Target Q-values computed using the target network.
            weights (torch.Tensor, optional): Weights for the loss computation (default is 1).

        Returns:
            torch.Tensor: Computed loss value.
        """

        td_errors = (-(q_targets * q_values.log()).sum(dim=1)) if self.distribution else (q_targets.detach() - q_values) ** 2
        weights = weights if self.prio_replay else 1
        loss = (weights * td_errors).mean()

        return loss, td_errors

    def update_network(self, loss):
        """
        Perform a gradient descent step to update the Q-network.

        Parameters:
            loss (torch.Tensor): The computed loss to backpropagate.
        """
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()

        # Reset noise if extenstion 'Noisy' is active
        if self.noisy:
            self.q_network.reset_noise()
            self.target_network.reset_noise()

    def train_step(self):
        """
        Perform one training step: sample a batch, compute loss, and update the network.

        Returns:
            torch.Tensor: The computed loss for this training step.
        """
        # Buffer size needs to be larger than batch size
        if len(self.buffer) < self.batch_size:
            return

        # Unpack batch and clip actions
        batch = self.sample_batch()
        states, actions, rewards, next_states, dones, *optional = batch
        # actions = actions.to(self.device)
        # rewards = rewards.to(self.device)
        # dones = dones.to(self.device)
        # Pfusch am Bau -> Siehe dueling bug (wtf?)
        actions = torch.clamp(actions, 0, self.action_size - 1)

        # Handle weights if 'Prioritized Replay' is active
        if self.prio_replay:
            weights = optional[0].to(self.device)
        else:
            weights = torch.ones(rewards.size(0)).to(self.device)

        # Compute targets and q-values
        q_targets = self.compute_targets(rewards, next_states, dones).to(self.device)
        if self.distribution:
            q_network_output = self.q_network.create_dist(states)
            q_network_output.to(self.device)
        else:
            q_network_output = self.q_network(states)  # self.q_network(*states) if isinstance(states, tuple) else self.q_network(states)
            q_network_output.to(self.device)

        # Compute loss

        if self.distribution:
            batch_indices = torch.arange(q_network_output.size(0)).to(self.device)
            q_values = q_network_output[batch_indices, actions]
        else:
            q_values = q_network_output.gather(1, actions.unsqueeze(1)).squeeze()
        loss, *td_errors_opt = self.compute_loss(q_values, q_targets, weights)
        td_errors = td_errors_opt[0] if td_errors_opt else None

        # Update networks and priorities
        self.update_network(loss)

        if self.prio_replay:
            self.update_priorities(td_errors.detach().abs().cpu().numpy())

        if self.global_step_count % self.target_update_interval == 0:
            self.update_target_weights()

        self.global_step_count += 1

        return loss

    def evaluate(self, last_evaluation_reward, episode):
        """
        Evaluates the model's performance on the validation set and checks for stagnation in progress.

        Parameters:
            last_evaluation_reward (float): The reward from the last evaluation to compare against.
            episode (int): The current episode number for logging purposes.

        Returns:
            tuple: A tuple containing:
                - last_evaluation_reward (float): The updated evaluation reward from the current evaluation.
                - terminate (bool): A flag indicating whether training should be terminated due to stagnation.
        """

        # Run the evaluation for a specified number of episodes without updating cell visits for heatmap generation
        print("Evaluation Start")
        old_cvc = exp.save_cell_visit_count(self.env)
        eval_reward = self.run(mode="validation", num_episodes=self.evaluation_episodes, eval=True)
        exp.set_cell_visit_count(self.env, old_cvc)

        # Compute average evaluation reward and log
        average_eval_reward = np.mean(eval_reward)
        wandb.log({"Charts/Evaluation Rewards": average_eval_reward}, episode)

        # Check if progress has stagnated and set terminate variable accordingly
        terminate = False
        difference = average_eval_reward - last_evaluation_reward
        if difference < last_evaluation_reward * self.evaluation_treshold and self.evaluation_treshold != 0:
            terminate = True

        # Update last evaluation reward
        last_evaluation_reward = average_eval_reward

        return last_evaluation_reward, terminate

    def run(self, mode, num_episodes, eval=False):
        """
        Run the DQN agent in the environment for a specified number of episodes.

        Parameters:
            mode (str): Mode of operation ("training", "testing" or "validation").
            num_episodes (int): Number of episodes to run.
            eval (bool): Whether this run is an evaluation (default is False).

        Returns:
            list: Total rewards collected in each episode.
        """

        # Initialize variables to track rewards and evaluation progress
        rewards = []
        last_evaluation_reward = -100

        # Reset cell visit count for heatmap creation
        exp.reset_cell_visit_count(self.env)

        # Enable/disable training mode based on 'mode' argument
        training = mode == "training"

        # Loop through episodes
        for episode in range(num_episodes):

            # Adjust render flag for rendering
            exp.set_render_flag(self.env, mode, eval, episode)

            # Initialize episode reward, loss, and done flag
            reward_episode = 0
            loss_episode = 0
            done = False

            # Reset environment
            state = self.env.reset(mode, structure=self.nn_structure)

            # Perform environment steps
            while not done:

                # Perform one environment step
                action = self.select_action(state, training)
                reward, next_state, done = self.env.step(action)
                reward_episode += reward

                # Store transition data and update if in training mode
                if training:
                    self.record(state, action, reward, next_state, done)
                    loss = self.train_step()
                    if loss is not None:
                        loss_episode += loss

                # Update state for next step
                state = next_state

            # Track total reward per episode
            rewards.append(reward_episode)

            # Print progress
            print(f"{mode} episode {episode + 1} / {num_episodes}, total reward: {reward_episode}")

            # Update epsilon and log if training
            if training:
                # Update epsilon
                self.update_epsilon()

                # Log relevant data
                log_dict = {
                    "Charts/Reward per Episode": reward_episode,
                    "Losses/Loss per Episode": loss,
                    "Losses/Cumulative Loss": loss_episode,
                    "Hyperparameters/Learning Rate": self.learning_rate,
                    "Hyperparameters/Epsilon": self.epsilon,
                }
                wandb.log(log_dict, episode)

            # Log experiment KPIs and heatmap
            exp.log(self.env, mode, training, eval, episode, num_episodes)

            # Log render
            exp.log_render(self.env, mode, episode)

            # Perform evaluation if current episode is an evaluation episode
            if training and (episode + 1) % self.eval_interval == 0 and episode < num_episodes - 1:
                last_evaluation_reward, terminate = self.evaluate(last_evaluation_reward, episode)
                if terminate:
                    print(f"Training terminated after {episode + 1} episodes as training progress has terminated.")
                    rewards = rewards[: episode + 1]
                    break

        # Print average rewards over total training/validation run
        print(f"--- {mode} finished - average reward after {num_episodes} episodes: {np.mean(rewards)} ---")

        return rewards
