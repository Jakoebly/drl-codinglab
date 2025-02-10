import numpy as np
import torch
from collections import deque
import random
import wandb
from typing import List, Tuple

from src import neural_nets as nnets, experiment_settings as expset
from algorithms import dqn_extensions as dqn_ext


class DQN_Agent:
    def __init__(self, env, config, device):
        """
        Initializes the DQN agent with environment, configuration and device settings.

        Parameters:
            env: The environment object the agent interacts with.
            config (dict): A dictionary containing configuration settings.
            device: The device (CPU/GPU) to run computations on.
        """

        # Set device and env variables
        self.device = device
        self.env = env

        # Unpack config
        general = config["general"]
        nn_architecture = config["nn_architecture"]
        hyperparameters = config["hyperparameters"]
        eval_settings = config["eval_settings"]
        extensions = config.get("extensions", [])

        # General variables
        self.state_size = general["state_size"]
        self.action_size = general["action_size"]
        self.global_step_count = 0

        # Evaluation settings
        self.eval_interval = eval_settings["eval_interval"]
        self.evaluation_treshold = eval_settings["evaluation_treshold"]
        self.evaluation_episodes = eval_settings["evaluation_episodes"]

        # NN-architecture
        self.nn_architecture = nn_architecture
        self.nn_structure = nn_architecture["structure"]

        # Hyperparameters
        self.gamma = hyperparameters["gamma"]
        self.epsilon = hyperparameters["epsilon"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.learning_rate = hyperparameters["learning_rate"]
        self.batch_size = hyperparameters["batch_size"]
        self.target_update_interval = hyperparameters["target_update_interval"]

        # Replay buffer initialization
        self.buffer = deque(maxlen=hyperparameters["buffer_size"])

        # Network initialization
        self.q_network = nnets.build_network(self.device, self.nn_architecture, self.state_size, self.action_size, self.learning_rate).to(self.device)
        self.target_network = nnets.build_network(self.device, self.nn_architecture, self.state_size, self.action_size, self.learning_rate).to(self.device)
        self.update_target_weights()

        # Apply CNN-specific methods if structure is CNN
        if self.nn_structure == "CNN":
            self._override_with_cnn_behavior()

        # Extension initialization
        self.extension_parameters = config["extension_parameters"]
        self.extensions = []

        if "vanilla" in extensions:
            extensions = []
        elif "rainbow" in extensions:
            extensions = dqn_ext.EXTENSION_ORDER
        else:
            extensions = [ext for ext in dqn_ext.EXTENSION_ORDER if ext in extensions]

        # Loop through all specified extensions
        for ext_name in extensions:

            # Validate that extension exists
            if ext_name not in dqn_ext.EXTENSION_REGISTRY:
                raise ValueError(f"Unknown extension: {ext_name}")

            # Create initialize extension instance
            ExtensionClass = dqn_ext.EXTENSION_REGISTRY[ext_name]
            extension_instance = ExtensionClass(self)
            self.extensions.append(extension_instance)
            print(f"Initialized {ext_name} extension")

            # Override existing methods with extension-specific implementations
            for method_name in dir(extension_instance):
                if method_name.startswith("__") or not callable(getattr(extension_instance, method_name)):
                    continue
                if hasattr(self, method_name):
                    setattr(self, method_name, getattr(extension_instance, method_name))

    def _override_with_cnn_behavior(self):
        """Overrides default methods with CNN-specific implementations."""

        self.to_tensor = lambda state, batch=False: dqn_ext.CNNBehavior.to_tensor(self, state, batch)
        self.select_action = lambda state, training=True: dqn_ext.CNNBehavior.select_action(self, state, training)
        self.sample_batch = lambda: dqn_ext.CNNBehavior.sample_batch(self)
        self.compute_targets = lambda rewards, next_states, dones: dqn_ext.CNNBehavior.compute_targets(self, rewards, next_states, dones)

    def to_tensor(self, state, batch=False):
        """
        Converts a state into a tensor suitable for model input.

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

    def load_weights(self, path):
        """Loads model weights from a given file path."""

        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()

    def save_weights(self, path):
        """Saves model weights to a given file path."""

        torch.save(self.q_network.state_dict(), path)

    def update_target_weights(self):
        """Synchronizes the weights of the target network with the Q-network."""

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def compute_targets(self, rewards, next_states, dones):
        """
        Computes the target Q-values for training using the Bellman equation.

        Parameters:
            rewards (torch.Tensor): Rewards received from the environment.
            next_states (torch.Tensor): Next states in the episode.
            dones (torch.Tensor): Boolean flags indicating whether the episode is finished (1 if done, 0 otherwise).

        Returns:
            torch.Tensor: Computed target Q-values.
        """

        q_targets_next = torch.max(self.target_network(next_states), dim=1)[0]
        q_targets = rewards + (1 - dones) * self.gamma * q_targets_next

        return q_targets

    def select_action(self, state, training=True) -> int:
        """
        Selects an action based on epsilon-greedy.

        Parameters:
            state (torch.Tensor): The current state of the environment.
            training (bool): Whether the agent is in training mode.

        Returns:
            torch.Tensor: The selected action.
            torch.Tensor: Log probability of the action.
        """

        # Select an action based on epsilon-greedy during training, select the action with highest Q-value otherwise
        if training and np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_size)
        else:
            state = self.to_tensor(state)
            with torch.no_grad():
                q_values = self.q_network(state)
            action = torch.argmax(q_values).item()

        return action

    def record(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer.

        Parameters:
            state (np.ndarray): State before taking the action and transitioning.
            action (int): Action taken by the agent.
            reward (float): Reward generated by taking the action.
            next_state (np.ndarray): State after taking the action and transitioning.
            done (bool): Boolean flag indicating whether the trajectory is finished (1 if done, 0 otherwise).
        """

        self.buffer.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        """Updates epsilon based on exponential decay."""

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def sample_batch(self):
        """
        Samples a batch of transitions from the replay buffer.

        Returns:
            tuple: A batch containing tensors for states, actions, rewards, next states, dones, and ones.
        """

        # Sample from buffer
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)

        # Transform to tensors
        states_tensor = self.to_tensor(states, batch=True)
        next_states_tensor = self.to_tensor(next_states, batch=True)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # TODO: ones tensor fehlt bzw. für was ist der?
        # Aggregate sampled data into tuple
        batch = (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)

        return batch

    def compute_loss(self, q_values, q_targets, weights=None):
        """
        Compute the Temporal Difference (TD) loss.

        Parameters:
            q_values (torch.Tensor): Q-values predicted by the Q-network for a batch of sampled states and actions.
            q_targets (torch.Tensor): Target Q-values computed using the target network.
            weights (torch.Tensor, optional): Weights for the loss computation (default is 1).

        Returns:
            torch.Tensor: Computed loss.
        """

        loss = self.q_network.loss_fn(q_values, q_targets.detach())

        #TODO: Warum wird hier None returned?
        return loss, None

    def update_network(self, loss):
        """
        Performs a gradient descent step to update the Q-network.

        Parameters:
            loss (torch.Tensor): The computed loss used for backpropagation.
        """

        # Perform update
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()

        # Reset noise if extension 'Noisy' is active
        if any(isinstance(ext, dqn_ext.NoisyDQNExtension) for ext in self.extensions):
            self.q_network.reset_noise()
            self.target_network.reset_noise()

    def train_step(self):
        """
        Performs one training step: sample a batch, compute loss, and update the network.

        Returns:
            torch.Tensor: The computed loss for this training step.
        """

        # Buffer size needs to be larger than batch size
        if len(self.buffer) < self.batch_size:
            return

        # Sample and unpack batch of data
        batch = self.sample_batch()
        states, actions, rewards, next_states, dones, *optional = batch

        # TODO: kann weg?
        # Pfusch am Bau -> Siehe dueling bug (wtf?)
        # Clip actions
        actions = torch.clamp(actions, 0, self.action_size - 1)

        # TODO: optional is not implemented, sample_batch does not return any optional
        # Handle weights if extension 'Prioritized Replay' is active
        if optional:
            weights = optional[0].to(self.device)
        else:
            weights = torch.ones(rewards.size(0)).to(self.device)

        # Compute targets
        q_targets = self.compute_targets(rewards, next_states, dones).to(self.device)

        # Compute Q-values based on whether extension 'distribution' is active
        if any(isinstance(ext, dqn_ext.DistributionalDQNExtension) for ext in self.extensions):
            q_network_output = self.q_network.create_dist(*states) if isinstance(states, tuple) else self.q_network.create_dist(states)
            q_network_output.to(self.device)
        else:
            q_network_output = self.q_network(*states) if isinstance(states, tuple) else self.q_network(states)
            q_network_output.to(self.device)

        # Compute loss based on whether extension 'distribution' is active
        if any(isinstance(ext, dqn_ext.DistributionalDQNExtension) for ext in self.extensions):
            batch_indices = torch.arange(q_network_output.size(0)).to(self.device)
            selected_logits = q_network_output[batch_indices, actions]
            loss = self.compute_loss(q_targets, selected_logits)
        else:
            q_values = q_network_output.gather(1, actions.unsqueeze(1)).squeeze()
            loss, *td_errors_opt = self.compute_loss(q_values, q_targets, weights)
            td_errors = td_errors_opt[0] if td_errors_opt else None

        # Update network
        self.update_network(loss)

        # Update priorities if extension 'priority' is active
        if any(isinstance(ext, dqn_ext.PrioritizedReplayBuffer) for ext in self.extensions):
            self.update_priorities(td_errors.detach().abs().cpu().numpy())

        # Update target weights if current step is an update step
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

        # Run the evaluation for a specified number of episodes
        print("Evaluation Start")
        eval_reward = self.run(mode="validation", num_episodes=self.evaluation_episodes, eval=True)
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

    def run(self, mode, num_episodes, eval=False) -> List[float]:
        """
        Runs the DQN agent in the environment for a specified number of episodes.

        Parameters:
            mode (str): Mode of operation ("training", "testing" or "validation").
            num_episodes (int): Number of episodes to run.
            eval (bool): Whether the run is an evaluation run (default is False).

        Returns:
            list: List of total rewards collected per episode.
        """

        # Initialize variables to track rewards and evaluation progress
        rewards = []
        last_evaluation_reward = float('-inf')

        # TODO: EXPERIMENTS!
        if expset.heatmap:
            self.env.reset_cell_visit_count()

        # Enable/disable training mode based on 'mode' argument
        training = mode == "training"

        # Loop through episodes
        for episode in range(num_episodes):

            # Initialize episode reward, loss, done
            reward_episode = 0
            loss_episode = 0
            done = False

            # Reset environment
            state = self.env.reset(mode, structure=self.nn_structure)

            # Perform trajectory
            while not done:
                action = self.select_action(state, training)
                reward, next_state, done = self.env.step(action)
                reward_episode += reward

                # Store data and update if in training mode
                if training:
                    self.record(state, action, reward, next_state, done)
                    loss = self.train_step()
                    if loss is not None:
                        loss_episode += loss

                # Update state for next step
                state = next_state

            # Track total reward per episode
            rewards.append(reward_episode)

            # Update epsilong and log relevant data if in training mode
            if training:
                self.update_epsilon()

                log_dict = {
                    "Charts/Reward per Episode": reward_episode,
                    "Losses/Loss per Episode": loss,
                    "Losses/Cumulative Loss": loss_episode,
                    "Hyperparameters/Learning Rate": self.learning_rate,
                    "Hyperparameters/Epsilon": self.epsilon,
                }
                wandb.log(log_dict, episode)

            # TODO: EXPERIMENTS!
            # Log heatmaps
            if expset.heatmap:
                if not eval and (episode + 1) % expset.heatmap_log_interval == 0 and (episode + 1) >= expset.heatmap_log_interval:
                    self.env.generate_and_log_heatmap(mode, episode)

            # Print progress
            print(f"{mode} episode {episode + 1} / {num_episodes}, total reward: {reward_episode}")

            # Perform evaluation if current episode is an evaluation episode
            # TODO: Check func
            if training and (episode + 1) % self.eval_interval == 0 and episode < num_episodes - 1:
                last_evaluation_reward, terminate = self.evaluate(last_evaluation_reward, episode)
                if terminate:
                    print(f"Training terminated after {episode + 1} episodes as training progress has terminated.")
                    rewards = rewards[: episode + 1]
                    break

        # Print average rewards over total training/validation run
        print(f"--- {mode} finished - average reward after {num_episodes} episodes: {np.mean(rewards)} ---")

        return rewards
