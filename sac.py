from collections import deque
import os
import time
import numpy as np
from typing import List, Tuple
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from environment import Environment
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------------------------------------
# TODO:
# - Optimize get_obs() and NN structure (CNNs, Recurrent Units, ...)
# - Temperature optimization with tf.Variable() => loss function is similar to policy loss (inlcude E_D[.])
# - Try out SACD with greedy policy as input to test if it is working
# - NN updates could also happen only happen after every ith episode and with mutliple batches
# - First x episodes completely random action? => "warm-up steps" so that buffer is sufficiently large
# - Add rendering option
# --------------------------------------------------------------------------------------------------------------------


class SAC_Agent:
    """Soft Actor-Critic algorithm. Source: https://spinningup.openai.com/en/latest/algorithms/sac.html (OpenAI) and https://arxiv.org/abs/1910.07207 (Christodoulou 2019)."""

    def __init__(self, env: Environment, gamma: float, initial_alpha: float, alpha_decay: float, rho: float, mem_maxlen: int, batch_size: int, learning_rate: float):
        # Environment
        self.env = env

        # General parameters
        self.state_size = env.get_state_len()
        self.action_size = env.get_action_len()
        self.gamma = gamma
        self.alpha = initial_alpha
        self.alpha_decay = alpha_decay
        self.rho = rho

        # Replay buffer. Format: (s,a,r,s',d) in a flattened form
        self.buffer_size = mem_maxlen
        self.buffer_width = 2 * self.state_size + 3  # current_state, next_state, action, reward, and done flag
        self.replay_buffer = np.empty((self.buffer_size, self.buffer_width))
        self.batch_size = batch_size

        # Neural networks
        self.actor = self._nn_model(input_size=self.state_size, output_size=self.action_size, output_act_fct="softmax")
        self.critic_1 = self._nn_model(input_size=self.state_size + 1, output_size=1, output_act_fct="linear")  # critics need as input both the state and the action
        self.critic_2 = self._nn_model(input_size=self.state_size + 1, output_size=1, output_act_fct="linear")  # critics need as input both the state and the action
        self.critic_tgt_1 = self._clone_nn(self.critic_1)
        self.critic_tgt_2 = self._clone_nn(self.critic_2)

        # Neural network gradients
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function(reduce_retracing=True)
    def _select_action(self, state: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Draws an action from the probability distribution that is outputed by the actor NN given the current state. Returns the action and its probability."""
        # Compute probabilities for each action and observation
        p = self.actor(tf.reshape(state, [-1, self.state_size]))  # Probability for each action and observation

        # Sample actions based on probabilities
        actions = tf.random.categorical(tf.math.log(p), num_samples=1)  # Sample actions; input must be log probs
        actions = tf.squeeze(tf.cast(actions, tf.int32), axis=-1)  # Remove the extra dimension

        # Extract probabilities of the chosen actions
        indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)  # Create indices [[row_num, action]] for chosen actions
        probabilities = tf.gather_nd(p, indices)  # Gather probabilities of the chosen actions

        return actions, probabilities

    @tf.function(reduce_retracing=True)
    def _sample_observation(self, buffer_tensor: tf.Tensor, current_buffer_size: int) -> Tuple[tf.Tensor]:
        """Randomly samples an observation from the replay buffer. Returns `(current_states, actions, rew, next_states, d)`, all of them being tensors."""
        # Generate random indices for sampling
        indices = tf.random.uniform(shape=(self.batch_size,), minval=0, maxval=current_buffer_size, dtype=tf.int32)

        # Gather samples based on the indices
        batch = tf.gather(buffer_tensor, indices)

        return (batch[:, 0:75], batch[:, 75], batch[:, 76], batch[:, 77:152], batch[:, 152])  # form: (current_states, actions, rew, next_states, d)

    @tf.function(reduce_retracing=True)
    def _update_alpha(self):
        """Updates current alpha according to alpha decay."""
        self.alpha *= self.alpha_decay

    def _nn_model(self, input_size: int, output_size: int, output_act_fct: str):
        """Returns a NN with an input layer the size of `input_size`, 2 hidden layers with a RELU act. fct., and an output layer with size `output_size` and act. fct. `output_act_fct`."""
        model = keras.Sequential(
            [
                # Two hidden layers with a RELU act. fct. and 64 nodes each
                layers.Input(shape=(input_size,)),
                layers.Dense(128, activation="relu"),  # was: 64
                layers.Dense(128, activation="relu"),  # was: 64
                # Output layer with custom size and act. fct.
                layers.Dense(output_size, activation=output_act_fct),
            ]
        )
        return model

    def _clone_nn(self, og_model):
        """Clones a NN, including its weights."""
        model_clone = keras.models.clone_model(og_model)  # copy structure
        model_clone.set_weights(og_model.get_weights())  # copy weights
        return model_clone

    @tf.function(reduce_retracing=True)
    def _update_value_parameters(self, current_state, action, rew, next_state, d):
        """Update critic NNs via MSE. Inputs should be a tensor."""
        # Define loss function
        with tf.GradientTape() as tape_1:  # record operations for automatic differentiation
            # Calculate intermediary values
            next_action, next_prob = self._select_action(next_state)
            critic_tgt_input = tf.concat([next_state, tf.cast(tf.reshape(next_action, [self.batch_size, 1]), tf.float32)], axis=1)  # combine state and action tensors into one
            clipped_q_val = tf.squeeze(tf.minimum(self.critic_tgt_1(critic_tgt_input), self.critic_tgt_2(critic_tgt_input)))  # tf.squeeze to flatten tensor
            y = tf.cast(rew, tf.float32) + tf.cast((1 - d), tf.float32) * (clipped_q_val - self.alpha * tf.math.log(next_prob))  # castings necessary as tf only works with equal dtypes

            # Calculate loss functions for first critics
            critic_loss_input = tf.concat([current_state, tf.reshape(action, [self.batch_size, 1])], axis=1)  # combine state and action tensors into one
            critic_1_loss = tf.reduce_mean((tf.squeeze(self.critic_1(critic_loss_input)) - y) ** 2)  # tf.queeze to flatten tensor

        with tf.GradientTape() as tape_2:  # record operations for automatic differentiation
            # Calculate intermediary values
            next_action, next_prob = self._select_action(next_state)
            critic_tgt_input = tf.concat([next_state, tf.cast(tf.reshape(next_action, [self.batch_size, 1]), tf.float32)], axis=1)  # combine state and action tensors into one
            clipped_q_val = tf.squeeze(tf.minimum(self.critic_tgt_1(critic_tgt_input), self.critic_tgt_2(critic_tgt_input)))  # tf.squeeze to flatten tensor
            y = tf.cast(rew, tf.float32) + tf.cast((1 - d), tf.float32) * (clipped_q_val - self.alpha * tf.math.log(next_prob))  # castings necessary as tf only works with equal dtypes

            # Calculate loss functions for second critic
            critic_loss_input = tf.concat([current_state, tf.reshape(action, [self.batch_size, 1])], axis=1)  # combine state and action tensors into one
            critic_2_loss = tf.reduce_mean((tf.squeeze(self.critic_2(critic_loss_input)) - y) ** 2)  # tf.queeze to flatten tensor

        # Compute gradients
        critic_1_grads = tape_1.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape_2.gradient(critic_2_loss, self.critic_2.trainable_variables)

        # Update NNs parameters
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))

    @tf.function(reduce_retracing=True)
    def _update_policy_parameters(self, current_state):
        """Update actor NN via loss function from https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py. Input should be a tensor."""
        # Define Loss Function
        with tf.GradientTape() as tape:  # record operations for automatic differentiation
            # Calculate probability for each action and observation
            probs = self.actor(tf.reshape(current_state, [-1, 75]))

            # Repeat current_state tensor to have each state self.action_size times
            states_expanded = tf.tile(tf.expand_dims(current_state, axis=1), [1, self.action_size, 1])  # shape: (batch_size, action_size, state_size)

            # Create tensor of all possible actions and repeat it batch_size times
            actions_expanded = tf.expand_dims(tf.tile(tf.range(self.action_size)[None, :], [self.batch_size, 1]), axis=2)  # shape: (batch_size, action_size, 1)

            # Concat each repeated action to each repeated state
            critic_tgt_input = tf.concat([states_expanded, tf.cast(actions_expanded, tf.float32)], axis=2)  # shape: (batch_size, action_size, state_size + 1)

            # Flatten tensors to shape (batch_size * action_size, state_size + 1) and compute Q-values
            q1_vals = tf.squeeze(self.critic_tgt_1(tf.reshape(critic_tgt_input, [-1, self.state_size + 1])))  # shape: (batch_size * action_size,)
            q2_vals = tf.squeeze(self.critic_tgt_2(tf.reshape(critic_tgt_input, [-1, self.state_size + 1])))  # shape: (batch_size * action_size,)

            # Reshape and compute clipped Q-values
            clipped_q_vals = tf.minimum(tf.reshape(q1_vals, [self.batch_size, self.action_size]), tf.reshape(q2_vals, [self.batch_size, self.action_size]))  # shape: (batch_size, action_size)

            # Compute policy loss
            inner_term = self.alpha * tf.math.log(probs + 1e-8) - clipped_q_vals  # +1e-8 to prevent possibility of log 0 error
            policy_loss = tf.reduce_mean(tf.reduce_sum(probs * inner_term, axis=1))  # tf.reduce_sum to compute dotproduct of probs and inner term

        # Compute gradient
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)

        # Update NN parameters
        self.actor_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

    @tf.function
    def _update_tgt_networks(self):
        """Updates critic tgt NN weights with critic NN weights using Polyak Averaging."""
        # Get weights as TensorFlow variables
        critic_1_weights = self.critic_1.trainable_variables
        critic_2_weights = self.critic_2.trainable_variables
        critic_tgt_1_weights = self.critic_tgt_1.trainable_variables
        critic_tgt_2_weights = self.critic_tgt_2.trainable_variables

        # Perform Polyak averaging for each layer
        new_tgt_weights_1 = [self.rho * tgt_w + (1 - self.rho) * src_w for tgt_w, src_w in zip(critic_tgt_1_weights, critic_1_weights)]
        new_tgt_weights_2 = [self.rho * tgt_w + (1 - self.rho) * src_w for tgt_w, src_w in zip(critic_tgt_2_weights, critic_2_weights)]

        # Apply updated weights to target networks
        for var, new_val in zip(self.critic_tgt_1.trainable_variables, new_tgt_weights_1):
            var.assign(new_val)
        for var, new_val in zip(self.critic_tgt_2.trainable_variables, new_tgt_weights_2):
            var.assign(new_val)

    def _visualize_results(self, rewards: List[float]):
        """Visualizes the episode rewards of the SAC algorithm."""
        # Console Message
        print(f"Average episode reward: {np.average(rewards)}")
        print(f"Maximum episode reward: {np.max(rewards)}")

        # Create time-series line chart
        df = pd.DataFrame(np.array(rewards), columns=["Rewards"])
        plt.figure(figsize=(10, 6))
        plt.plot(df["Rewards"], label="Episode Data", color="#0066BD")
        plt.title("SAC Performance")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.show()

    def run_SAC(self, no_episodes: int, mode: str = "training", visualize_results: bool = True) -> List[float]:
        """Runs the SAC algorithm for `no_episodes`episodes. `mode`must be `training`, `validation`, or `testing`."""
        rewards = []  # contains rewards per episode
        current_buffer_size = 0  # number of filled rows in replay buffer

        # Console message
        print(f"Starting {mode}...")
        start_time = time.time()

        # Load model if in validation or testing mode
        if mode != "training":
            self.actor.load_weights("./checkpoints/sac-actor.weights.h5")
            self.critic_1.load_weights("./checkpoints/sac-critic1.weights.h5")
            self.critic_2.load_weights("./checkpoints/sac-critic2.weights.h5")
            self.critic_tgt_1.load_weights("./checkpoints/sac-critic1-tgt.weights.h5")
            self.critic_tgt_2.load_weights("./checkpoints/sac-critic2-tgt.weights.h5")

        # Run all episodes
        for episode in range(no_episodes):
            # Console message
            if (episode + 1) % 100 == 0:
                print("Episode: " + str(episode + 1) + " / " + str(no_episodes))

            # Reset environment to starting position
            current_state = self.env.flatten_obs(self.env.reset(mode))  # get starting position
            episode_rew = 0  # reset total reward per episode

            for t in range(self.env.episode_steps):
                # Choose and perform action
                action, _ = self._select_action(tf.convert_to_tensor(current_state))  # sample action
                action = int(action)  # unpack tensor
                rew, next_state, d = self.env.step(action)  # observe outcome of action
                next_state = self.env.flatten_obs(next_state)  # flatten observation to 1d array

                # Store observation in replay buffer for NNs updates
                if mode == "training":
                    self.replay_buffer[current_buffer_size % self.buffer_size] = current_state + [action, rew] + next_state + [d]  # modulo to prevent buffer overflow

                episode_rew += rew  # increase total reward per episode
                current_buffer_size += 1

                # End episode if done
                if d:
                    break

                # Continue with next state
                current_state = next_state

            # Update internal variables
            self._update_alpha()
            rewards.append(episode_rew)

            # Update NNs
            if mode == "training":
                # Prepare training batch
                train_current_state, train_action, train_rew, train_next_state, train_d = self._sample_observation(
                    tf.convert_to_tensor(self.replay_buffer, dtype=tf.float32), tf.minimum(current_buffer_size, self.buffer_size)
                )  # min to prevent buffer overflow and make sure all inputs are tensors

                # Perform gradient methods
                self._update_value_parameters(train_current_state, train_action, train_rew, train_next_state, train_d)
                self._update_policy_parameters(train_current_state)
                self._update_tgt_networks()

        # Print and visualize results
        end_time = time.time()
        print(f"Completed. Total runtime: {(end_time-start_time):.5f} seconds")
        if visualize_results:
            self._visualize_results(rewards)

        # Save model weights in training
        if mode == "training":
            # Create folder if it doesn't exist
            if not os.path.exists("./checkpoints"):
                os.makedirs("./checkpoints")

            # Save weights
            self.actor.save_weights("./checkpoints/sac-actor.weights.h5")
            self.critic_1.save_weights("./checkpoints/sac-critic1.weights.h5")
            self.critic_2.save_weights("./checkpoints/sac-critic2.weights.h5")
            self.critic_tgt_1.save_weights("./checkpoints/sac-critic1-tgt.weights.h5")
            self.critic_tgt_2.save_weights("./checkpoints/sac-critic2-tgt.weights.h5")

        return rewards


if __name__ == "__main__":
    # Initialize environment
    env = Environment(variant=0, data_dir="./data")

    # Initialize SAC agent with hyperparameters
    # fmt: off
    agent = SAC_Agent(env=env,
                      gamma=0.99,  # was: 0.99
                      initial_alpha=0.1,  # was: 1; try 0.1 - 0.001
                      alpha_decay=1,  # was: 0.995
                      rho=0.01,  # was: 0.2
                      mem_maxlen=int(1e5),  # was: int(1e7)
                      batch_size=64,  # was: 20
                      learning_rate=0.0001)  # was: 0.0005
    # fmt: on

    # Perform episode runs
    rewards = agent.run_SAC(no_episodes=800, mode="training", visualize_results=True)
