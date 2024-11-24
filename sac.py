from collections import deque
import random
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
# - Switch to tf-only syntax, at least for gradient methods => Necessary to use tf functions for simple math? e.g., tf.math.multiply()
# - NN updates could also happen only happen after every ith episode and with mutliple batches
# - First x episodes completely random action?
# - Temperature optimization
# - NN structure as input parameters
# - Policy loss function correct? Reparameterization necessary?
# - @tf.function currently not compatible with numpy operations
# - Algorithm seems to de-converge again (see picture)
# --------------------------------------------------------------------------------------------------------------------


class SAC_Agent:
    """Soft Actor-Critic algorithm. Source: https://spinningup.openai.com/en/latest/algorithms/sac.html (OpenAI) and https://arxiv.org/abs/1910.07207 (Christodoulou 2019)."""

    def __init__(self, env: Environment, gamma: float, initial_alpha: float, alpha_decay: float, rho: float, mem_maxlen: int, batch_size: int):
        # Environment
        self.env = env

        # General Parameters
        self.state_size = env.get_state_len()
        self.action_size = env.get_action_len()
        self.gamma = gamma
        self.alpha = initial_alpha
        self.alpha_decay = alpha_decay
        self.rho = rho

        # Replay Buffer. Format: (s,a,r,s',d)
        self.replay_buffer = deque(maxlen=mem_maxlen)  # supports more efficient append()
        self.batch_size = batch_size

        # Neural Networks
        self.actor = self.nn_model(input_size=self.state_size, output_size=self.action_size, output_act_fct="softmax")
        self.critic_1 = self.nn_model(input_size=self.state_size + 1, output_size=1, output_act_fct="linear")  # critics need as input both the state and the action
        self.critic_2 = self.nn_model(input_size=self.state_size + 1, output_size=1, output_act_fct="linear")  # critics need as input both the state and the action
        self.critic_tgt_1 = self.clone_nn(self.critic_1)
        self.critic_tgt_2 = self.clone_nn(self.critic_2)

        # Neural Network Gradients
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    def select_action(self, state: tf.Tensor) -> Tuple[int, float]:
        """Draws an action from the probability distribution that is outputed by the actor NN given the current state. Returns the action and its probability."""
        p = self.actor(tf.reshape(state, [-1, 75]))  # probability for each action and observation
        actions = [np.random.choice([0, 1, 2, 3, 4], p=p[i].numpy()) for i in range(len(p))]  # sample actions for all observations
        probabilities = [p[i, actions[i]] for i in range(len(actions))]  # probabilities of sampled actions
        return tf.convert_to_tensor(actions), tf.squeeze(probabilities)  # tf.squeeze flattens tensor

    def sample_observation(self) -> Tuple[tf.Tensor]:
        """Randomly samples an observation from the replay buffer. Returns `(current_states, actions, rew, next_states, d)`, all of them being tensors."""
        # Randomly Sample from Replay Buffer
        batch = random.choices(self.replay_buffer, k=self.batch_size)

        # Group Features and Convert Them to Tensors
        current_states = tf.convert_to_tensor([batch[i][0] for i in range(len(batch))])
        actions = tf.convert_to_tensor([batch[i][1] for i in range(len(batch))])
        rew = tf.convert_to_tensor([batch[i][2] for i in range(len(batch))])
        next_states = tf.convert_to_tensor([batch[i][3] for i in range(len(batch))])
        d = tf.convert_to_tensor([batch[i][4] for i in range(len(batch))])

        return (current_states, actions, rew, next_states, d)

    def update_alpha(self):
        """Updates current alpha according to alpha decay."""
        self.alpha *= self.alpha_decay

    def nn_model(self, input_size: int, output_size: int, output_act_fct: str):
        """Returns a NN with an input layer the size of `input_size`, 2 hidden layers with a RELU act. fct., and an output layer with size `output_size` and act. fct. `output_act_fct`."""
        model = keras.Sequential(
            [
                # Two hidden layers with a RELU act. fct. and 64 nodes each
                layers.Dense(64, activation="relu", input_shape=(input_size,)),
                layers.Dense(65, activation="relu"),
                # Output layer with custom size and act. fct.
                layers.Dense(output_size, activation=output_act_fct),
            ]
        )
        return model

    def clone_nn(self, og_model):
        """Clones a NN, including its weights."""
        model_clone = keras.models.clone_model(og_model)  # copy structure
        model_clone.set_weights(og_model.get_weights())  # copy weights
        return model_clone

    # @tf.function
    def update_value_parameters(self, current_state, action, rew, next_state, d):
        """Update critic NNs via MSE. Inputs should be a tensor."""
        # Define Loss Function
        with tf.GradientTape() as tape_1:  # record operations for automatic differentiation
            # Calculate Intermediary Values
            next_action, next_prob = self.select_action(next_state)
            critic_tgt_input = tf.concat([next_state, tf.reshape(next_action, [self.batch_size, 1])], axis=1)  # combine state and action tensors into one
            clipped_q_val = tf.squeeze(tf.minimum(self.critic_tgt_1(critic_tgt_input), self.critic_tgt_2(critic_tgt_input)))  # tf.squeeze to flatten tensor
            y = tf.cast(rew, tf.float32) + tf.cast((1 - d), tf.float32) * (clipped_q_val - self.alpha * tf.math.log(next_prob))  # castings necessary as tf only works with equal dtypes

            # Calculate Loss Functions for First Critics
            critic_loss_input = tf.concat([current_state, tf.reshape(action, [self.batch_size, 1])], axis=1)  # combine state and action tensors into one
            critic_1_loss = tf.reduce_mean((tf.squeeze(self.critic_1(critic_loss_input)) - y) ** 2)  # tf.queeze to flatten tensor

        with tf.GradientTape() as tape_2:  # record operations for automatic differentiation
            # Calculate Intermediary Values
            next_action, next_prob = self.select_action(next_state)
            critic_tgt_input = tf.concat([next_state, tf.reshape(next_action, [self.batch_size, 1])], axis=1)  # combine state and action tensors into one
            clipped_q_val = tf.squeeze(tf.minimum(self.critic_tgt_1(critic_tgt_input), self.critic_tgt_2(critic_tgt_input)))  # tf.squeeze to flatten tensor
            y = tf.cast(rew, tf.float32) + tf.cast((1 - d), tf.float32) * (clipped_q_val - self.alpha * tf.math.log(next_prob))  # castings necessary as tf only works with equal dtypes

            # Calculate Loss Functions for Second Critics
            critic_loss_input = tf.concat([current_state, tf.reshape(action, [self.batch_size, 1])], axis=1)  # combine state and action tensors into one
            critic_2_loss = tf.reduce_mean((tf.squeeze(self.critic_2(critic_loss_input)) - y) ** 2)  # tf.queeze to flatten tensor

        # Compute Gradients
        critic_1_grads = tape_1.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape_2.gradient(critic_2_loss, self.critic_2.trainable_variables)

        # Update NNs Parameters
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))

    # @tf.function
    def update_policy_parameters(self, current_state):
        """Update actor NN via loss function from https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py. Input should be a tensor."""
        # Define Loss Function
        with tf.GradientTape() as tape:  # record operations for automatic differentiation
            # Calculate Intermediary Values
            action, prob = self.select_action(current_state)
            critic_tgt_input = tf.concat([current_state, tf.reshape(action, [self.batch_size, 1])], axis=1)  # combine state and action tensors into one
            clipped_q_val = tf.squeeze(tf.minimum(self.critic_tgt_1(critic_tgt_input), self.critic_tgt_2(critic_tgt_input)))  # tf.queeze to flatten tensor

            # Calculate Loss Function for Actor
            policy_loss = tf.reduce_mean(prob * (self.alpha * tf.math.log(prob + 1e-8) - clipped_q_val))  # +1e-8 to prevent possibility of log 0 error

        # Compute Gradient
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)

        # Update NN Parameters
        self.actor_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

    # @tf.function
    def update_tgt_networks(self):
        """Updates critic tgt NN weights with critic NN weights using Polyak Averaging."""
        # Model Weigths
        critic_1_weights = self.critic_1.get_weights()
        critic_2_weights = self.critic_2.get_weights()
        critic_tgt_1_weights = self.critic_tgt_1.get_weights()
        critic_tgt_2_weights = self.critic_tgt_2.get_weights()

        # Polyak Averaging on a Per Layer Basis
        new_tgt_weights_1 = [self.rho * critic_tgt_1_weights[layer] + (1 - self.rho) * critic_1_weights[layer] for layer in range(len(critic_tgt_1_weights))]
        new_tgt_weights_2 = [self.rho * critic_tgt_2_weights[layer] + (1 - self.rho) * critic_2_weights[layer] for layer in range(len(critic_tgt_2_weights))]

        # Update Critic Tgt Model Weights
        self.critic_tgt_1.set_weights(new_tgt_weights_1)
        self.critic_tgt_2.set_weights(new_tgt_weights_2)

    def visualize_results(self, rewards: List[float]):
        """Visualizes the episode rewards of the SAC algorithm."""
        # Console Message
        print(f"Average episode reward: {np.average(rewards)}")
        print(f"Maximum episode reward: {np.max(rewards)}")

        # Create Time-Series Line Chart
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
        rewards = deque(maxlen=no_episodes)  # contains rewards per episode

        # Console Message
        print(f"Starting {mode}...")

        # Run All Episodes
        for episode in range(no_episodes):
            # Console Message
            if (episode + 1) % 100 == 0:
                print("Episode: " + str(episode + 1) + " / " + str(no_episodes))

            # Reset Environment to Starting Position
            current_state = self.env.flatten_obs(self.env.reset(mode))  # get starting position
            episode_rew = 0  # reset total reward per episode

            for t in range(self.env.episode_steps):
                # Choose and Perform Action
                action, _ = self.select_action(tf.convert_to_tensor(current_state))  # sample action
                rew, next_state, d = self.env.step(action)  # observe outcome of action
                next_state = self.env.flatten_obs(next_state)  # flatten observation to 1d array

                # Store Observation in Replay Buffer
                if mode == "training":
                    self.replay_buffer.append([current_state, action, rew, next_state, d])  # for NN update
                episode_rew += rew  # increase total reward per episode

                # End Episode if Done
                if d:
                    break

                # Continue with next state
                current_state = next_state

            # Update internal variables
            self.update_alpha()
            rewards.append(episode_rew)

            # Update NNs
            if mode == "training":
                train_current_state, train_action, train_rew, train_next_state, train_d = self.sample_observation()
                self.update_value_parameters(train_current_state, train_action, train_rew, train_next_state, train_d)
                self.update_policy_parameters(train_current_state)
                self.update_tgt_networks()

        if visualize_results:
            self.visualize_results(rewards)

        return rewards


if __name__ == "__main__":
    # Initialize Environment
    env = Environment(variant=0, data_dir="./data")

    # Initialize SAC Agent with Hyperparameters
    # fmt: off
    agent = SAC_Agent(env=env,
                      gamma=0.99,
                      initial_alpha=0.95,
                      alpha_decay=1,
                      rho=0.2,
                      mem_maxlen=int(1e7),
                      batch_size=20)
    # fmt: on

    # Perform Episode Runs
    rewards = agent.run_SAC(no_episodes=800, mode="training", visualize_results=True)
