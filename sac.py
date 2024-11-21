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
# - Switch to tf-only syntax, at least for gradient methods
# - NN updates could also happen only happen after every ith episode and with mutliple batches
# - First x episodes completely random action?
# - Temperature optimization
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

    def select_action(self, state: List) -> Tuple[int, float]:
        """Draws an action from the probability distribution that is outputed by the actor NN given the current state. Returns the action and its probability."""
        probabilities = self.actor(np.array(state))
        action = np.random.choice([0, 1, 2, 3, 4], p=probabilities.numpy())
        return action, probabilities[0, action]

    def sample_observation(self) -> List[List]:
        """Randomly samples a single observation from the replay buffer."""
        return random.choices(self.replay_buffer, k=self.batch_size)

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

    @tf.function
    def update_value_parameters(self, current_state, action, rew, next_state, d):
        """Update critic NNs via MSE. Inputs should be a tensor."""
        # TODO: Necessary to use tf functions for simple math? e.g., tf.math.multiply()
        with tf.GradientTape() as tape:  # record operations for automatic differentiation
            # Calculate Intermediary Values
            next_action, next_prob = self.select_action(next_state)
            clipped_q_val = tf.minimum(self.critic_tgt_1(next_state, next_action), self.critic_tgt_2(next_state, next_action))
            y = rew + self.gamma * (1 - d) * (clipped_q_val - self.alpha * tf.math.log(next_prob))

            # Calculate Loss Functions for Both Critics
            critic_1_loss = tf.reduce_mean((self.critic_1(current_state, action) - y) ** 2)
            critic_2_loss = tf.reduce_mean((self.critic_2(current_state, action) - y) ** 2)

        # Compute Gradients
        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)

        # Update NNs Parameters
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))

    @tf.function
    def update_policy_parameters(self, current_state):
        # TODO: Is this the right loss function formula? Reparameterization necessary?
        with tf.GradientTape() as tape:  # record operations for automatic differentiation
            # Calculate Intermediary Values
            action, prob = self.select_action(current_state)
            clipped_q_val = tf.minimum(self.critic_tgt_1(current_state, action), self.critic_tgt_2(current_state, action))

            # Calculate Loss Function for Actor
            policy_loss = tf.reduce_mean(self.alpha * tf.math.log(prob + 1e-8) - clipped_q_val)  # +1e-8 to prevent possibility of log 0 error

        # Compute Gradient
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)

        # Update NN Parameters
        self.actor_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

    @tf.function
    def update_tgt_networks(self):
        # Polyak Averaging
        self.critic_tgt_1.set_weights(self.rho * self.critic_tgt_1.get_weights() + (1 - self.rho) * self.critic_1.get_weights())
        self.critic_tgt_2.set_weights(self.rho * self.critic_tgt_2.get_weights() + (1 - self.rho) * self.critic_2.get_weights())

    def run_SAC(self, no_episodes: int, mode: str = "training") -> List[float]:
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
            current_state = self.env.reset(mode)  # get starting position
            episode_rew = 0  # reset total reward per episode

            for t in range(self.env.episode_steps):
                # Choose and Perform Action
                action, _ = self.select_action(current_state)  # sample action
                rew, next_state, d = self.env.step(action)  # observe outcome of action

                # Store Observation in Replay Buffer
                if mode == "training":
                    self.replay_buffer.append(current_state, action, rew, next_state, d)  # for NN update
                episode_rew += rew  # increase total reward per episode

                # End Episode if Done
                if d:
                    break

            # Update internal variables
            self.update_alpha()
            rewards.append(episode_rew)

            # Update NNs
            self.update_value_parameters()
            self.update_policy_parameters()
            self.update_tgt_networks()

        return rewards


def _visualize_results(rewards: List[float]):
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


def SAC_algorithm(env: Environment, no_episodes: int, mode: str, visualize: bool = True) -> List[float]:
    """Main method. Executes the SAC algorithm for the given environment, visualizes it and returns the average achieved rewards."""
    # Initialize SAC Agent with Hyperparameters
    # fmt: off
    agent = SAC_Agent(env=env,
                      gamma=0.995,
                      initial_alpha=1,
                      alpha_decay=0.9,
                      rho=0.5,
                      mem_maxlen=1000,
                      batch_size=10)
    # fmt: on

    # Perform Episode Runs
    rewards = agent.run_SAC(no_episodes=no_episodes, mode=mode)

    # Visualize results
    if visualize:
        _visualize_results(rewards)

    return rewards


if __name__ == "__main__":
    env = Environment(variant=0, data_dir="./data")
    SAC_algorithm(env=env, no_episodes=800, mode="training")
