import random
import numpy as np
import os
from typing import List, Tuple
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from keras import layers, optimizers
from environment import Environment
import matplotlib.pyplot as plt


# TODO: extensions: LR annealing, advantage normalization

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.values = []
        self.nextvalues = []
        self.logprobs = []
        self.advantages = []
        self.returns = []

        self.batch_size = batch_size

    def store(self, state, action, reward, next_state, done, value, next_value, log_prob):
        # Assuming all inputs are already tensors, directly append them
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.values.append(value)
        self.nextvalues.append(next_value)
        self.logprobs.append(log_prob)

    def clear(self):
        # Clear all lists (no tensor-specific changes needed here, as lists are not tensorized)
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.values.clear()
        self.nextvalues.clear()
        self.logprobs.clear()
        self.advantages.clear()
        self.returns.clear()

    def to_tensor(self):
        # Convert all lists of tensors to a single tensor
        return {
            'states': tf.convert_to_tensor(self.states, dtype=tf.float32),
            'actions': tf.convert_to_tensor(self.actions, dtype=tf.int32),
            'rewards': tf.convert_to_tensor(self.rewards, dtype=tf.float32),
            'next_states': tf.convert_to_tensor(self.next_states, dtype=tf.float32),
            'dones': tf.convert_to_tensor(self.dones, dtype=tf.float32),
            'values': tf.convert_to_tensor(self.values, dtype=tf.float32),
            'next_values': tf.convert_to_tensor(self.nextvalues, dtype=tf.float32),
            'log_probs': tf.convert_to_tensor(self.logprobs, dtype=tf.float32),
            'advantages': tf.convert_to_tensor(self.advantages, dtype=tf.float32),
            'returns': tf.convert_to_tensor(self.returns, dtype=tf.float32)
        }

    def generate_batches(self, batch_size, minibatch_size, num_epochs):
        # Convert stored data to tensors
        data = self.to_tensor()

        # Create datasets for actor and critic
        dataset_actor = tf.data.Dataset.from_tensor_slices({
            'states': data['states'],
            'actions': data['actions'],
            'log_probs': data['log_probs'],
            'advantages': data['advantages']
        })

        dataset_critic = tf.data.Dataset.from_tensor_slices({
            'states': data['states'],
            'returns': data['returns']
        })

        # Shuffle and batch datasets for actor and critic
        dataset_actor_shuffled = dataset_actor.shuffle(buffer_size=batch_size, reshuffle_each_iteration=True)
        dataset_actor_shuffled_batched = dataset_actor_shuffled.batch(minibatch_size).repeat(num_epochs)

        dataset_critic_shuffled = dataset_critic.shuffle(buffer_size=batch_size, reshuffle_each_iteration=True)
        dataset_critic_shuffled_batched = dataset_critic_shuffled.batch(minibatch_size).repeat(num_epochs)

        return dataset_actor_shuffled_batched, dataset_critic_shuffled_batched


class PPOAgent:
    def __init__(self, agent_hyperparameters, batch_size):
        # Hyperparameters
        self.nn_architecture = agent_hyperparameters["nn_architecture"]  # TODO: define for actor and critic separately
        self.learning_rate = agent_hyperparameters["learning_rate"]
        self.gamma = agent_hyperparameters["gamma"]
        self.lam = agent_hyperparameters["lam"]
        self.clip_ratio = agent_hyperparameters["clip_ratio"]
        self.entropy_bonus = agent_hyperparameters["entropy_bonus"]

        self.state_size = agent_hyperparameters["state_size"]
        self.num_actions = agent_hyperparameters["num_actions"]

        # Memory Initialization
        self.memory = PPOMemory(batch_size)

        # Network Initialization
        self.actor = self.make_actor(self.state_size, self.num_actions)
        self.critic = self.make_critic(self.state_size, 1)

        self.actor_optimizer = self.actor.optimizer
        self.critic_optimizer = self.critic.optimizer

    def make_actor(self, input_size, output_size):
        # Input layer
        inputs = tf.keras.Input(shape=(input_size,))

        # Iteratively add hidden layers according to nn_architecture
        prev_layer = inputs
        for no_neurons in self.nn_architecture:
            prev_layer = layers.Dense(no_neurons, activation='relu')(prev_layer)

        # Output layer
        outputs = layers.Dense(output_size, activation='softmax')(prev_layer)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate))

        return model

    def make_critic(self, input_size, output_size):
        # Input layer
        inputs = tf.keras.Input(shape=(input_size,))

        # Iteratively add hidden layers according to nn_architecture
        prev_layer = inputs
        for no_neurons in self.nn_architecture:
            prev_layer = layers.Dense(no_neurons, activation='relu')(prev_layer)

        # Output layer
        outputs = layers.Dense(output_size, activation='linear')(prev_layer)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate))

        return model

    def save_weights(self, path):
        # TODO: Adjust paths
        actorpath = path + "actor.h5"
        criticpath = path + "critic.h5"
        self.actor.save_weights(actorpath)
        self.critic.save_weights(criticpath)
        print("saved")

    def load_weights(self, path):
        # TODO: Adjust paths
        actorpath = path + "actor.h5"
        criticpath = path + "critic.h5"
        self.policy.load_weights(actorpath)
        self.critic.load_weights(criticpath)
        print("loaded")

    def get_action(self, state):
        probs = self.actor.predict(tf.reshape(state, [1, -1]))
        action = np.random.choice(self.num_actions, p=probs[0])
        return tf.convert_to_tensor(action), tf.convert_to_tensor(probs[0, action])

    def get_value(self, state):
        value = self.critic.predict(tf.reshape(state, [-1, self.state_size]))
        return tf.convert_to_tensor(value[0][0])

    def calc_advantage(self, values, next_values, rewards, dones):
        advantages = tf.zeros_like(values)
        returns = tf.zeros_like(values)
        advantage = tf.zeros_like(values[0])

        for t in tf.range(len(values) - 1, -1, -1):
            mask = tf.cast(1 - dones[t], tf.float32)

            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            advantage = delta + self.gamma * self.lam * mask * advantage

            advantages = tf.tensor_scatter_nd_update(advantages, [[t]], [advantage])
            returns = tf.tensor_scatter_nd_update(returns, [[t]], [advantage + values[t]])

        return advantages, returns

    @tf.function
    def update_actor(self, data):
        for mini_batch in data:
            # Extract tensors directly from the mini_batch
            states = mini_batch['states']
            actions = mini_batch['actions']
            old_log_probs = mini_batch['log_probs']
            advantages = mini_batch['advantages']

            with tf.GradientTape() as tape:
                # Forward pass to get the new probabilities from the actor
                new_probs = self.actor(states)

                # Create a categorical distribution from the probabilities
                dist = tfp.distributions.Categorical(probs=new_probs)

                # Compute the new log probabilities
                new_log_probs = dist.log_prob(actions)

                # Compute the probability ratio (for PPO)
                prob_ratio = tf.math.exp(new_log_probs - old_log_probs)

                # Clip the probability ratio
                clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

                # Compute the entropy for regularization
                entropy = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-8), axis=1)
                entropy = tf.reduce_mean(entropy)

                # Compute the actor loss (PPO objective)
                actor_loss = -tf.reduce_mean(
                    tf.minimum(prob_ratio * advantages, clipped_prob_ratio * advantages)
                ) + self.entropy_bonus * entropy

            # Compute gradients
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

            # Apply gradients to update the actor network
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        print("Updated Actor")

    @tf.function
    def update_critic(self, data):
        for mini_batch in data:
            # Extract tensors directly from the mini_batch
            states = mini_batch['states']
            returns = mini_batch['returns']

            with tf.GradientTape() as tape:
                # Compute the predicted values for the states
                values = self.critic(states)

                # Compute the critic loss
                critic_loss = tf.reduce_mean(tf.square(values - returns))

            # Compute gradients
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

            # Apply gradients to update the critic network
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        print("Updated Critic")


class GymEnvironment:
    def __init__(self, variant, data_dir, agent_hyperparameters):
        self.env = Environment(variant, data_dir)

        self.num_episodes = agent_hyperparameters["num_episodes"]
        self.num_trajectories = agent_hyperparameters["num_trajectories"]
        self.num_steps_trajectory = agent_hyperparameters["num_steps_trajectory"]
        self.batch_size = self.num_trajectories * self.num_steps_trajectory

        self.minibatch_size = self.batch_size // agent_hyperparameters["num_minibatches"]
        self.num_epochs = agent_hyperparameters["num_epochs"]

    def flatten_state(self, state):
        # Assuming 'state' is a tensor
        agent_pos, free_capacity, item_locs, item_times = state
        flattened = tf.concat([
            tf.expand_dims(agent_pos[0], 0),  # First coordinate of agent position
            tf.expand_dims(agent_pos[1], 0),  # Second coordinate of agent position
            tf.expand_dims(free_capacity, 0),  # Ensure free_capacity is 1-dimensional
            tf.reshape(item_locs, [-1]),  # Flatten item_locs
            tf.reshape(item_times, [-1])  # Ensure item_times is flattened to a 1D tensor
        ], axis=0)  # Concatenate along the first axis (axis 0)
        return flattened

    def reset(self):
        # Reset the environment and return the flattened state as a tensor
        state = self.env.reset("training")  # Assuming this returns a structure that can be converted
        return self.flatten_state(state)

    def step(self, action):
        # Take a step in the environment with the action and return the flattened state, reward, and done status
        reward, next_state, done = self.env.step(action)  # Assuming step returns the next state and reward
        next_state_flattened = self.flatten_state(next_state)
        return next_state_flattened, reward, done

    def runPPO(self, agent, training=True):

        rew_episode = []

        for step in range(1, self.num_episodes + 1):

            print("-" * 200)
            print(f"EPISODE: {step} / {self.num_episodes}")
            print("-" * 200)

            rew_trajectory = tf.zeros((), dtype=tf.float32)  # Initialize total reward as a tensor

            state = self.reset()  # State is now a tensor

            for t in range(self.num_steps_trajectory):

                print("-" * 10, f" Step: {t} / {self.num_steps_trajectory} ", "-" * 10)

                # Get action and log_prob as tensors from the agent
                action, log_prob = agent.get_action(state)

                # Take a step in the environment and get the next state, reward, and done flag (all tensors)
                next_state, reward, done = self.step(action)

                # Get the state value (as a tensor)
                value = agent.get_value(state)
                next_value = agent.get_value(next_state)

                # Accumulate reward for the trajectory
                rew_trajectory += reward

                if training:
                    pass
                    # Store the experience (tensor values) in the agent's memory
                    # agent.memory.store(state, action, reward, next_state, done, value, next_value, log_prob)

                state = next_state  # Update state for the next step

            if training and step % self.num_trajectories == 0:
                print("starting update")

                # Compute advantages and returns (these are tensors)
                advantages, returns = agent.calc_advantage(agent.memory.values, agent.memory.nextvalues,
                                                           agent.memory.rewards, agent.memory.dones)

                # Extend memory with advantages and returns (both are tensors)
                agent.memory.advantages.extend(advantages)
                agent.memory.returns.extend(returns)

                # Generate batches from memory (this should return tensors)
                dataset_actor, dataset_critic = agent.memory.generate_batches(self.batch_size, self.minibatch_size,
                                                                              self.num_epochs)

                # Update actor and critic networks using the generated batches (these operations will use tensors)
                agent.update_actor(dataset_actor)
                agent.update_critic(dataset_critic)

                # Clear memory after the update
                agent.memory.clear()

            # Append the total reward of the trajectory to the rewards list (as a tensor)
            rew_episode.append(rew_trajectory)
            print(rew_episode)

        return rew_episode




if __name__ == "__main__":

    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    nn_architecture = [128, 128]
    learning_rate = 0.0004
    gamma = 0.99
    lam = 0.95
    clip_ratio = 0.2
    entropy_bonus = 0.5
    state_size = 75
    num_actions = 5

    num_episodes = 800
    num_trajectories = 1
    num_steps_trajectory = 200
    num_minibatches = 6
    num_epochs = 10


    agent_hyperparameters = {
        "nn_architecture": nn_architecture,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "lam": lam,
        "clip_ratio": clip_ratio,
        "entropy_bonus": entropy_bonus,
        "state_size": state_size,
        "num_actions": num_actions,
        "num_episodes": num_episodes,
        "num_trajectories": num_trajectories,
        "num_steps_trajectory": num_steps_trajectory,
        "num_minibatches": num_minibatches,
        "num_epochs": num_epochs,

    }

    data_dir = "C:/Users/jakob/OneDrive/Uni/Master/3. Semester/Coding Lab/drl-codinglab/data"

    GymEnvironment = GymEnvironment(variant=0, data_dir=data_dir, agent_hyperparameters=agent_hyperparameters)

    agent = PPOAgent(agent_hyperparameters, GymEnvironment.batch_size)

    rewards = GymEnvironment.runPPO(agent, training=True)


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
