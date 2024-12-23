import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import random
from collections import deque


# Define the CNN-based Q-Network
class CNNQNetwork(tf.keras.Model):
    def __init__(self, input_shape, action_size):
        super(CNNQNetwork, self).__init__()
        # CNN layers
        self.conv1 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation="relu")
        self.conv2 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu")
        self.conv3 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="relu")
        
        # Flatten and fully connected layers
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation="relu")
        self.fc2 = layers.Dense(action_size)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)  # Output: Q-values for each action


# Define the DQN Agent
class CNNDQNAgent:
    def __init__(self, input_shape, action_size, gamma=0.99, lr=0.005, batch_size=64, memory_size=5000, target_update=25):
        self.input_shape = input_shape
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.target_update = target_update
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        # Initialize networks
        self.q_network = CNNQNetwork(input_shape, action_size)
        self.target_network = CNNQNetwork(input_shape, action_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        # Sync target network
        self.update_target_network()

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    @tf.function
    def act(self, state):
        if tf.random.uniform(()) < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.q_network(state)
        return tf.argmax(q_values[0]).numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        # Compute target Q-values
        next_q_values = tf.reduce_max(self.target_network(next_states), axis=1)
        q_targets = rewards + (1 - dones) * self.gamma * next_q_values

        with tf.GradientTape() as tape:
            q_values = tf.reduce_sum(self.q_network(states) * tf.one_hot(actions, self.action_size), axis=1)
            loss = self.loss_fn(q_targets, q_values)

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))
        return loss

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert observations to proper tensors
        states = tf.convert_to_tensor(np.stack(states), dtype=tf.float32)
        next_states = tf.convert_to_tensor(np.stack(next_states), dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Train the network
        loss = self.train_step(states, actions, rewards, next_states, dones)

        # Update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Update target network
        if len(self.memory) % self.target_update == 0:
            self.update_target_network()


# Main Training Loop
@tf.function
def train_episode(env, agent):
    state = env.reset(mode="training")
    state = np.transpose(state, (2, 0, 1))  # Transpose to (height, width, channels)
    state = tf.convert_to_tensor(state[np.newaxis, ...], dtype=tf.float32)
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        reward, next_state, done = env.step(action)
        next_state = np.transpose(next_state, (2, 0, 1))
        next_state = tf.convert_to_tensor(next_state[np.newaxis, ...], dtype=tf.float32)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
    return total_reward


def train_cnn_dqn(env, episodes=800):
    env.reset(mode="training")  # Initialize the environment
    input_shape = env.get_obs().shape  # Get the shape of the CNN observation
    input_shape = (input_shape[1], input_shape[0], input_shape[2])  # Adjust to (height, width, channels)
    action_size = env.get_action_len()
    agent = CNNDQNAgent(input_shape, action_size)

    train_rewards = []
    for episode in range(episodes):
        total_reward = train_episode(env, agent)
        train_rewards.append(total_reward)

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

    return train_rewards


# Usage
if __name__ == "__main__":
    from environment import Environment  # Assuming the provided environment script is named environment.py
    from Utils import utils as ut
    env = Environment(variant=0, data_dir="drl-codinglab/data/", render=False)
    train_rewards = train_cnn_dqn(env)
    ut.rewards_plotting(train_rewards)
