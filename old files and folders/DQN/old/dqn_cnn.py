import tensorflow as tf
from keras import layers
import numpy as np
import random
from environment import Environment
from collections import deque

from typing import List, Tuple

from utils import utils as ut


class DQN_Agent:
    def __init__(self, env: Environment,
                 gamma: float,
                 epsilon: float,
                 batch_size: int,
                 epsilon_min: float,
                 epsilon_decay: float,
                 learning_rate: float,
                 actual_update_interval : int,
                 target_update_interval : int,
                 eval_interval: int,
                 mem_maxlen: int):
        
        self.env = env
        self.state_size = self.env.get_state_len()
        self.action_size = self.env.get_action_len()

        self.gamma = gamma
        self.epsilon = tf.Variable(initial_value=epsilon, dtype = tf.float32, trainable = False)
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate

        # Replay buffer
        self.replay_buffer = deque(maxlen=mem_maxlen)

        self.actual_model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_weights()

        # Other attributes
        self.actual_update_interval = actual_update_interval
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval

        # Optimizer and Loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()


    def _build_model(self):
        """Builds a CNN-based Q-network."""
        input_shape = (self.env.vertical_cell_count, self.env.horizontal_cell_count, 5)  # 4 channels
        model = tf.keras.Sequential([
            layers.Input(shape=input_shape, name="grid_input"),

            # Convolutional layers
            layers.Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same"),
            #layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same"),
            layers.Conv2D(256, kernel_size=3, strides=1, activation="relu", padding="same"),
            #layers.Conv2D(64, kernel_size=3, strides=1, activation="relu", padding="same"),
            #layers.MaxPooling2D(pool_size=(2, 2)),

            #layers.GlobalAveragePooling2D(),
            layers.Flatten(),

            # Fully connected layers
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),

            # Output layer: one Q-value per action
            layers.Dense(self.action_size, activation="softmax"),
        ])
        return model

    
    def update_target_weights(self):
        self.target_model.set_weights(self.actual_model.get_weights())
        
    def select_random_action(self, state):
        """Select an action using epsilon-greedy policy."""
        #if tf.random.uniform(()) < self.epsilon.numpy():
        if tf.random.uniform(()) < self.epsilon:
            # Select a random action using TensorFlow
            return random.randint(0, self.action_size - 1)
            #return tf.random.uniform([], maxval=self.action_size, dtype=tf.int32)
        else:
            return self.select_greedy_action(state)

    def select_greedy_action(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        q_values = self.actual_model(state, training=False)
        return tf.argmax(q_values[0]).numpy()
    
    def record(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_batch(self):
        """Sample a batch from replay buffer."""
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(actions, dtype=tf.int32),
            tf.convert_to_tensor(rewards, dtype=tf.float32),
            tf.convert_to_tensor(next_states, dtype=tf.float32),
            tf.convert_to_tensor(dones, dtype=tf.float32),
        )
    
    def _sample_observation(self) -> Tuple[tf.Tensor]:
        """Randomly samples an observation from the replay buffer. Returns `(current_states, actions, rew, next_states, d)`, all of them being tensors."""
        # Randomly Sample from Replay Buffer
        batch = random.choices(self.replay_buffer, k = self.batch_size)
        # Group Features and Convert Them to Tensors
        current_states = tf.convert_to_tensor([batch[i][0] for i in range(len(batch))])
        actions = tf.convert_to_tensor([batch[i][1] for i in range(len(batch))])
        rew = tf.convert_to_tensor([batch[i][2] for i in range(len(batch))])
        next_states = tf.convert_to_tensor([batch[i][3] for i in range(len(batch))])
        d = tf.convert_to_tensor([batch[i][4] for i in range(len(batch))])

        return (current_states, actions, rew, next_states, d)

    @tf.function
    def compute_q_targets(self, reward, done, max_q_next):
        """Compute the target Q-values for the Bellman equation."""
        q_target = reward + (1 - done) * self.gamma * max_q_next
        return q_target
    
    # Updates epsilon after each episode as long as it is larger than our defined minvalue
    @tf.function
    def update_epsilon(self):
        #self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon.assign(tf.maximum(self.epsilon_min, self.epsilon * self.epsilon_decay))

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """Perform one gradient descent step."""
        with tf.GradientTape() as tape:
            # Predict Q-values for current and next states
            q_values = self.actual_model(states, training=True)
            q_next_values = self.target_model(next_states, training=False)

            # Compute the TD target
            max_q_next = tf.reduce_max(q_next_values, axis=1)
            targets = self.compute_q_targets(rewards, dones, max_q_next)

            # Update Q-values for the actions taken
            actions_one_hot = tf.one_hot(actions, depth=self.action_size)
            q_values_pred = tf.reduce_sum(q_values * actions_one_hot, axis=1)
            loss = self.loss_fn(targets, q_values_pred)

        # Backpropagation
        grads = tape.gradient(loss, self.actual_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actual_model.trainable_variables))
        return loss

    def train_DQN(self, no_episodes: int, render: bool = False):
        """Train the agent."""
        train_rewards = []
        for episode in range(no_episodes):
            if (episode + 1) % 100 == 0:
                inter_train_rew = np.mean(train_rewards[-100:])
                print(f"--- Episode:  {str(episode + 1)} / {str(no_episodes)} --- Avg. Reward Last 100 Episodes: {inter_train_rew} --- Epsilon: {self.epsilon} ---" )

            #current_state = self.env.flatten_obs(self.env.reset("training"))
            current_state = self.env.reset("training")
            episode_rew = 0

            for step in range(self.env.episode_steps):
                action = self.select_random_action(state = current_state)  # sample action
                rew, next_state, d = self.env.step(action)  # observe outcome of action
                #next_state = self.env.flatten_obs(next_state)  # flatten observation to 1d array

                episode_rew += rew
                self.record(current_state, action, rew, next_state, d)

                #self.update_actual_weights()

                # Perform training step
                if len(self.replay_buffer) >= self.batch_size and step % self.actual_update_interval == 0:
                    states, actions, rewards, next_states, dones = self.sample_batch()
                    loss = self.train_step(states, actions, rewards, next_states, dones)

                if step % self.target_update_interval == 0:
                    self.update_target_weights()  # Updating the weights of the target network every t episodes

                if render:
                    self.env.render()  # Call the render method

                if d:
                    break
                
                current_state = next_state

            self.update_epsilon()
            train_rewards.append(episode_rew)

            print(f"Episode {episode + 1}/{no_episodes}: Total Reward = {episode_rew}")

            # EVALUATION - NOT NEEDED RIGHT NOW
            '''if (episode + 1) % self.eval_interval == 0:
                eval_reward = self.test_DQN(no_episodes = 50, mode = "evaluation", visualize_results=False)
                average_reward = np.mean(eval_reward)
                print(f"Average Reward of last 100 runs after episode {episode + 1}: {average_reward:.2f}")

                # Checking for the training termination criteria
                if average_reward >= 2.5:
                    print(f"Training terminated after {episode + 1} episodes as the target average reward was achieved.")
                    # Adjusting reward array length in case of early termination (easier to handle for visualizations)
                    rewards = rewards[:episode + 1]
                    break'''   # Stop training if the target reward is reached

            #print("episode: {}/{} | score: {} | e: {:.3f}".format(episode + 1, no_episodes, rew, agent.epsilon))

        print(f"Training finished. Average Reward: {np.mean(train_rewards)}")
        trained_model = self.actual_model

        return train_rewards, trained_model

    def test_DQN(self, no_episodes: int, eval: bool = False,) -> List[float]:
        test_rewards = []
        # TODO: Define which weights to load
        if not eval:
            self.actual_model.load_weights("./checkpoints/sac-actor.weights.h5")
            self.target_model.load_weights("./checkpoints/sac-actor.weights.h5")

        for episode in range(no_episodes):
            if (episode + 1) % 100 == 0:
                print(f"--- Episode:  {str(episode + 1)} / {str(no_episodes)} ---")

            current_state = self.env.flatten_obs(self.env.reset("training"))  # get starting position
            episode_rew = 0

            for step in range(self.env.episode_steps):
                #action = self.select_greedy_action(state = tf.convert_to_tensor(current_state))
                action = self.select_greedy_action(state = current_state)
                rew, next_state, d = self.env.step(action)  # observe outcome of action
                next_state = self.env.flatten_obs(next_state)  # flatten observation to 1d array
                episode_rew += rew  # increase total reward per episode

                if d:
                    break

                current_state = next_state

            test_rewards.append(episode_rew)
            print(f"Episode {episode + 1}/{no_episodes}: Total Reward = {episode_rew}")

        print(f"Testing finished. Average Reward: {np.mean(test_rewards)}")

        return test_rewards

    def run(self, no_episodes: int, mode: str = "training", render: bool = False):
        # Console Message
        print(f"Starting {mode}...")

        if mode == "training":
            rewards, trained_model = self.train_DQN(no_episodes = no_episodes, render = render)
            #ut.store_model(trained_model)

        else:
            rewards = self.test_DQN(no_episodes = no_episodes, eval = False)

        return rewards

if __name__ == "__main__":
    # Initialize Environment
    env = Environment(variant = 0, data_dir = "drl-codinglab/data/", render = False)

    # Initialize DQN Agent with Hyperparameters
    # fmt: off
    agent = DQN_Agent(env = env,
                      gamma = 0.99,
                      epsilon = 1,
                      batch_size = 128,
                      epsilon_min = 0.15,
                      epsilon_decay = 0.995,
                      learning_rate = 0.005,
                      actual_update_interval = 15,
                      target_update_interval = 60,
                      eval_interval = 50,
                      mem_maxlen = 2500)
    # fmt: on
    rewards = agent.run(no_episodes = 800, mode = "training", render = False)
    ut.rewards_plotting(rewards)