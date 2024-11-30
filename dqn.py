import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import random
#from Utils import utils
from environment import Environment
from collections import deque

from typing import List, Tuple

class DQN_Agent:
    def __init__(self, env: Environment,
                 gamma: float,
                 epsilon: float,
                 batch_size: int,
                 epsilon_min: float,
                 epsilon_decay: float,
                 learning_rate: float,
                 target_update_interval : int,
                 eval_interval: int,
                 mem_maxlen: int):
        
        self.env = env
        self.state_size = self.env.get_state_len()
        self.action_size = self.env.get_action_len()

        self.gamma = gamma
        #self.epsilon = epsilon
        self.epsilon = tf.Variable(initial_value=epsilon, dtype = tf.float32, trainable = False)
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate

        self.replay_buffer = deque(maxlen=mem_maxlen)
        #self.actual_model = self._nn_model(input_size = self.state_size, output_size=self.action_size, output_act_fct = "softmax")
        #self.target_model = self._nn_model(input_size = self.state_size, output_size=self.action_size, output_act_fct = "softmax")
        self.actual_model = self._build_model()
        self.target_model = self._build_model()
        #self.update_target_weights()

        # Other attributes
        self.target_update_interval = target_update_interval
        self.eval_interval = eval_interval

        # Optimizer and Loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    '''def _nn_model(self, input_size: int, output_size: int, output_act_fct: str, load_old_model = False):
        """Returns a NN with an input layer the size of `input_size`, 2 hidden layers with a RELU act. fct., and an output layer with size `output_size` and act. fct. `output_act_fct`."""
        model = keras.Sequential(
            [
                # Two hidden layers with a RELU act. fct. and 64 nodes each
                layers.Dense(128, activation="relu", input_shape=(input_size,)),  # was: 64
                layers.Dense(128, activation="relu"),  # was: 64
                # Output layer with custom size and act. fct.
                layers.Dense(output_size, activation = output_act_fct),
            ]
        )
        return model'''
    
    def _nn_model(self, input_size: int, output_size: int, output_act_fct: str, load_old_model=False):
        """Returns a NN with input layer of size `input_size`, 2 hidden layers, and an output layer."""
        model = keras.Sequential(
            [
                keras.Input(shape=(input_size,)),  # Input layer
                layers.Dense(128, activation="relu"),  # Hidden layer 1
                layers.Dense(128, activation="relu"),  # Hidden layer 2
                layers.Dense(output_size, activation=output_act_fct),  # Output layer
            ]
        )
        # Compile the model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr), loss="mse")
        return model

    def _build_model(self):
        """Builds the Q-network."""
        model = tf.keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(self.action_size, activation="linear")
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')
        return model
    
    
    '''def nn_model(self, state_size, action_size, load_old_model):
        """Define the neural network."""
        inputs = tf.keras.Input(shape=(state_size,))
        prev_layer = inputs
        for no_neurons in self.nn_architecture:
            prev_layer = tf.keras.layers.Dense(no_neurons, activation='relu')(prev_layer)

        outputs = tf.keras.layers.Dense(action_size)(prev_layer)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr), loss = 'mse')

        if load_old_model:
            utils.load_model_weights(WEIGHT_PATH, model)
        return model'''
    
    '''def select_random_action(self, state = None):
        if tf.random.uniform(()) < self.epsilon.numpy():
            return random.randint(0, self.action_size - 1)
        else:
            return self.select_greedy_action(state)'''
    
    # Function to update weights of target network - check whether there is enough samples for the minibatch
    def update_target_weights(self):
        #if t % self.target_model_time == 0 and len(self.memory) >= self.batch_size:
        self.target_model.set_weights(self.actual_model.get_weights())
        #print("Updated Target Network")
        
    def select_random_action(self, state=None):
        """Select an action using epsilon-greedy policy."""
        if tf.random.uniform(()) < self.epsilon.numpy():
            # Use TensorFlow's random function
            #return tf.random.uniform(shape=(), maxval=self.action_size, dtype=tf.int32)
            return random.randint(0, self.action_size - 1)
        else:
            return self.select_greedy_action(state)

    def select_greedy_action(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        q_values = self.actual_model(state, training=False)
        return tf.argmax(q_values[0]).numpy()

    
    # Here newly observed transitions are stored in the experience replay buffer
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
    
    @tf.function
    def compute_q_targets(self, rewards, dones, max_q_next):
        """Compute the target Q-values for the Bellman equation."""
        return rewards + (1 - dones) * self.gamma * max_q_next

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


    # updates epsilon after each episode as long as it is larger than our defined minvalue
    @tf.function
    def update_epsilon(self):
        #self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon.assign(tf.maximum(self.epsilon_min, self.epsilon * self.epsilon_decay))





    '''def _sample_observation(self) -> Tuple[tf.Tensor]:
        """Randomly samples an observation from the replay buffer. Returns `(current_states, actions, rew, next_states, d)`, all of them being tensors."""
        # Randomly Sample from Replay Buffer
        batch = random.choices(self.replay_buffer, k = self.batch_size)
        # Group Features and Convert Them to Tensors
        current_states = tf.convert_to_tensor([batch[i][0] for i in range(len(batch))])
        actions = tf.convert_to_tensor([batch[i][1] for i in range(len(batch))])
        rew = tf.convert_to_tensor([batch[i][2] for i in range(len(batch))])
        next_states = tf.convert_to_tensor([batch[i][3] for i in range(len(batch))])
        d = tf.convert_to_tensor([batch[i][4] for i in range(len(batch))])

        return (current_states, actions, rew, next_states, d)'''
    
    '''@tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """Perform a single training step."""
        with tf.GradientTape() as tape:
            Q_values = self.actual_model(states, training=True)
            Q_target = tf.identity(Q_values)
            max_Q_next = tf.reduce_max(self.target_model(next_states, training=False), axis=1)
            targets = self.compute_targets(rewards, dones, max_Q_next)

            # Update Q-values for the actions taken
            indices = tf.stack([tf.range(self.batch_size), actions], axis=1)
            Q_target = tf.tensor_scatter_nd_update(Q_target, indices, targets)

            loss = self.loss_fn(Q_values, Q_target)

        # Apply gradients
        grads = tape.gradient(loss, self.actual_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actual_model.trainable_variables))
        return loss'''

    '''def update_actual_weights(self):
        # Checking if the replay buffer is large enough to be sampled
        if len(self.replay_buffer) >= self.batch_size:
            current_states, actions, rewards, next_states, dones = self._sample_observation()

            # Ensure all tensors have compatible types
            rewards = tf.cast(rewards, dtype=tf.float32)
            dones = tf.cast(dones, dtype=tf.float32)

            # This reshaping is necessary for the nn input!
            #current_states = np.squeeze(current_states, axis=1)
            #next_states = np.squeeze(next_states, axis=1)

            # Implementing the minibatch/replay buffer update algorithm as per the lecture
            Q_target = self.actual_model.predict(current_states)
            Q_next = self.target_model.predict(next_states)
            #max_Q_next = np.max(Q_next, axis=1)
            max_Q_next = tf.reduce_max(Q_next, axis=1)  # Ensure this is float32
            max_Q_next = tf.cast(max_Q_next, dtype=tf.float32)

            # Checking for terminal states with the "dones" flag
            #targets = rewards + (1 - dones) * self.gamma * max_Q_next
            targets = rewards + (1 - dones) * tf.constant(self.gamma, dtype=tf.float32) * max_Q_next

            for i, action in enumerate(actions):
                Q_target[i][action] = targets[i]
            self.actual_model.fit(current_states, Q_target, epochs=1, verbose=0)'''


    def train_DQN(self, no_episodes: int, mode, visualize_results: bool = True):
        """Train the agent."""
        #rewards = deque(maxlen = no_episodes)
        train_rewards = []

        for episode in range(no_episodes):
            # Console Message
            if (episode + 1) % 100 == 0:
                print("Episode: " + str(episode + 1) + " / " + str(no_episodes))

            current_state = self.env.flatten_obs(self.env.reset(mode))  # get starting position
            episode_rew = 0

            for step in range(self.env.episode_steps):
                # Choose and Perform Action
                #action = self.select_random_action(state = tf.convert_to_tensor(current_state))  # sample action
                action = self.select_random_action(state = current_state)  # sample action
                rew, next_state, d = self.env.step(action)  # observe outcome of action
                next_state = self.env.flatten_obs(next_state)  # flatten observation to 1d array

                episode_rew += rew  # increase total reward per episode

                #self.replay_buffer.append([current_state, action, rew, next_state, d]) # Store Observation in Replay Buffer for NN update
                self.record(current_state, action, rew, next_state, d)

                #self.update_actual_weights()  # Updating the weights using a replay buffer
                # Perform training step
                if len(self.replay_buffer) >= self.batch_size:
                    states, actions, rewards, next_states, dones = self.sample_batch()
                    self.train_step(states, actions, rewards, next_states, dones)

                if step % self.target_update_interval == 0:
                    self.update_target_weights()  # Updating the weights of the target network every t episodes

                if d:
                    break
                
                # Continue With Next State
                current_state = next_state

            # Update Internal Variables
            self.update_epsilon()
            train_rewards.append(episode_rew)
            #print(f"Step: {episode + 1}/{no_episodes} | Score: {episode_rew:.2f} | Epsilon: {self.epsilon:.3f}")
            print(f"Episode {episode + 1}/{no_episodes}: Total Reward = {episode_rew}")

            # Function that evaluates the agent's performance for every x episodes
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

        return rewards, self.replay_buffer


    def test_DQN(self, no_episodes: int, mode, eval: bool = False, visualize_results: bool = True) -> List[float]:
        rewards = deque(maxlen = no_episodes)
        # TODO: Define which weights to load
        if not eval:
            self.actual_model.load_weights("./checkpoints/sac-actor.weights.h5")
            self.target_model.load_weights("./checkpoints/sac-actor.weights.h5")

        for episode in range(no_episodes):
            # Console Message
            if (episode + 1) % 100 == 0:
                print("Episode: " + str(episode + 1) + " / " + str(no_episodes))

            current_state = self.env.flatten_obs(self.env.reset(mode))  # get starting position
            episode_rew = 0

            for step in range(self.env.episode_steps):
                # Choose and Perform Action
                #action = self.select_greedy_action(state = tf.convert_to_tensor(current_state))
                action = self.select_greedy_action(state = current_state)
                rew, next_state, d = self.env.step(action)  # observe outcome of action
                next_state = self.env.flatten_obs(next_state)  # flatten observation to 1d array
                episode_rew += rew  # increase total reward per episode
                current_state = next_state

                if d:
                    break

            rewards.append(episode_rew)
            print(f"Step: {episode + 1}/{no_episodes} | Score: {episode_rew:.2f} | Epsilon: {self.epsilon:.3f}")

        return rewards

    def run(self, no_episodes: int, mode: str = "training", visualize_results: bool = True):
        # Console Message
        print(f"Starting {mode}...")

        if mode == "training":
            rewards, replay_buffer = self.train_DQN(no_episodes = no_episodes, mode=mode,visualize_results=visualize_results)

        else:
            rewards = self.test_DQN(no_episodes = no_episodes, visualize_results=visualize_results)

        return rewards


if __name__ == "__main__":
    # Initialize Environment
    env = Environment(variant = 0, data_dir = "drl-codinglab/data/")

    # Initialize SAC Agent with Hyperparameters
    # fmt: off
    agent = DQN_Agent(env = env,
                      gamma = 0.95,
                      epsilon = 1,
                      batch_size = 128,
                      epsilon_min = 0.1,
                      epsilon_decay = 0.999,
                      learning_rate = 0.01,
                      target_update_interval = 50,
                      eval_interval = 50,
                      mem_maxlen = 2000)
    # fmt: on
    rewards = agent.run(no_episodes = 1000, mode = "training", visualize_results = True)
    print(rewards)