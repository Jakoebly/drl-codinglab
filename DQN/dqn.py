import tensorflow as tf
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm
from Utils import utils
from environment import Environment
from collections import deque
#from settings import *
#import Utils.hyperparameter_tuning as hp

class DQN_AGENT:
    def __init__(self, env: Environment,
                 gamma: float,
                 epsilon: float,
                 batch_size: int,
                 epsilon_min: float,
                 epsilon_decay: float,
                 learning_rate: float,
                 target_update_step : int,
                 mem_maxlen: int):
        
        self.env = env
        self.state_size = self.env.get_state_len()
        self.action_size = self.env.get_action_len()

        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = learning_rate

        self.replay_buffer = deque(maxlen=mem_maxlen)
        self.actual_model = self.nn_model(input_size = self.state_size, output_size=self.action_size, output_act_fct="softmax")
        self.target_model = self.nn_model(input_size = self.state_size, output_size=self.action_size, output_act_fct="softmax")

        # Other attributes
        self.target_update_step = target_update_step
        self.memory = deque(maxlen=2000)

    def _nn_model(self, input_size: int, output_size: int, output_act_fct: str, load_old_model = False):
        """Returns a NN with an input layer the size of `input_size`, 2 hidden layers with a RELU act. fct., and an output layer with size `output_size` and act. fct. `output_act_fct`."""
        model = keras.Sequential(
            [
                # Two hidden layers with a RELU act. fct. and 64 nodes each
                layers.Dense(128, activation="relu", input_shape=(input_size,)),  # was: 64
                layers.Dense(128, activation="relu"),  # was: 64
                # Output layer with custom size and act. fct.
                layers.Dense(output_size, activation=output_act_fct),
            ]
        )
        return model
    
    
    '''def nn_model(self, state_size, action_size, load_old_model):
        """Define the neural network."""
        inputs = tf.keras.Input(shape=(state_size,))
        prev_layer = inputs
        for no_neurons in self.nn_architecture:
            prev_layer = tf.keras.layers.Dense(no_neurons, activation='relu')(prev_layer)

        outputs = tf.keras.layers.Dense(action_size)(prev_layer)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')

        if load_old_model:
            utils.load_model_weights(WEIGHT_PATH, model)

        return model'''
    
    def _select_action(self, state: tf.Tensor, mode: str = "training"):
        if mode == "training":
            return self.select_randon_action()
        else:
            return self.select_greedy_action(state)
    
    def select_random_action(self):
        if np.random.uniform(0, 1) <= self.epsilon:
            act = random.randrange(self.action_size)
        return act
    
    def select_greedy_action(self, state):
        Q_values = self.model.predict(state)
        act = np.argmax(Q_values[0])
        return act
    
    # Selecting an action depending on the current state and our current policy
    '''def select_random_action(self, state):
        # Random chance of exploring new action instead of acting on highest Q-value
        if training and np.random.uniform(0, 1) <= self.epsilon:
            act = random.randrange(self.action_size)
        # greedy action - acting on highest Q-value    
        else:
            Q_values = self.model.predict(state)
            act = np.argmax(Q_values[0])
            print("--------------------------------------------------------")
        return act'''
    
    # Here newly observed transitions are stored in the experience replay buffer
    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    

    def update_weights(self):
        # Checking if the replay buffer is large enough to be sampled
        if len(self.memory) >= self.batch_size:
            minibatch = random.sample(self.memory, self.batch_size)
            # Extracting the individual arguments
            current_states = np.array([transition[0] for transition in minibatch])
            actions = np.array([transition[1] for transition in minibatch])
            rewards = np.array([transition[2] for transition in minibatch])
            next_states = np.array([transition[3] for transition in minibatch])
            dones = np.array([transition[4] for transition in minibatch])

            # This reshaping is necessary for the nn input!
            current_states = np.squeeze(current_states, axis=1)
            next_states = np.squeeze(next_states, axis=1)

            # Implementing the minibatch/replay buffer update algorithm as per the lecture
            Q_target = self.model.predict(current_states)
            Q_next = self.target_model.predict(next_states)
            max_Q_next = np.max(Q_next, axis=1)
            # Checking for terminal states with the "dones" flag
            targets = rewards + (1 - dones) * self.gamma * max_Q_next
            for i, action in enumerate(actions):
                Q_target[i][action] = targets[i]
            self.model.fit(current_states, Q_target, epochs=1, verbose=0)

    '''def evaluate(self):
        print(" --- EVALUATION ROUND --- ")
        total_rewards = environment.runDQN(agent, 100, eval_interval, target_reward, training=False, evaluation=True)
        average_reward = np.mean(total_rewards)
        return average_reward'''


    # updates epsilon after each episode as long as it is larger than our defined minvalue
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Function to update weights of target network - check whether there is enough samples for the minibatch
    def update_target_weights(self, t):  # t for time step at which target model will be updated
        if t % self.target_model_time == 0 and len(self.memory) >= self.batch_size:
            self.target_model.set_weights(self.model.get_weights())
            print("------ TARGET WEIGHTS UPDATED --------")
        
    def run_DQN(self, no_episodes: int, mode: str = "training", visualize_results: bool = True) -> List[float]:
        rewards = deque(maxlen=no_episodes)

                # Console Message
        print(f"Starting {mode}...")

        # TODO: Define which weights to load
        if mode == "testing":
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
                action = self._select_action(tf.convert_to_tensor(current_state))  # sample action
                rew, next_state, d = self.env.step(action)  # observe outcome of action
                next_state = self.env.flatten_obs(next_state)  # flatten observation to 1d array

                # Store Observation in Replay Buffer
                # TODO: Differentiate between training and testing on abstract level
                if mode == "training":
                    self.replay_buffer.append([current_state, action, rew, next_state, d])  # for NN update
                episode_rew += rew  # increase total reward per episode

                if d:
                    break

                # Continue With Next State
                current_state = next_state

                if mode = "training":
                    agent.record(state, action, reward, next_state, done)
                    agent.update_weights()
                    agent.update_target_weights(t)

                # Update Internal Variables
                self.update_epsilon()
                rewards.append(episode_rew)
                print(f"Step: {steps + 1}/{no_episodes} | Score: {rwd:.2f} | Epsilon: {agent.epsilon:.3f}")

if __name__ == "__main__":
    # Initialize Environment
    env = Environment(variant=0, data_dir="./data")

    # Initialize SAC Agent with Hyperparameters
    # fmt: off
    agent = DQN_Agent(env=env,
                      gamma=0.99,  # was: 0.99
                      initial_alpha=0.2,  # was: 1
                      alpha_decay=1,  # was: 0.995
                      rho=0.01,  # was: 0.2
                      mem_maxlen=int(1e7),  # was: int(1e7)
                      batch_size=64)  # was: 20
    # fmt: on

    # Perform Episode Runs
    rewards = agent.run_SAC(no_episodes=800, mode="training", visualize_results=True)
