import random
import numpy as np
import pandas as pd
import tensorflow as tf
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
        self.log_probs = []
        self.advantages = []
        self.returns = []

        self.batch_size = batch_size

    def store(self, state, action, reward, next_state, done, value, next_value, log_probs):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.values.append(value)
        self.nextvalues.append(next_value)
        self.log_probs.append(log_probs)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.values.clear()
        self.nextvalues.clear()
        self.log_probs.clear()
        self.advantages.clear()
        self.returns.clear()

    def to_tensor(self):
        return {
            'states': tf.convert_to_tensor(self.states, dtype=tf.float32),
            'actions': tf.convert_to_tensor(self.actions, dtype=tf.int32),
            'rewards': tf.convert_to_tensor(self.rewards, dtype=tf.float32),
            'next_states': tf.convert_to_tensor(self.next_states, dtype=tf.float32),
            'dones': tf.convert_to_tensor(self.dones, dtype=tf.float32),
            'values': tf.convert_to_tensor(self.values, dtype=tf.float32),
            'next_values': tf.convert_to_tensor(self.nextvalues, dtype=tf.float32),
            'log_probs': tf.convert_to_tensor(self.log_probs, dtype=tf.float32),
            'advantages': tf.convert_to_tensor(self.advantages, dtype=tf.float32),
            'returns': tf.convert_to_tensor(self.returns, dtype=tf.float32)
        }

    def generate_batches(self, batch_size, minibatch_size, num_epochs):

        # Convert data to tensors
        data = self.to_tensor()

        #  Extract data for actor data set
        dataset_actor = tf.data.Dataset.from_tensor_slices({
            'states': data['states'],
            'actions': data['actions'],
            'log_probs': data['log_probs'],
            'advantages': data['advantages']
        })

        # Extract data for critic data set
        dataset_critic = tf.data.Dataset.from_tensor_slices({
            'states': data['states'],
            'returns': data['returns']
        })

        # Create shuffled and batched datasets for stochastic mini-batch gradient descent
        dataset_actor_shuffled = dataset_actor.shuffle(buffer_size=batch_size, reshuffle_each_iteration=True)
        dataset_actor_shuffled_batched = dataset_actor_shuffled.batch(minibatch_size).repeat(num_epochs)

        dataset_critic_shuffled = dataset_critic.shuffle(buffer_size=batch_size, reshuffle_each_iteration=True)
        dataset_critic_shuffled_batched = dataset_critic_shuffled.batch(minibatch_size).repeat(num_epochs)

        return dataset_actor_shuffled_batched, dataset_critic_shuffled_batched

class PPO_Agent:
    def __init__(self, environment, agent_hyperparameters, training_hyperparameters):

        # Environment
        self.env = environment

        # Agent Hyperparameters
        self.nn_architecture = agent_hyperparameters["nn_architecture"]
        self.state_size = agent_hyperparameters["state_size"]
        self.num_actions = agent_hyperparameters["num_actions"]

        self.learning_rate = agent_hyperparameters["learning_rate"]
        self.gamma = agent_hyperparameters["gamma"]
        self.lam = agent_hyperparameters["lam"]
        self.clip_ratio = agent_hyperparameters["clip_ratio"]
        self.entropy_bonus = agent_hyperparameters["entropy_bonus"]
        self.epsilon = tf.Variable(initial_value=agent_hyperparameters["epsilon"], dtype = tf.float32, trainable = False)
        self.epsilon_min = agent_hyperparameters["epsilon_min"]
        self.epsilon_decay = agent_hyperparameters["epsilon_decay"]

        # Training Hyperparameters
        self.num_episodes = training_hyperparameters["num_episodes"]
        self.num_trajectories = training_hyperparameters["num_trajectories"]
        self.num_steps_trajectory = training_hyperparameters["num_steps_trajectory"]
        self.batch_size = self.num_trajectories * self.num_steps_trajectory

        self.minibatch_size = self.batch_size // training_hyperparameters["num_minibatches"]
        self.num_epochs = training_hyperparameters["num_epochs"]

        # Memory Initialization
        self.memory = PPOMemory(self.batch_size)

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

    def get_action_epsgreedy(self, state):
        state = tf.reshape(state, (1, -1))

        # Get action probabilities using the actor model
        probs = self.actor(state, training=False)

        # Convert probabilities to log probabilities
        log_probs = tf.math.log(probs + 1e-8)

        # Sample an action
        if tf.random.uniform(()) < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = tf.argmax(probs, axis=1)

        return action, tf.gather(log_probs[0], action)


    def get_action(self, state):
        # Reshape state
        state = tf.reshape(state, (1, -1))

        # Get action probabilities using the actor model
        probs = self.actor(state, training=False)

        # Convert probabilities to log probabilities
        log_probs = tf.math.log(probs + 1e-8)

        # Sample an action based on probabilities
        action = tf.random.categorical(probs, 1)
        action = tf.squeeze(action, axis=-1)
        # action = tf.math.argmax(probs, axis=1)

        # Return sampled action and its log probability
        return action, tf.gather(log_probs[0], action)

    # TODO: überarbeiten
    def get_value(self, state):
        # Reshape state
        state = tf.reshape(state, (1, -1))

        # Predict the value using the critic model
        value = self.critic(state, training=False)

        # Extract the scalar value
        return tf.squeeze(value, axis=-1)

    '''def calc_advantage_old(self, values, next_values, rewards, dones):

        # Create empty tensors for value storage
        advantages = tf.zeros_like(values)
        returns = tf.zeros_like(values)
        advantage = tf.zeros_like(values[0])

        # Iterate backwards through the dataset
        for t in tf.range(len(values) - 1, -1, -1):
            # Set mask = 0 if done = 1 and mask = 0 otherwise
            mask = tf.cast(1 - dones[t], tf.float32)

            # Compute deltas and advantages using the mask to ensure no TD-learning at last step
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            advantage = delta + self.gamma * self.lam * mask * advantage


            advantages = tf.tensor_scatter_nd_update(advantages, [[t]], [advantage])
            returns = tf.tensor_scatter_nd_update(returns, [[t]], [advantage + values[t]])

        return advantages, returns'''

    def calc_advantage(self, values, next_values, rewards, dones):

        # Create TensorArray for storing advantages and returns
        advantages = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        returns = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        advantage = tf.zeros_like(values[0])  # Initialize advantage to zero

        # Loop through trajectories in reverse order
        for t in range(len(values) - 1, -1, -1):

            # Set mask = 0 if done = 1 and mask = 0 otherwise and convert to float32
            mask = tf.cast(1 - dones[t], tf.float32)

            # Calculate TD error
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]

            # Update advantage via GAE
            advantage = delta + self.gamma * self.lam * mask * advantage

            # Update corresponding TensorArrays
            advantages = advantages.write(t, advantage)
            returns = returns.write(t, advantage + values[t])

        # Convert TensorArray to tensors after the loop
        advantages = advantages.stack()
        returns = returns.stack()

        # Normalize advantages to have zero mean and unit variance
        # advantage_mean = tf.reduce_mean(advantages)
        # advantage_std = tf.math.reduce_std(advantages) + 1e-8  # Add epsilon for numerical stability
        # advantages = (advantages - advantage_mean) / advantage_std

        return advantages, returns

    @tf.function
    def update_actor(self, data):
        # Iterate over mini-batches
        for mini_batch in data:
            states = mini_batch['states']
            actions = mini_batch['actions']
            old_log_probs = mini_batch['log_probs']
            advantages = mini_batch['advantages']

            # Use GradientTape for computing gradients
            with tf.GradientTape() as tape:
                new_probs = self.actor(states)
                dist = tfp.distributions.Categorical(new_probs)
                new_log_probs = dist.log_prob(actions)

                # Compute probability ratio and clipped ratio
                prob_ratio = tf.math.exp(new_log_probs - old_log_probs)
                clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

                # Compute entropy
                entropy = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-8), axis=1)
                entropy = tf.reduce_mean(entropy)

                # Compute actor loss
                actor_loss = -(tf.reduce_mean(tf.minimum(prob_ratio * advantages,
                                                         clipped_prob_ratio * advantages)) + self.entropy_bonus * entropy)

            # Compute gradients and apply them
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # tf.print("Updated Actor")

    @tf.function
    def update_critic(self, data):
        # Iterate over mini-batches
        for mini_batch in data:
            states = mini_batch['states']
            returns = mini_batch['returns']

            # Use GradientTape for computing gradients
            with tf.GradientTape() as tape:
                values = self.critic(states)
                critic_loss = tf.reduce_mean(tf.square(values - returns))

            # Compute gradients and apply them
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # tf.print("Updated Critic")

    def update_epsilon(self):
        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon.assign(tf.maximum(self.epsilon_min, self.epsilon * self.epsilon_decay))

    def run(self, mode, visualize_results, render):

        # Reward per episode
        rew_episode = []

        # Iterate over all episodes
        for step in range(1, self.num_episodes + 1):

            # Reward per trajectory
            rew_trajectory = 0

            # Reset environment
            state = self.env.reset(mode)

            # Iterate over the maximum number of time steps per trajectory
            for t in range(self.num_steps_trajectory):

                # tf.print("-" * 10, f" Step: {t} / {self.num_steps_trajectory} ", "-" * 10)

                # Execute transition and update reward
                action, log_prob = self.get_action(state)
                reward, next_state, done = self.env.step(action)
                value = self.get_value(state)
                next_value = self.get_value(next_state)
                rew_trajectory += reward

                # Store transition if in training
                if mode == "training":
                    self.memory.store(state, action, reward, next_state, done, value, next_value, log_prob)

                # Update state for next iteration
                state = next_state

                # Update the renderer
                if render:
                    self.env.render()

            # If in training and current episode is an update episode
            if mode == "training" and step % self.num_trajectories == 0:

                # Compute advantages and returns and store them
                advantages, returns = self.calc_advantage(self.memory.values, self.memory.nextvalues, self.memory.rewards, agent.memory.dones)
                self.memory.advantages.extend(advantages)
                self.memory.returns.extend(returns)

                # Generate shuffled and batched datasets for training based on stored transitions
                dataset_actor, dataset_critic = self.memory.generate_batches(self.batch_size, self.minibatch_size, self.num_epochs)

                # Update networks
                self.update_actor(dataset_actor)
                self.update_critic(dataset_critic)

                # Update epsilon
                self.update_epsilon()

                # Clear memory
                self.memory.clear()

            # Append total trajectory reward
            rew_episode.append(rew_trajectory)

            tf.print(f"EPISODE: {step} / {self.num_episodes}: Reward: {rew_episode[step-1]}")

        return rew_episode


if __name__ == "__main__":

    # Set seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Set data directory and variant
    data_dir = "../data"  # DONE: specify relative path to data directory (e.g., './data', not './data/variant_0')
    variant = 0  # DONE: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)

    render = False

    # Initialize environment
    env = Environment(variant, data_dir, render)

    # Set hyperparameters
    nn_architecture = [128, 128, 128]
    state_size = 159 # env.get_state_len()
    num_actions = env.get_action_len()
    learning_rate = 0.0001
    gamma = 0.99
    lam = 0.95
    clip_ratio = 0.2
    entropy_bonus = 0.5
    epsilon = 1
    epsilon_min = 0.1
    epsilon_decay = 0.997

    num_episodes = 800
    num_trajectories = 3
    num_steps_trajectory = 200
    num_minibatches = 40
    num_epochs = 10

    # Create dictionary for agent hyperparameters
    agent_hyperparameters = {
        "nn_architecture": nn_architecture,
        "state_size": state_size,
        "num_actions": num_actions,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "lam": lam,
        "clip_ratio": clip_ratio,
        "entropy_bonus": entropy_bonus,
        "epsilon": epsilon,
        "epsilon_decay": epsilon_decay,
        "epsilon_min": epsilon_min
    }

    # Create dictionary for training hyperparameters
    training_hyperparameters = {
        "num_episodes": num_episodes,
        "num_trajectories": num_trajectories,
        "num_steps_trajectory": num_steps_trajectory,
        "num_minibatches": num_minibatches,
        "num_epochs": num_epochs,
    }

    # Initialize agent
    agent = PPO_Agent(env, agent_hyperparameters, training_hyperparameters)

    # Execute training
    rewards = agent.run("training", False, render)


    print(f"Average episode reward: {np.average(rewards)}")
    print(f"Maximum episode reward: {np.max(rewards)}")

    # Create Time-Series Line Chart
    df = pd.DataFrame(np.array(rewards), columns=["Rewards"])
    plt.figure(figsize=(10, 6))
    plt.plot(df["Rewards"], label="Episode Data", color="#0066BD")
    plt.title("PPO Performance")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()















