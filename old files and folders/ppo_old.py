import random
from datetime import datetime
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
import wandb


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

        data = self.to_tensor()

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

        dataset_actor_shuffled = dataset_actor.shuffle(buffer_size=batch_size, reshuffle_each_iteration=True)
        dataset_actor_shuffled_batched = dataset_actor_shuffled.batch(minibatch_size).repeat(num_epochs)

        dataset_critic_shuffled = dataset_critic.shuffle(buffer_size=batch_size, reshuffle_each_iteration=True)
        dataset_critic_shuffled_batched = dataset_critic_shuffled.batch(minibatch_size).repeat(num_epochs)

        return dataset_actor_shuffled_batched, dataset_critic_shuffled_batched


class PPOAgent:
    def __init__(self, agent_hyperparameters, batch_size):
        # Hyperparameters
        self.nn_architecture = agent_hyperparameters["nn_architecture"]
        self.learning_rate = agent_hyperparameters["learning_rate"]
        self.gamma = agent_hyperparameters["gamma"]
        self.lam = agent_hyperparameters["lam"]
        self.clip_ratio = agent_hyperparameters["clip_ratio"]
        self.entropy_bonus = agent_hyperparameters["entropy_bonus"]
        self.entropy_bonus_min = agent_hyperparameters["entropy_bonus_min"]
        self.entropy_bonus_decay = agent_hyperparameters["entropy_bonus_decay"]


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

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='actor')
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

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='critic')
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate))

        return model

    def save_weights(self):
        actorpath = "./checkpoints/PPO/ppo_actor.weights.h5"
        criticpath = "./checkpoints/PPO/ppo_critic.weights.h5"
        self.actor.save_weights(actorpath)
        self.critic.save_weights(criticpath)
        print("saved")

    def load_weights(self):
        actorpath = "./checkpoints/PPO/ppo_actor.weights.h5"
        criticpath = "./checkpoints/PPO/ppo_critic.weights.h5"
        self.actor.load_weights(actorpath)
        self.critic.load_weights(criticpath)
        print("saved")

    # TODO: überarbeiten
    def get_action(self, state):
        probs = self.actor.predict(state.reshape(1, -1))
        print(f"Probability Output: {probs}")
        log_probs = tf.math.log(probs + 1e-8)
        # action = np.random.choice(self.num_actions, p=probs[0])
        action = np.argmax(probs)
        print(f"Selected Action: {action}")
        return action, log_probs[0, action]

    # TODO: überarbeiten
    def get_value(self, state):
        value = self.critic.predict(state.reshape(1, -1))
        return value[0][0]


    # TODO: ChatGPT fragen wie tf.function compatible
    def calc_advantage(self, values, next_values, rewards, dones):

        advantages = np.zeros_like(values)
        returns = np.zeros_like(values)
        advantage = 0

        for t in reversed(range(len(values))):
            mask = 1
            if dones[t]:
                mask = 0

            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            advantage = delta + self.gamma * self.lam * mask * advantage

            advantages[t] = advantage
            returns[t] = advantage + values[t]

        # Normalize advantages to have zero mean and unit variance
        print(f"Adv: {advantages}")
        advantage_mean = np.mean(advantages)
        advantage_std = np.std(advantages) + 1e-8
        advantages = (advantages - advantage_mean) / advantage_std
        print(f"Norm. Adv.: {advantages}")

        return advantages, returns

    # @tf.function
    def update_actor(self, data, episode):
        for mini_batch in data:

            states, actions, old_log_probs, advantages = mini_batch['states'], mini_batch['actions'], mini_batch['log_probs'], mini_batch['advantages']

            with (tf.GradientTape() as tape):
                new_probs = self.actor(states)
                dist = tfp.distributions.Categorical(new_probs)
                new_log_probs = dist.log_prob(actions)

                prob_ratio = tf.math.exp(new_log_probs - old_log_probs)

                clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

                entropy = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-8), axis=1)
                entropy = tf.reduce_mean(entropy)

                actor_loss = -(tf.reduce_mean(tf.minimum(prob_ratio * advantages, clipped_prob_ratio * advantages)) + self.entropy_bonus * entropy)

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)

            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            wandb_run.log({"Losses/Actor Loss": actor_loss}, episode)
            wandb_run.log({"Losses/Entropy Loss": entropy}, episode)

        print("Updated Actor")

    # @tf.function
    def update_critic(self, data, episode):
        for mini_batch in data:

            states, returns = mini_batch['states'], mini_batch['returns']

            with tf.GradientTape() as tape:
                values = self.critic(states)
                critic_loss = tf.reduce_mean(tf.square(values - returns))


            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)

            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

            wandb_run.log({"Losses/Critic Loss": critic_loss}, episode)
        print("Updated Critic")


class GymEnvironment:
    def __init__(self, variant, data_dir, agent_hyperparameters, mode):
        self.env = Environment(variant, data_dir, render=False)
        self.mode = mode

        self.num_episodes = agent_hyperparameters["num_episodes"]
        self.num_trajectories = agent_hyperparameters["num_trajectories"]
        self.num_updates = agent_hyperparameters["num_updates"]
        self.num_steps_trajectory = agent_hyperparameters["num_steps_trajectory"]
        self.batch_size = self.num_trajectories * self.num_steps_trajectory

        self.minibatch_size = self.batch_size // agent_hyperparameters["num_minibatches"]
        self.num_epochs = agent_hyperparameters["num_epochs"]

    '''def flatten_state(self, state):
        agent_pos, free_capacity, item_locs, item_times = state
        flattened = [agent_pos[0], agent_pos[1], free_capacity, *sum(item_locs, ()), *item_times]
        return np.array(flattened)'''

    '''def reset(self):
        return self.flatten_state(self.env.reset("training"))'''

    '''def step(self, action):
        reward, next_state, done = self.env.step(action)
        return self.env.flatten_obs(next_state), reward, done'''

    def runPPO(self, agent, training=True):

        # agent.load_weights()

        rew_episode = []
        best = -1000
        update = 1

        for step in range(1, self.num_episodes + 1):

            print("-" * 200)
            print(f"EPISODE: {step} / {self.num_episodes}")
            print("-" * 200)

            rew_trajectory = 0

            state = self.env.reset(self.mode)

            for t in range(self.num_steps_trajectory):

                print("-" * 10, f" Step: {t} / {self.num_steps_trajectory} ", "-" * 10)

                action, log_prob = agent.get_action(state)
                reward, next_state, done = self.env.step(action)
                value = agent.get_value(state)
                next_value = agent.get_value(next_state)
                rew_trajectory += reward

                if training:
                    agent.memory.store(state, action, reward, next_state, done, value, next_value, log_prob)

                state = next_state


                # self.env.renderer.render()

            if training and step % self.num_trajectories == 0:

                print("starting update")

                # Learning Rate Annealing
                #frac = 1.0 - (update - 1.0) / self.num_updates
                #lr_new = frac * agent.learning_rate
                #agent.learning_rate = lr_new
                #agent.actor.optimizer.learning_rate = lr_new
                #agent.critic.optimizer.learning_rate = lr_new
                #update += 1

                advantages, returns = agent.calc_advantage(agent.memory.values, agent.memory.nextvalues, agent.memory.rewards, agent.memory.dones)
                agent.memory.advantages.extend(advantages)
                agent.memory.returns.extend(returns)

                dataset_actor, dataset_critic = agent.memory.generate_batches(self.batch_size, self.minibatch_size, self.num_epochs)

                agent.update_actor(dataset_actor, step)
                agent.update_critic(dataset_critic, step)

                # Update entropy bonus
                # agent.entropy_bonus = max(agent.entropy_bonus_min, agent.entropy_bonus * agent.entropy_bonus_decay)

                agent.memory.clear()

            rew_episode.append(rew_trajectory)

            wandb_run.log({"Charts/Reward per Episode": rew_trajectory})
        return rew_episode



if __name__ == "__main__":

    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    num_episodes = 800
    num_trajectories = 5
    num_steps_trajectory = 200
    num_minibatches = 60
    num_epochs = 20
    num_updates = num_episodes // num_trajectories

    nn_architecture = [128, 128, 128]
    state_size = 161
    num_actions = 5
    learning_rate = 0.00015
    gamma = 0.99
    lam = 0.95
    clip_ratio = 0.1
    entropy_bonus = 0.5
    entropy_bonus_min = 0.0001
    entropy_bonus_decay = (entropy_bonus - entropy_bonus_min) ** (1 / num_updates)

    agent_hyperparameters = {
        "nn_architecture": nn_architecture,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "lam": lam,
        "clip_ratio": clip_ratio,
        "entropy_bonus": entropy_bonus,
        "entropy_bonus_min": entropy_bonus_min,
        "entropy_bonus_decay": entropy_bonus_decay,
        "state_size": state_size,
        "num_actions": num_actions,
        "num_episodes": num_episodes,
        "num_trajectories": num_trajectories,
        "num_steps_trajectory": num_steps_trajectory,
        "num_minibatches": num_minibatches,
        "num_epochs": num_epochs,
        "num_updates": num_updates
    }

    data_dir = "../data"
    exp_name = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"CL__PPO__Var0__{exp_name}"

    wandb_run = wandb.init(
        entity="gowiththetensorflow",
        project="Coding_Lab",
        config=agent_hyperparameters,
        name=run_name,
        save_code=True,
        group="PPO",
        tags=["0"],
        job_type="train/validate"
    )

    GymEnvironment = GymEnvironment(variant=0, data_dir=data_dir, agent_hyperparameters=agent_hyperparameters, mode="training")

    agent = PPOAgent(agent_hyperparameters, GymEnvironment.batch_size)

    rewards = GymEnvironment.runPPO(agent, training=True)


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
