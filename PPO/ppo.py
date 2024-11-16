import numpy as np
import scipy.signal
import tensorflow as tf
from keras import layers, optimizers
from dataclasses import dataclass
from typing import List
from environment import Environment
import Utils.utils as ut

@dataclass
class Transition:
    state: np.array
    action: int
    reward: float
    next_state: np.array
    is_done: bool
    logits: np.array
    value: float

class GymEnvironment:
    def __init__(self, variant, data_dir):
        self.env = Environment(variant, data_dir)
        self.max_timesteps = self.env.episode_steps
        self.state_visit_count = np.zeros((self.env.vertical_cell_count, self.env.horizontal_cell_count))

    def flatten_state(self, state):
        agent_pos, free_capacity, item_locs, item_times = state
        flattened = [
            agent_pos[0], agent_pos[1], free_capacity, *sum(item_locs, ()), *item_times
        ]
        return np.array(flattened)

    def reset(self):
        return self.flatten_state(self.env.reset("training"))

    def step(self, action):
        reward, next_state, done = self.env.step(action)
        return self.flatten_state(next_state), reward, done

    def trainPPO(self, agent, no_episodes):
        return self.runPPO(agent, no_episodes, training=True)

    def runPPO(self, agent, no_episodes, training=False):
        rew_total = []

        for episode in range(no_episodes):
            rew_episode = []

            next_state = self.reset()
            rew_agent = 0
            trajectory: List[Transition] = []

            for t in range(self.max_timesteps):
                state = next_state
                logit, action = agent.select_action(state)
                next_state, reward, done = self.step(action)
                rew_agent += reward

                # Tracking state visits for heatmap
                agent_pos = state[:2]  # Extract agent position
                self.state_visit_count[tuple(agent_pos)] += 1

                # Collect transitions if training
                if training:
                    transition = Transition(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        is_done=done,
                        logits=logit,
                        value=agent.calc_value(state)
                    )
                    trajectory.append(transition)

                if done or t == self.max_timesteps - 1:
                    rew_episode.append(rew_agent)
                    if training:
                        states, actions, rewards, next_states, is_dones, logits, values = zip(
                            *[(t.state, t.action, t.reward, t.next_state, t.is_done, t.logits, t.value) for t in trajectory]
                        )

                        states_array = np.array(states)
                        actions_array = np.array(actions)
                        rewards_array = np.array(rewards)
                        next_states_array = np.array(next_states)
                        logits_array = np.array(logits)
                        values_array = np.array(values)

                        advantage_est = agent.calc_advantage(next_state, rewards_array, values_array, done, t)
                        terminal_value = agent.calc_value(next_state) if not done else 0
                        target_input = np.append(rewards_array, terminal_value)
                        targets = agent.calc_targets(target_input)[:-1]

                        log_probs = [policy_probabilities(logits_array[i], actions_array[i]) for i in range(len(actions_array))]
                        agent.memory.store_many(states_array, actions_array, rewards_array, log_probs, values_array, next_states_array, advantage_est, targets)
                    break

            rew_total.append(rew_episode)
            if training:
                tf.print('-' * 200)
                tf.print(f'Episode {episode} of {no_episodes}')
                tf.print('-' * 70)
                tf.print(f'Total Rewards (per trajectory): {rew_episode}')

                data = agent.memory.to_tensor()
                data_size = agent.memory.size()
                batch_size = min(agent.batch_size, data_size)

                dataset_policy = tf.data.Dataset.from_tensor_slices({
                    'states': data['states'],
                    'actions': data['actions'],
                    'log_probs': data['log_probs'],
                    'advantages': data['advantages']
                })

                dataset_values = tf.data.Dataset.from_tensor_slices({
                    'states': data['states'],
                    'targets': data['targets']
                })

                shuffled_dataset_policy = dataset_policy.shuffle(buffer_size=data_size, reshuffle_each_iteration=True)
                shuffled_batched_dataset_policy = shuffled_dataset_policy.batch(batch_size).repeat(agent.train_actor_iterations)

                shuffled_dataset_value = dataset_values.shuffle(buffer_size=data_size, reshuffle_each_iteration=True)
                shuffled_batched_dataset_value = shuffled_dataset_value.batch(batch_size).repeat(agent.train_critic_iterations)

                agent.update_policy_parameters(shuffled_batched_dataset_policy)
                tf.print(f'Updated policy function.')

                agent.update_value_parameters(shuffled_batched_dataset_value)
                tf.print(f'Updated value function')

                agent.memory.clear()
                tf.print(f'Reset agent memory.')
                tf.print('-' * 200)

        return rew_total

def policy_probabilities(logit, action):
    probabilities = tf.nn.softmax(logit)
    log_probs = tf.math.log(probabilities + 1e-8)
    return log_probs[action]

def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPO_Agent:
    def __init__(self, agent_hyperparameters):
        self.memory = Memory()

        self.seed = agent_hyperparameters["seed"]
        self.state_size = 75  # Updated input size
        self.action_size = agent_hyperparameters["no_of_actions"]
        self.gamma = agent_hyperparameters["gamma"]
        self.lam = agent_hyperparameters["lam"]
        self.clip_ratio = agent_hyperparameters["clip_ratio"]
        self.actors = agent_hyperparameters["actors"]
        self.learning_rate = agent_hyperparameters["learning_rate"]
        self.batch_size = agent_hyperparameters["batch_size"]
        self.entropy_bonus = agent_hyperparameters["entropy_bonus"]
        self.train_actor_iterations = agent_hyperparameters["train_actor_iterations"]
        self.train_critic_iterations = agent_hyperparameters["train_critic_iterations"]
        self.nn_architecture = agent_hyperparameters["nn_architecture"]

        self.actor = self.nn_model(self.state_size, self.action_size)
        self.critic = self.nn_model(self.state_size, 1)

        self.actor_optimizer = self.actor.optimizerder
        self.critic_optimizer = self.critic.optimizer

    def select_action(self, state):
        logits = self.actor.predict(state.reshape(1, -1)).flatten()
        action = np.random.choice(5, p=tf.nn.softmax(logits).numpy())
        return logits, action

    def calc_advantage(self, state, rewards, values, done, t):
        deltas = rewards + self.gamma * np.roll(values, -1) - values
        terminal_value = self.calc_value(state) if not done else 0
        deltas[-1] = rewards[-1] + self.gamma * terminal_value - values[-1]
        return discounted_cumulative_sums(deltas, self.gamma * self.lam)

    def calc_targets(self, rewards):
        return discounted_cumulative_sums(rewards, self.gamma)

    def nn_model(self, state_size, output_size):
        inputs = tf.keras.Input(shape=(state_size,))
        prev_layer = inputs
        for no_neurons in self.nn_architecture:
            prev_layer = layers.Dense(no_neurons, activation='relu')(prev_layer)
        outputs = layers.Dense(output_size)(prev_layer)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def calc_value(self, state):
        val = self.critic.predict(state.reshape(1, -1))
        return val[0][0]

    @tf.function
    def update_policy_parameters(self, data):
        for batch in data:
            states, actions, log_probs, advantages = batch['states'], batch['actions'], batch['log_probs'], batch['advantages']
            with tf.GradientTape() as tape:
                new_logits = self.actor(states)
                new_log_probs = tf.map_fn(lambda j: policy_probabilities(new_logits[j], actions[j]), tf.range(tf.shape(new_logits)[0]), fn_output_signature=tf.float32)
                prob_ratio = tf.exp(new_log_probs - log_probs)
                clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                new_probs = tf.nn.softmax(new_logits)
                entropy = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-10), axis=1)
                entropy = tf.reduce_mean(entropy)
                pol_loss = -(tf.reduce_mean(tf.minimum(prob_ratio * advantages, clipped_prob_ratio * advantages)) + self.entropy_bonus * entropy)
            pol_grads = tape.gradient(pol_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(pol_grads, self.actor.trainable_variables))

    @tf.function
    def update_value_parameters(self, data):
        for batch in data:
            states, targets = batch['states'], batch['targets']
            with tf.GradientTape() as tape:
                predictions = self.critic(states)
                val_loss = tf.reduce_mean(tf.square(predictions - targets))
            val_grads = tape.gradient(val_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(val_grads, self.critic.trainable_variables))

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.next_states = []
        self.advantages = []
        self.targets = []

    def store_many(self, states, actions, rewards, log_probs, values, next_states, advantages, targets):
        self.states.extend(states)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.log_probs.extend(log_probs)
        self.values.extend(values)
        self.next_states.extend(next_states)
        self.advantages.extend(advantages)
        self.targets.extend(targets)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.next_states.clear()
        self.advantages.clear()
        self.targets.clear()

    def to_tensor(self):
        return {
            'states': tf.convert_to_tensor(self.states, dtype=tf.float32),
            'actions': tf.convert_to_tensor(self.actions, dtype=tf.int32),
            'rewards': tf.convert_to_tensor(self.rewards, dtype=tf.float32),
            'next_states': tf.convert_to_tensor(self.next_states, dtype=tf.float32),
            'log_probs': tf.convert_to_tensor(self.log_probs, dtype=tf.float32),
            'values': tf.convert_to_tensor(self.values, dtype=tf.float32),
            'advantages': tf.convert_to_tensor(self.advantages, dtype=tf.float32),
            'targets': tf.convert_to_tensor(self.targets, dtype=tf.float32)
        }

    def size(self):
        return len(self.states)

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    variant = 2
    data_dir = 'path_to_data'
    no_of_actions = 5

    config = ut.read_config("path_to_config")
    agent_hyperparameters = {
        "seed": seed,
        "no_of_actions": no_of_actions,
        "gamma": config["gamma"],
        "lam": config["lam"],
        "clip_ratio": config["clip_ratio"],
        "actors": config["actors"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "entropy_bonus": config["entropy_bonus"],
        "train_actor_iterations": config["train_actor_iterations"],
        "train_critic_iterations": config["train_critic_iterations"],
        "nn_architecture": config["nn_architecture"],
    }

    agent = PPO_Agent(agent_hyperparameters)
    train_environment = GymEnvironment(variant, data_dir)

    train_episodes = 200
    rew_train = train_environment.trainPPO(agent, train_episodes)
    ut.store_model_weights("path_to_weights", agent.actor)

    test_environment = GymEnvironment(variant, data_dir)
    test_episodes = 50
    agent.actors = 1
    rew_test = test_environment.runPPO(agent, test_episodes)
    ut.plot_rewards(np.mean(np.array(rew_test), axis=1).flatten(), 'PPO Testing Rewards')
