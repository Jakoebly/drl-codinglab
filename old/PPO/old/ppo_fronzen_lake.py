import numpy as np
import scipy.signal
import tensorflow as tf
from utils import utils as ut

from datetime import datetime
from keras import layers, optimizers
from dataclasses import dataclass
from typing import List
from environment import CustomFrozenLake
from settings import *    # import const values


class GymEnvironment:
    def __init__(self, slippery=False, render=False):

        # Create customized OpenAI Gym environment
        self.env = CustomFrozenLake(slippery, render)
        self.max_timesteps = self.env.max_timesteps
        self.state_visit_count = np.zeros(NO_STATES)     # count number of visits per state

    def trainPPO(self, agent, no_episodes):
        return self.runPPO(agent, no_episodes, training=True)

    def runPPO(self, agent, no_episodes, training=False):
        rew_total = []

        for episode in range(no_episodes):
            rew_episode = []

            # collect transitions
            for n in range(0, agent.actors):
                next_state = np.array(self.env.reset()[0]).reshape(1, -1)
                rew_agent = 0

                trajectory: List[Transition] = []
                for t in range(self.max_timesteps):
                    state = next_state
                    logit, action = agent.select_action(state)
                    next_state, reward, done, _, _ = self.env.step(action)
                    next_state = np.array(next_state).reshape(1, -1)
                    rew_agent += reward

                    # Tracking state visits for heatmap
                    self.state_visit_count[next_state[0][0]] += 1

                    # for every transition, do this
                    if training:
                        # store relevant transition information such as rewards, values, etc.
                        transition = Transition(
                            state=int(state[0][0]),
                            action=action,
                            reward=reward,
                            next_state=int(next_state[0][0]),
                            is_done=done,
                            logits=logit,
                            value=agent.calc_value(state)
                        )
                        trajectory.append(transition)

                    # TODO (optional):
                    #  replace this if structure by adding a new data structure to store a transitions list
                    #  (i.e. a trajectory) and execute the logic below as postprocessing for each trajectory
                    #  (could be done in parallel)

                    # for every trajectory, do this
                    if done or t == self.max_timesteps - 1:
                        rew_episode.append(rew_agent)
                        if training:
                            # data to store after iterations
                            states, actions, rewards, next_states, is_dones, logits, values \
                                = zip(*[(t.state, t.action, t.reward, t.next_state, t.is_done, t.logits, t.value)
                                        for t in trajectory])

                            states_array = np.array(states)
                            actions_array = np.array(actions)
                            rewards_array = np.array(rewards)
                            next_states_array = np.array(next_states)
                            logits_array = np.array(logits)
                            values_array = np.array(values)

                            # call function for calculation of advantages
                            advantage_est = agent.calc_advantage(next_state, rewards_array, values_array, done, t)

                            # calc targets for value function update
                            terminal_value = agent.calc_value(next_state) if not done else 0
                            target_input = np.append(rewards_array, terminal_value)
                            targets = agent.calc_targets(target_input)[:-1]

                            # calculate log probs for current transition
                            log_probs = [policy_probabilities(logits_array[i], actions_array[i])
                                         for i in range(len(actions_array))]

                            agent.memory.store_many(states_array, actions_array, rewards_array, log_probs, values_array,
                                                    next_states_array, advantage_est, targets)

                        break

            rew_total.append(rew_episode)
            if training:
                tf.print('-' * 200)
                tf.print(f'Episode {episode} of {no_episodes}')
                tf.print('-' * 70)
                tf.print(f'Total Rewards (per trajectory): {rew_episode}')

                # Prepare data for updates
                data = agent.memory.to_tensor()
                data_size = agent.memory.size()
                batch_size = min(agent.batch_size, data_size)   # set batch size

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

                # preprocess policy function dataset
                shuffled_dataset_policy = dataset_policy.shuffle(buffer_size=data_size, reshuffle_each_iteration=True)
                shuffled_batched_dataset_policy = shuffled_dataset_policy.batch(batch_size).repeat(
                    count=agent.train_actor_iterations)

                # preprocess value function dataset
                shuffled_dataset_value = dataset_values.shuffle(buffer_size=data_size, reshuffle_each_iteration=True)
                shuffled_batched_dataset_value = shuffled_dataset_value.batch(batch_size).repeat(
                    count=agent.train_critic_iterations)

                agent.update_policy_parameters(shuffled_batched_dataset_policy)
                tf.print(f'Updated policy function.')

                agent.update_value_parameters(shuffled_batched_dataset_value)
                tf.print(f'Updated value function')

                agent.memory.clear()
                tf.print(f'Reset agent memory.')
                tf.print('-' * 200)

        return rew_total


def policy_probabilities(logit, action):
    """
    Compute probabilities of taking actions a by using the outputs of actor NN (the logits)
    :param logit: The logits output by the actor neural network
    :param action: The action taken
    :return: Probabilities of the action
    """
    probabilities = tf.nn.softmax(logit)        # do softmax calculation to get probabilities
    log_probs = tf.math.log(probabilities + 1e-8)     # compute log probabilities with epsilon to avoid log(0)
    return log_probs[action]    # return probability for action


# Sum of discounted rewards of vectors --> useful for advantage estimates or total rewards
def discounted_cumulative_sums(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPO_Agent:
    def __init__(self, agent_hyperparameters):
        self.memory = Memory()

        # set hyperparameters
        self.seed = agent_hyperparameters["seed"]
        self.state_size = agent_hyperparameters["no_of_states"]
        self.action_size = agent_hyperparameters["no_of_actions"]
        self.gamma = agent_hyperparameters["gamma"]
        self.lam = agent_hyperparameters["lam"]
        self.clip_ratio = agent_hyperparameters["clip_ratio"]   # possibly decrease this over time
        self.actors = agent_hyperparameters["actors"]   # only really needed if setting is stochastic
        self.learning_rate = agent_hyperparameters["learning_rate"]
        self.batch_size = agent_hyperparameters["batch_size"]
        self.entropy_bonus = agent_hyperparameters["entropy_bonus"]
        self.train_actor_iterations = agent_hyperparameters["train_actor_iterations"]
        self.train_critic_iterations = agent_hyperparameters["train_critic_iterations"]
        self.nn_architecture = agent_hyperparameters["nn_architecture"]
        # Optional: Add more hyperparameters here

        # Initialize NNs
        self.actor = self.nn_model(self.state_size, self.action_size)
        self.critic = self.nn_model(self.state_size, 1)

        # Initialize optimizers
        self.actor_optimizer = self.actor.optimizer
        self.critic_optimizer = self.critic.optimizer

    def select_action(self, state):
        """
        Action selection, i.e., sample an action from policy pi
        :param state: state id
        :return: action
        """
        assert self.actor is not None, 'Actor has not been set'
        logits = self.actor.predict(state).flatten()
        action = np.random.choice(self.action_size, p=tf.nn.softmax(logits).numpy())    # sample action
        return logits, action

    def calc_advantage(self, state, rewards, values, done, t):
        """
        Advantage estimate using TD lamda as presented in the lecture.
        "To achieve a better bias/variance trade-off, the n-step temporal difference error."
        "Implement here the calculation of the advantage, e.g., using TD-lambda or eligibility traces"
        :param state: next state after state that terminated
        :param rewards: list of rewards up to this point
        :param values: list of value function estimates
        :param done: boolean indicating if episode was terminated
        :param t: index of the current timestep (i.e. the final timestep)
        :return: list of calculated advantages
        """
        assert (t + 1) == len(values) == len(rewards) > 0, 'faulty input'

        # r_t + gamma * V(s_t+1) - V(s_t)
        deltas = rewards + self.gamma * np.roll(values, -1) - values  # calculate deltas
        terminal_value = self.calc_value(state) if not done else 0
        deltas[-1] = rewards[-1] + self.gamma * terminal_value - values[-1]

        advantages = discounted_cumulative_sums(deltas, self.gamma * self.lam)  # discount and sum deltas

        return advantages

    def calc_targets(self, rewards):
        return discounted_cumulative_sums(rewards, self.gamma)

    def nn_model(self, state_size, output_size):
        """
        Defines the neural network here, accounting for the different requirements of the value
        function and policy function approximation in the in- and outputs.

        :param state_size: number of states
        :param output_size: output size
        :return: a neural network model with the given input and output sizes
        """

        # Define model layers, use one-hot encoding
        inputs = tf.keras.Input(shape=(1,))
        enc = layers.Lambda(lambda x: tf.one_hot(tf.cast(x, tf.int32), state_size))(inputs)
        enc_flat = layers.Flatten()(enc)

        prev_layer = enc_flat
        for no_neurons in self.nn_architecture:
            prev_layer = layers.Dense(no_neurons, activation='relu')(prev_layer)

        outputs = layers.Dense(output_size)(prev_layer)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate))

        return model

    def calc_value(self, state):
        """
        Calculates the value function approximation.
        :param state: current state
        :return: value estimation
        """
        val = self.critic.predict(state)
        return val[0][0]

    @tf.function  # This is a wrapper that when adding it in front of a function, consisting only of tf syntax,
    # can improve speed
    def update_policy_parameters(self, data):
        for batch in data:
            states = batch['states']
            actions = batch['actions']
            log_probs = batch['log_probs']
            advantages = batch['advantages']

            with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
                # Use the advantages and calculated policies to calculate the clipping function here and calculate
                # the loss function L(θ) = E[ min(r_t(θ) * A_t, clip(r_t(θ), 1 - ε, 1 + ε) * A_t) ]

                # compute the new logits of the actions under the current policy
                new_logits = self.actor(states)

                # calculate new probabilities for actions (workaround, as tf does not allow zipping here)
                new_log_probs = tf.map_fn(lambda j: policy_probabilities(new_logits[j], actions[j]),
                                          tf.range(tf.shape(new_logits)[0]),
                                          fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.float32))

                # r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
                prob_ratio = tf.exp(new_log_probs - log_probs)

                # clip probability ratio (up/lower bound)
                clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

                # calculate an entropy bonus to encourage more exploration and prevent oscillation
                # H(p)= −∑ p(a)⋅log(p(a)) (formula like in BAML)
                new_probs = tf.nn.softmax(new_logits)
                entropy = -tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-10), axis=1)
                entropy = tf.reduce_mean(entropy)

                # compute the policy loss function and the entropy bonus
                pol_loss = -(tf.reduce_mean(tf.minimum(prob_ratio * advantages,
                                                       clipped_prob_ratio * advantages))
                             + self.entropy_bonus * entropy)

            pol_grads = tape.gradient(pol_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(pol_grads, self.actor.trainable_variables))

    @tf.function  # This is a wrapper that when adding it in front of a function, consisting only of tf syntax,
    # can improve speed
    def update_value_parameters(self, data):
        for batch in data:
            states = batch['states']  # s_t
            targets = batch['targets']  # R_t
            with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
                # Use advantages and calculated policies to calculate the clipping function here and calculate the loss
                # function L(φ) = mean squared error (V(s_t) - R_t)
                predictions = self.critic(states)   # V(s_t)
                val_loss = tf.reduce_mean(tf.square(predictions-targets))

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
        assert (len(states) == len(actions) == len(rewards) == len(values) == len(advantages)
                == len(targets)), 'Missing values'

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
            'states': tf.convert_to_tensor(self.states, dtype=tf.int32),
            'actions': tf.convert_to_tensor(self.actions, dtype=tf.int32),
            'rewards': tf.convert_to_tensor(self.rewards, dtype=tf.float32),
            'next_states': tf.convert_to_tensor(self.next_states, dtype=tf.int32),
            'log_probs': tf.convert_to_tensor(self.log_probs, dtype=tf.float32),
            'values': tf.convert_to_tensor(self.values, dtype=tf.float32),
            'advantages': tf.convert_to_tensor(self.advantages, dtype=tf.float32),
            'targets': tf.convert_to_tensor(self.targets, dtype=tf.float32)
        }

    def size(self):
        assert (len(self.states) == len(self.actions) == len(self.rewards) == len(self.values) == len(self.advantages)
                == len(self.targets)), 'Missing values'
        return len(self.states)


@dataclass
class Transition:
    state: int
    action: int
    reward: float
    next_state: int
    is_done: bool
    logits: np.array
    value: float


if __name__ == "__main__":
    # hp.random_search_ppo_hyperparameters()
    # exit(0)

    # Set a seed for reproducibility
    seed = RANDOM_SEED
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Define whether the environment is slippery or not
    slippery = IS_SLIPPERY  # (False: deterministic, True: stochastic)

    # Define number of states and actions
    no_of_states = NO_STATES
    no_of_actions = NO_ACTIONS

    # Define agent hyperparameters and create agent
    load_old_model = DO_LOAD_MODEL
    config = ut.read_config(CONFIG_PATH)
    agent_hyperparameters = {
        "seed": seed,
        "no_of_states": no_of_states,
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
        # Optional: Add more hyperparameters here
    }
    agent = PPO_Agent(agent_hyperparameters)

    # Define parameters and create train environment
    train_episodes = TRAINING_EPISODES  # Play around with this number, default: 200
    train_render = DO_RENDER_TRAINING  # Set to True to render the GUI
    train_environment = GymEnvironment(slippery=slippery, render=train_render)

    # Train agent or load model
    if load_old_model:
        ut.load_model_weights(WEIGHT_PATH, agent.actor)
    else:
        train_start_time = datetime.now()
        rew_train = train_environment.trainPPO(agent, train_episodes)
        ut.store_model_weights(WEIGHT_PATH, agent.actor)

        # Visualize Training
        train_reward_per_episode = np.mean(np.array(rew_train), axis=1).flatten()
        max_reward_per_episode = np.max(np.array(rew_train), axis=1).flatten()
        ut.plot_performance_table(train_reward_per_episode, 'PPO Training Performance')
        ut.plot_rewards(train_reward_per_episode, 'PPO Avg. Training Rewards')
        ut.plot_rewards(max_reward_per_episode, 'PPO Training Max Rewards')
        # ut.plot_convergence_rate(train_reward_per_episode, 'PPO Training Convergence Rate')
        ut.plot_state_visit_heatmap(train_environment.state_visit_count, train_environment.env.desc)
        ut.visualize_value_function(agent.critic, no_of_states, train_environment.env.desc)
        ut.store_info(train_start_time, train_reward_per_episode, agent_hyperparameters, RUNS_PATH,
                      'PPO_train')

    # Define parameters and create test environment
    test_episodes = TEST_EPISODES  # Play around with this number, default: 50 (more episodes for better evaluation)
    agent.actors = 1  # This is set to one here as multiple actors are only required for training
    test_render = DO_RENDER_TEST  # Set to True to render the GUI
    test_environment = GymEnvironment(slippery=slippery, render=test_render)

    # Test agent
    test_start_time = datetime.now()
    rew_test = test_environment.runPPO(agent, test_episodes)

    # Test Visualization
    test_reward_per_episode = np.mean(np.array(rew_test), axis=1).flatten()
    ut.plot_performance_table(test_reward_per_episode, 'PPO Testing Performance')
    ut.plot_rewards(test_reward_per_episode, 'PPO Testing Rewards', False)

    ut.visualize_policy(agent.actor, no_of_states, test_environment.env.desc)
    ut.store_info(test_start_time, test_reward_per_episode, agent_hyperparameters, RUNS_PATH, 'PPO_test')