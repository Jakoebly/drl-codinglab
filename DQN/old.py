import tensorflow as tf
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm
from Utils import utils     # in case of issues with this import in vscode, try running it using launch.json
from environment import CustomFrozenLake
from collections import deque
from settings import *
import Utils.hyperparameter_tuning as hp


class GymEnvironment:

    def __init__(self, slippery=False, render=False):
        self.env = CustomFrozenLake(slippery, render)   # Creating environment
        self.max_timesteps = self.env.max_timesteps
        self.is_slippery = slippery
        self.state_visit_count = np.zeros(self.env.observation_space.n)     # to check how often a state is visited

    def trainDQN(self, agent, no_episodes, eval_interval, target_reward):
        rew = self.runDQN(agent, no_episodes, eval_interval, target_reward, training=True, evaluation=False)
        # Automatically save weights of trained network
        # agent.model.save_weights(weight_names, overwrite=True) OG implementation
        utils.store_model_weights(WEIGHT_PATH, agent.model)
        return rew

    def runDQN(self, agent, no_episodes, eval_interval, target_reward, training=False, evaluation=False):
        # Running the DQN. Weights are not updated if training is false. Evaluation flag needed for special case of
        # evaluation performance within trainings
        rew = np.zeros(no_episodes)     # rewards per episode

        # iterating over episodes
        for episode in tqdm(range(no_episodes)) if not evaluation else range(no_episodes):
            state, info = self.env.reset()  # Initializing starting state

            # state = np.reshape(state, [1, agent.state_size])
            state = np.array(state).reshape(1, -1)      # replaced OG implementation

            done = False    # done flag for termination purposes
            rwd = 0

            # Iterating over individual steps within an episode
            for t in range(self.max_timesteps):
                action = agent.select_action(state, training=training)  # Selecting action based on e-greedy
                next_state, reward, done, _, _ = self.env.step(action)
                print(f"Took action: {action}, got reward: {reward}")

                # next_state = np.reshape(next_state, [1, agent.state_size])
                next_state = np.array(next_state).reshape(1, -1)    # replaced OG implementation
                rwd += reward


                # Updates to be performed for each step within a training episode
                if training:
                    self.state_visit_count[next_state[0][0]] += 1  # Tracking state visits for heatmap

                    # Recording and saving the step instance to the replay buffer
                    agent.record(state, action, reward, next_state, done)
                    agent.update_weights()  # Updating the weights using a replay buffer
                    agent.update_target_weights(t)  # Updating the weights of the target network every t episodes

                state = next_state
                if done:
                    break

            rew[episode] = rwd

            # Updating the epsilon after each training episode as per the epsilon decay rate
            if training:
                agent.update_epsilon()

            # Function that evaluates the agent's performance for every x episodes by calling runDQN directly and
            # returning an average of total rewards for 100 runs, if the objective is reached, we terminate training
            if (episode + 1) % eval_interval == 0 and training and not evaluation:
                average_reward = agent.evaluate()
                print(f"Average Reward of last 100 runs after episode {episode + 1}: {average_reward:.2f}")
                # Checking for the training termination criteria
                if average_reward >= target_reward:
                    print(f"Training terminated after {episode + 1} episodes as the target average reward {target_reward} was achieved.")
                    # Adjusting reward array length in case of early termination (easier to handle for visualizations)
                    rew = rew[:episode + 1]
                    break   # Stop training if the target reward is reached

            print("episode: {}/{} | score: {} | e: {:.3f}".format(episode + 1, no_episodes, rew, agent.epsilon))

        return rew  # Rewards per episode as the result of training or testing
    

class DQN_Agent:
    def __init__(self, no_of_states, no_of_actions, load_old_model, agent_hyperparameters):
        self.state_size = no_of_states
        self.action_size = no_of_actions

        # Hyperparameters
        self.hyperparameters = agent_hyperparameters   # use for storing info
        self.gamma = agent_hyperparameters['gamma']    # discount rate
        self.epsilon = agent_hyperparameters['epsilon']    # eps-greedy exploration rate
        self.batch_size = agent_hyperparameters['batch_size']    # maximum size of the batches sampled from memory
        self.epsilon_min = agent_hyperparameters['epsilon_min']     # minimum eps-greedy exploration rate
        self.epsilon_decay = agent_hyperparameters['epsilon_decay']   # decay of eps-greedy exploration rate
        self.lr = agent_hyperparameters['learning_rate']  # learning rate
        self.nn_architecture = agent_hyperparameters['nn_architecture']  # nn_architecture

        # Initializing the neural network models of weight and target weight networks
        self.model = self.nn_model(no_of_states, no_of_actions, load_old_model)
        self.target_model = self.nn_model(no_of_states, no_of_actions, load_old_model)

        # Defining times at which target weights are synchronized
        self.target_model_time = 25

        # Maximal size of memory buffer
        self.memory = deque(maxlen=2000)

    def nn_model(self, state_size, action_size, load_old_model):
        # Defining your neural network & compile it

        inputs = tf.keras.Input(shape=(1,))
        enc = tf.keras.layers.Lambda(lambda x: tf.one_hot(tf.cast(x, tf.int32), state_size))(inputs)
        enc_flat = tf.keras.layers.Flatten()(enc)
        prev_layer = enc_flat
        for no_neurons in self.nn_architecture:
            prev_layer = tf.keras.layers.Dense(no_neurons, activation='relu')(prev_layer)

        outputs = tf.keras.layers.Dense(action_size)(prev_layer)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss='mse')

        # If you already have a set of weights from previous training, you can load them here
        if load_old_model == 1:
            utils.load_model_weights(WEIGHT_PATH, model)

        return model

    # Selecting an action depending on the current state and our current policy
    def select_action(self, state, training=True):
        # Random chance of exploring new action instead of acting on highest Q-value
        if training and np.random.uniform(0, 1) <= self.epsilon:
            act = random.randrange(self.action_size)
        # greedy action - acting on highest Q-value    
        else:
            Q_values = self.model.predict(state)
            act = np.argmax(Q_values[0])
            print("--------------------------------------------------------")
        return act

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

    def evaluate(self):
        print(" --- EVALUATION ROUND --- ")
        total_rewards = environment.runDQN(agent, 100, eval_interval, target_reward, training=False, evaluation=True)
        average_reward = np.mean(total_rewards)
        return average_reward

    # updates epsilon after each episode as long as it is larger than our defined minvalue
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Function to update weights of target network - check whether there is enough samples for the minibatch
    def update_target_weights(self, t):  # t for time step at which target model will be updated
        if t % self.target_model_time == 0 and len(self.memory) >= self.batch_size:
            self.target_model.set_weights(self.model.get_weights())
            print("------ TARGET WEIGHTS UPDATED --------")


# If you want to set a random seed:
seed = RANDOM_SEED
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

if __name__ == "__main__":
    # hp.random_search_dqn_hyperparameters()
    # exit(0)

    ################## TRAINING AREA ##################
    # Define whether the environment is slippery or not
    slippery = IS_SLIPPERY  # (False: deterministic, True: stochastic)
    train_render = DO_TRAIN_RENDER  # Set to True to render the GUI
    environment = GymEnvironment(slippery=slippery, render=train_render)

    # The number of states and actions
    no_of_states = environment.env.observation_space.n
    no_of_actions = environment.env.action_space.n

    #---------------- IMPORTANT -----------------
    # If you want to load weights of an already trained model, set this flag to 1
    load_old_model = int(DO_LOAD_MODEL)
    #--------------------------------------------

    # Own addition: Changing names of weights to be created/loaded
    weight_names = WEIGHT_PATH

    # Initializing agent
    config = utils.read_config(CONFIG_PATH)
    agent = DQN_Agent(no_of_states, no_of_actions, load_old_model, config)

    # Training agent parameters - DON'T CHANGE
    no_episodes = NO_TRAIN_EPISODES
    eval_interval = EVAL_INTERVAL
    target_reward = TARGET_REWARD     # reward for stopping training

    # Determining training rewards after training
    if load_old_model == 0:     # Since we only need these plots if we are training a new model
        train_start_time = datetime.now()
        rew_train = environment.trainDQN(agent, no_episodes, eval_interval, target_reward)

        utils.plot_state_visit_heatmap(environment.state_visit_count, environment.env.desc)
        utils.plot_performance_table(rew_train, 'DQN Training Performance')
        utils.plot_rewards(rew_train, 'DQN Training Rewards')
        utils.store_info(train_start_time, rew_train, agent.hyperparameters, RUNS_PATH, 'DQN train')

    ################## Testing AREA ##################
    # Run your agent
    no_episodes_run = NO_TEST_EPISODES
    test_start_time = datetime.now()
    rew_test = environment.runDQN(agent, no_episodes_run, eval_interval, target_reward)

    # Implement here a function visualizing/plotting, e.g.,
    # your agent's performance over the number of training episodes
    # Here you can watch a simulation on how your agent performs after being trained.
    # NOTE that this part will try to load an existing set of weights, therefore set visualize_agent to TRUE, when you
    # already saved a set of weights from a training session

    utils.plot_rewards(rew_test, 'DQN Testing Performance', False)
    utils.plot_performance_table(rew_test, 'DQN Testing Rewards')
    utils.store_info(test_start_time, rew_test, agent.hyperparameters, RUNS_PATH, 'DQN test')
    utils.visualize_policy(agent.model, no_of_states, environment.env.desc)
