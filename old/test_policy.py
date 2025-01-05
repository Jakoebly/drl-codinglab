# compute average reward per test episode with trained policy


from environment import Environment
from ppo_old import PPOAgent
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

 
def test_policy(env):
    test_rew_plot = []
    test_rew_avg = 0. # initialize reward tracking

    for i in range(100):  # loop over 100 test episodes

        print("-" * 200)
        print(f"EPISODE: {i+1} / 100")
        print("-" * 200)

        test_rew_traj = 0

        obs = env.reset('test')  # get initial obs

        for j in range(200):  # loop over 200 steps per episode

            print("-" * 10, f" Step: {j+1} / {200} ", "-" * 10)

            act = agent.get_action(obs)[0]  # TODO: get action for the obs from your trained policy
            print(act)
            rew, next_obs, _ = env.step(act)  # take one step in the environment
            test_rew_avg += rew  # track rewards
            test_rew_traj += rew
            obs = next_obs  # continue from the new obs

        test_rew_plot.append(test_rew_traj)

    avg_test_rew = test_rew_avg / 100  # compute the average reward per episode


    print(f"Avg. Val Reward: {avg_test_rew}")  # print the result

    # Create Time-Series Line Chart
    df = pd.DataFrame(np.array(test_rew_plot), columns=["Rewards"])
    plt.figure(figsize=(10, 6))
    plt.plot(df["Rewards"], label="Episode Data", color="#0066BD")
    plt.title("PPO Performance")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()


if __name__ == '__main__':

    data_dir = "../data"  # TODO: specify relative path to data directory (e.g., './data', not './data/variant_0')
    variant = 0  # TODO: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)

    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    num_episodes = 800
    num_trajectories = 5
    num_steps_trajectory = 200
    num_minibatches = 40
    num_epochs = 10
    num_updates = num_episodes // num_trajectories

    nn_architecture = [128, 128, 128]
    state_size = 136
    num_actions = 5
    learning_rate = 0.0002
    gamma = 0.99
    lam = 0.95
    clip_ratio = 0.2
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


    # GymEnvironment = ppo.GymEnvironment(variant, data_dir=data_dir, agent_hyperparameters=agent_hyperparameters, mode="testing")
    env = Environment(variant=variant, data_dir=data_dir)  # initialize the environment
    agent = PPOAgent(agent_hyperparameters, 800)

    agent.load_weights()
    test_policy(env)  # test the trained policy

