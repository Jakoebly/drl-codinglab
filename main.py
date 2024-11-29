import argparse
import datetime
import os
import random
import numpy as np
import tensorflow as tf

from sac import SAC_Agent

from environment import Environment  # initialize environment
from SAC import sac
from DQN import dqn
from PPO import ppo

# DONE: parse arguments
parser = argparse.ArgumentParser(description="")  # instantiate the parser
parser.add_argument("--experiment_name", type=str, default=datetime.now().strftime("%Y-%m-%d-%H.%M"))
parser.add_argument("--variant", type=int, help="Problem variant (0 = default, 1 = larger capacity, 2 = obstacles)")
parser.add_argument("--algorithm", type=str, help="Algorithm of the agent (dqn, ppo, sac)")  # required pos arg
parser.add_argument("--validation_model", type=str, nargs="?", help="Model weights for validation only")  # required pos arg
parser.add_argument("--test", action="store_true", help="Do a test run")  # switch arg
parser.add_argument("--save", action="store_true", help="Save run as JSON/CSV")  # switch arg
args = parser.parse_args()  # parse
if args.variant not in [0, 1, 2] or args.algorithm.lower() not in ["dqn", "ppo", "sac"]:
    raise ValueError("Invalid parser arguments")  # validate inputs

# Set seed
seed = 42  # DONE: set seed to allow for reproducibility of results
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Set paths and initialize environment
data_dir = "./data"  # DONE: specify relative path to data directory (e.g., './data', not './data/variant_0')
variant = args.variant  # DONE: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
env = Environment(variant, data_dir)

# DONE: execute training
if args.test:
    pass  # TODO: call test_policy.py
else:
    if args.validation:
        # TODO: Perform only validation with model weights given by path
        pass
    else:
        os.system(f"{args.algorithm}.py")


no_episodes = {  # Length of CSV instead of hard-coding
    "training": 800,
    "validation": 100,
    "testing": 100,
}

if args.algorithm.lower() == "dqn":
    # fmt: off
    # TODO: Add as config import
    agent = DQN_Agent(gamma = 0.95,
                      epsilon = 1,
                      batch_size = 32,
                      epsilon_min = 0.18,
                      epsilon_decay = 0.99,
                      learning_rate = 0.01,
                      target_update_interval = 25,
                      eval_interval = 50,
                      mem_maxlen = 1000)
    # fmt: on


elif args.algorithm.lower() == "sac":
    # fmt: off
    agent = SAC_Agent(env=env,
                      gamma=0.995,
                      initial_alpha=1,
                      alpha_decay=0.9,
                      rho=0.5,
                      mem_maxlen=1000,
                      batch_size=10)
    # fmt: on
    #rewards = agent.run_SAC(no_episodes=no_episodes[args.mode], mode=args.mode, visualize_results=True)

rewards = agent.run(no_episodes = no_episodes[args.mode], mode = args.mode, visualize_results = args.visualize)


# TODO: Save results (rewards, actions, and environment) for each episode in subfolder
if args.save_results:
    results_path = f"./results/{args.experiment_name}_{args.algorithm}_{args.mode}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"rewards": rewards, "replay_buffer": list(replay_buffer)}, f)
# TODO: Save models from training

# TODO: Visualization
if args.visualize:
    ut.visualize_results(rewards)