import argparse
import datetime
import os
import random
import numpy as np
import tensorflow as tf
from environment import Environment  # initialize environment

# DONE: parse arguments
parser = argparse.ArgumentParser(description="")  # instantiate the parser
parser.add_argument(
    "--experiment_name", type=str, default=datetime.now().strftime("%Y-%m-%d-%H.%M")
)
parser.add_argument(
    "--variant", type=int, help="Problem variant (0=default, 1=larger capacity, 2=obstacles)"
)
parser.add_argument(
    "--algorithm", type=str, help="Algorithm of the agent (dqn, ppo, sac)"
)  # required pos arg
parser.add_argument(
    "--validation_model", type=str, nargs="?", help="Model weights for validation only"
)  # required pos arg
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
variant = f"./data/variant_{args.variant}"  # DONE: specify problem variant (0 for base variant, 1 for first extension, 2 for second extension)
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

# TODO: Save results (rewards, actions, and environment) for each episode in subfolder
# TODO: Save models from training
