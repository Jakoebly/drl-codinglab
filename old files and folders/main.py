import argparse
import datetime
import os
import random
import numpy as np
import tensorflow as tf
import torch
from sac import SAC_Agent
from environment import Environment
import wandb
# from SAC import sac
# from DQN import dqn
# from PPO import ppo

# TODO: add functions and if name == main

def parse_args():
    # Specify args
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--experiment_name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M")) # TODO: difference exp name and run name
    parser.add_argument("--algorithm", type=str, help="algorithm to use ('dqn', 'ppo', 'sac')")
    parser.add_argument("--variant", type=int, help="problem variant (0 = default, 1 = added capacity, 2 = obstacles)")
    parser.add_argument("--mode", type=int, help="mode of the run ('train', 'test', 'validate')")

    # TODO: specify the below and decide which to use
    # Torch cuda related args
    parser.add_argument("--torch-deterministic", type=bool, default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")#

    # Wandb args
    # Flag to track on wandb
    parser.add_argument("--track", type=bool, default=False, nargs="?", const=True, help="if toggled, this experiment will be tracked with wandb")
    # Project name
    parser.add_argument("--wandb-project-name", type=str, default="coding lab", help="wandb's project name") # TODO: specify default
    # Entity name
    parser.add_argument("--wandb-entity", type=str, default=None, help="entity (team) of wandb's project") # TODO: specify default
    # Capture video
    parser.add_argument("--capture-video", type=bool, default=False, nargs="?", const=True, help="weather to capture videos of the agent performances")

    # TODO: think about which ones are necessary
    parser.add_argument("--validation_model", type=str, nargs="?", help="Model weights for validation only")
    parser.add_argument("--save", action="store_true", help="Indicate wether to save the run as JSON/CSV")
    args = parser.parse_args()


    # Ensure args are valid
    if args.algorithm.lower() not in ["dqn", "ppo", "sac"]:
        raise ValueError("Specify a valid algorithm ('dqn', 'ppo', 'sac')!")

    if args.variant not in [0, 1, 2]:
        raise ValueError("Specify a valid problem variant (0 = default, 1 = added capacity, 2 = obstacles)!")

    if args.mode not in ["train", "test", "validate"]:
        raise ValueError("Specify a valid mode ('train', 'test', 'validate')!")

    return args



if __name__ == "__main__":

    # Get args
    args = parse_args()

    variant = args.variant
    mode = args.mode
    run_name = f"CL__{args.algorithm.upper()}__Var{args.variant}__{args.mode}__{args.experiment_name}" # "CL__PPO__Var0__Train__20241221_1445"

    # Wandb Setup
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),  # TODO: add config here
            name=run_name,  # TODO: specify run name
            save_code=True,
        )

    # Set seeds
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)  # TODO: delete if tf is not used
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic




# Set paths
data_dir = "../data"


# Initialize environment

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
    agent = DQN_Agent(gamma = 0.99,
                      epsilon = 1,
                      batch_size = 32,
                      epsilon_min = 0.18,
                      epsilon_decay = 0.99,
                      learning_rate = 0.01,
                      target_update_interval = 25,
                      eval_interval = 50,
                      mem_maxlen = 1000)


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



rewards = agent.run(no_episodes = no_episodes[args.mode], mode = args.mode)


# TODO: Save results (rewards, actions, and environment) for each episode in subfolder
if args.save_results:
    results_path = f"./results/{args.experiment_name}_{args.algorithm}_{args.mode}.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"rewards": rewards, "replay_buffer": list(replay_buffer)}, f)
        
# TODO: Save models from training

# TODO: Visualization
if args.visualize:
    #visualize_results(rewards)








