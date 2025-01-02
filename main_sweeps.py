import os
import random
from email import parser

import numpy as np
import torch
import wandb
from environment import Environment
from utils import utils as ut, args, settings
# from algorithms.dqn import DQN_Agent
from algorithms.ppo import PPO_Agent
# from algorithms.sac import SAC_Agent

# TODO:
# - get state size abh. von cnn oder normal
# - bash for execute
# - implement renderer
# - outsource memory class / replay buffer
# - integrate device handling into DQN
# - discuss functions in utils
# - discuss where to store args, utils, settings, renderer and how to name the folders#
# - validate sweep arg
# - integrate running of sweep from file

def model_pipeline(config):

    # Initialize agent depending on the algorithm
    if algorithm == "DQN":
        agent = DQN_Agent(env, config, device)
    elif algorithm == "PPO":
        agent = PPO_Agent(env, config, device)
    elif algorithm == "SAC":
        agent = SAC_Agent(env, config).to(device)

    # Load weights if validating/testing
    if mode in ["validate", "test"]:
        agent.load_weights(f"{MODEL_DIR}/model.pt")

    # Perform training, validation or testing
    rew_train = None
    rew_val = None
    rew_test = None

    if mode == "train" and not args.sweep:
        rew_train = agent.run("training", config.num_episodes_train)
        rew_val = agent.run("validation", config.num_episodes_validate)
    elif mode == "train" and args.sweep:
        rew_train = agent.run("training", config.num_episodes_train)
    elif mode == "validate":
        rew_val = agent.run("validation", config.num_episodes_validate)
    elif mode == "test":
        rew_test = agent.run("testing", config.num_episodes_test)

    rewards = (rew_train, rew_val, rew_test)


    # Update best model
    ut.update_best(algorithm, variant, rewards, model_name, run_name, BEST_DIR)

    # Log the run locally
    if args.save:
        ut.save_run_info(rewards, config, CONFIG_DIR, agent, algorithm, variant, mode, run_name)

if __name__ == "__main__":

    # Get args
    args = args.parse_args()

    # Set seeds
    ut.set_seeds(settings.SEED, args.torch_deterministic)

    # Set device variable for CUDA usage
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Set run_name
    run_name = f"CL__{args.algorithm.upper()}__Var{args.variant}__{args.experiment_name}"

    # Load directories from settings
    BEST_DIR = settings.BEST_DIR
    DATA_DIR = settings.DATA_DIR

    # Set run-specific variables from args
    algorithm = args.algorithm.upper()
    variant = args.variant
    mode = args.mode.lower()
    best = args.best
    model_name = args.model_name
    if best:
        model_name = ut.load_best(BEST_DIR, variant, algorithm)

    # Create environment
    env = Environment(variant, DATA_DIR, render=False)

    # Set config directory to the general config folder in case of training
    CONFIG_DIR = settings.CONFIG_DIR

    # Set config directory to the folder associated with the model's training run in case of validating/testing
    if mode in ["validate", "test"]:
        MODEL_DIR = f"./logs/Variant_{variant}/{algorithm}/{model_name}"
        CONFIG_DIR = MODEL_DIR

    # Load config
    config = ut.load_config(env, CONFIG_DIR, algorithm)

    # Initialize wandb run
    if args.track:
        run = wandb.init(entity=args.wandb_entity,
                         project=args.wandb_projectname,
                         config=config,
                         name=run_name,
                         save_code=True,
                         group=algorithm,
                         tags=[f"Var_{variant}"],
                         job_type="Sweep" if args.sweep else (f"{mode}/validate" if mode == "train" else mode)
                         )
        with run:
            config = wandb.config
            model_pipeline(config)
    else:
        model_pipeline(config)

