import os
import random
from email import parser
from datetime import datetime
import numpy as np
import torch
import wandb
from environment import Environment
from src import utils as ut, args, settings
from algorithms.dqn import DQN_Agent
from algorithms.ppo import PPO_Agent
import yaml
# from algorithms.sac import SAC_Agent

# TODO:
# - get state size abh. von cnn oder normal
# - bash for execute
# - implement renderer
# - outsource memory class / replay buffer
# - integrate device handling into DQN
# - discuss functions in src
# - discuss where to store args, src, settings, renderer and how to name the folders#
# - validate sweep arg
# - integrate running of sweep from file

if __name__ == "__main__":
    # Get args
    args = args.parse_args()

    # Set seeds
    ut.set_seeds(settings.SEED, args.torch_deterministic)

    # Set device variable for CUDA usage
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Set run_name
    run_name = f"CL__{args.algorithm.upper()}__Var{args.variant}__{datetime.now().strftime('%Y%m%d_%H%M')}"
    if args.experiment_name is not None:
        run_name = f"CL__{args.algorithm.upper()}__Var{args.variant}__{datetime.now().strftime('%Y%m%d_%H%M')}__{args.experiment_name}"

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

    # Set config directory to the general config folder in case of training
    CONFIG_DIR = settings.CONFIG_DIR

    # Set config directory to the folder associated with the model's training run in case of validating/testing
    if mode in ["validate", "test"]:
        MODEL_DIR = f"./logs/Variant_{variant}/{algorithm}/{model_name}"
        CONFIG_DIR = MODEL_DIR

    # Create environment
    env = Environment(variant, DATA_DIR, mode, render=False)

    # Load config
    config = ut.load_config(env, CONFIG_DIR, algorithm)

    # Setup sweep if sweep flag is toggled
    if args.sweep:
        sweep_name = f"Sweep__{run_name}"
        spec = algorithm.lower()
        #spec = "feature"
        sweep_config = ut.load_sweep_config(CONFIG_DIR, sweep_name, spec)
        sweep_id = wandb.sweep(sweep_config, entity=args.wandb_entity, project=args.wandb_projectname)
        wandb.agent(sweep_id, entity=args.wandb_entity, project=args.wandb_projectname)


    # Login to wandb
    wandb.login(key=settings.WAND_API_KEY)
    print("Wandb Login Successful")

    # Initialize wandb run
    run = wandb.init(entity=args.wandb_entity,
                     project=args.wandb_projectname,
                     config=config,
                     name=run_name,
                     save_code=True,
                     mode="online" if args.track else "disabled",
                     group=algorithm,
                     tags=[f"Var_{variant}"],
                     job_type="Sweep" if args.sweep else (f"{mode}/validate" if mode == "train" else mode)
                     )

    config = wandb.config
    ut.update_env_features(env, config, algorithm)

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

    if mode == "train":
        rew_train = agent.run("training", config["general"]["num_episodes_train"]*2)
        rew_val = agent.run("validation", config["general"]["num_episodes_validate"])
    elif mode == "validate":
        rew_val = agent.run("validation", config["general"]["num_episodes_validate"])
    elif mode == "test":
        rew_test = agent.run("testing", config["general"]["num_episodes_test"])

    if rew_train is not None:
        wandb.log({"Charts/Avg. Train Reward": np.mean(rew_train)})
    if rew_val is not None:
        wandb.log({"Charts/Avg. Validation Reward": np.mean(rew_val)})
    if rew_test is not None:
        wandb.log({"Charts/Avg. Test Reward": np.mean(rew_test)})

    rewards = (rew_train, rew_val, rew_test)

    # Log render
    if env.render_enabled:
        train_images, val_images = env.renderer.get_images()
        if train_images is not None:
            wandb.log({"train_video": wandb.Video(train_images, fps=5, format="mp4")})
        if val_images is not None:
            wandb.log({"val_video": wandb.Video(val_images, fps=5, format="mp4")})

    # Update best model
    ut.update_best(algorithm, variant, rewards, model_name, run_name, BEST_DIR)

    # Log the run locally
    if args.save:
        ut.save_run_info(rewards, config, CONFIG_DIR, agent, algorithm, variant, mode, run_name, run)






