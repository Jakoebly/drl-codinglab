import os
import random
import numpy as np
import torch
import wandb
from environment import Environment
from utils import utils as ut, args
from algorithms.dqn import DQN_Agent
from algorithms.ppo import PPO_Agent
# from algorithms.sac import SAC_Agent

# TODO:
# - get state size abh. von cnn oder normal
# - bash for execute
# - implement renderer
# - outsource memory class / replay buffer
# - integrate PPO (i.e., translate to Pytorch and bring to new structure)
# - integrate device handling into DQN


if __name__ == "__main__":

    # Get args
    args = args.parse_args()

    # Set seeds
    seed = 42
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


    # Set device variable for CUDA usage
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    # Set run_name
    run_name = f"CL__{args.algorithm.upper()}__Var{args.variant}__{args.experiment_name}"

    # Set data path
    data_dir = "./data"

    # Set best path
    best_dir = "./algorithms/best.yaml"


    # Set run-specific variables from args
    algorithm = args.algorithm.upper()
    variant = args.variant
    mode = args.mode.lower()
    best = args.best
    model_name = args.model_name
    if best:
        model_name = ut.load_best(best_dir, variant, algorithm)


    # Make environment
    env = Environment(variant, data_dir)


    # Set config directory to the general config folder in case of training
    config_dir = "./config"
    # Set config directory to the folder associated with the model's training run in case of validating/testing
    if mode in ["validate", "test"]:
        model_dir = f"./logs/Variant_{variant}/{algorithm}/{model_name}"
        config_dir = model_dir


    # Load config
    config = ut.load_config(env, config_dir, algorithm)


    # Initialize wandb run
    wandb_run = None
    if args.track:
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project_name,
            config=config,
            name=run_name,
            save_code=True,
            group=algorithm,
            tags=[f"Var_{variant}"],
            job_type=f"{mode}/validate" if mode == "train" else mode
        )


    # Initialize agent depending on the algorithm
    if algorithm == "DQN":
        agent = DQN_Agent(wandb_run, env, config, device)
    elif algorithm == "PPO":
        agent = PPO_Agent(wandb_run, env, config, device)
    elif algorithm == "SAC":
        agent = SAC_Agent(env, config).to(device)


    # Load weights if validating/testing
    if mode in ["validate", "test"]:
        agent.load_weights(f"{model_dir}/model.pt")


    # Perform training, validation or testing
    rewards = agent.run(mode)

    # Update best model
    ut.update_best(algorithm, variant, rewards, model_name, run_name, best_dir)

    # Log the run
    if args.save:
        ut.save_run_info(rewards, config, config_dir, agent, algorithm, variant, mode, run_name)

    # Finish wandb run
    wandb_run.finish()





