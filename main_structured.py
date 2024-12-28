from datetime import datetime
import os
import argparse
import random
import numpy as np
import torch
import wandb
from environment import Environment
from utils import utils as ut
from dqn_pt_test import DQN_Agent

# TODO:
# - get state size abh. von cnn oder normal
# - bash for execute
# - adjust name format for runs (i.e., without the mode)
# - naming of runs to see if validate_after_run has been done instantly
# - evtl. save rewards separately as csv and only safe avg. reward in run_info
# - implement renderer
# - NN architecture as hyperparameter for DQN
# - Extensions as hyperparameter for DQN
# - keep track of best models
# - outsource memory class / replay buffer
# - tags for wandb
# - integrate PPO (i.e., translate to Pytorch and bring to new structure



def parse_args():

    # Specify args
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--experiment_name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--algorithm", type=str, help="Algorithm to use, choose from ('dqn', 'ppo', 'sac')")
    parser.add_argument("--variant", type=int, help="Problem variant to use, choose from (0 = default, 1 = added capacity, 2 = obstacles)")
    parser.add_argument("--mode", type=str, help="Mode of the run, choose from ('train', 'validate', 'test')")
    parser.add_argument("--model_name", type=str, nargs="?", help="Name of the model to load weights for validation/testing, format: 'CL__{args.algorithm.upper()}__Var{args.variant}__{args.experiment_name}'")
    parser.add_argument("--validate_after_training", action="store_true", help="If toggled, validation will be automatically performed after training")
    # TODO: validate bzw. implement
    parser.add_argument("--save", action="store_true", help="If toggled, the run will be saved under './logs'")


    # Cuda related args
    parser.add_argument("--torch-deterministic", type=bool, default=True, nargs="?", const=True, help="If toggled, `torch.backends.cudnn.deterministic` will be set to 'False'")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?", const=True, help="If toggled, cuda will be enabled by default")


    # WandB args
    parser.add_argument("--track", type=bool, default=False, nargs="?", const=True, help="If toggled, the run will be tracked with WandB")
    parser.add_argument("--wandb-project-name", type=str, default="Coding_Lab", help="WandB's project name")
    parser.add_argument("--wandb-entity", type=str, default="gowiththetensorflow", help="Entity (team) of WandB's project")
    # TODO: check if this works with the renderer
    parser.add_argument("--capture-video", type=bool, default=False, nargs="?", const=True, help="If toggled, a video of the agent's performance will be captured")

    args = parser.parse_args()


    # Validate args
    if args.algorithm.lower() not in ["dqn", "ppo", "sac"]:
        raise ValueError("Specify a valid algorithm ('dqn', 'ppo', 'sac')!")

    if args.variant not in [0, 1, 2]:
        raise ValueError("Specify a valid problem variant (0 = default, 1 = added capacity, 2 = obstacles)!")

    if args.mode.lower() not in ["train", "validate", "test"]:
        raise ValueError("Specify a valid mode ('train', 'validate', 'test')!")

    if args.mode.lower() == "train" and args.model_name is not None:
        raise ValueError("Make sure arg 'model_name' is 'None' for mode 'train' as training creates a new model!")

    if args.mode.lower() in ["validate", "test"] and args.model_name is None:
        raise ValueError("Make sure you provide an existing model that can be used for validation/testing!")

    if args.mode.lower() in ["validate", "test"] and args.validate_after_training is True:
        raise ValueError("Arg 'validate_after_training' is not applicable to modes 'validate' or 'test'!")

    return args



if __name__ == "__main__":

    # Get args
    args = parse_args()


    # Set seeds
    seed = 43
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


    # Set run-specific variables from args
    algorithm = args.algorithm.upper()
    variant = args.variant
    mode = args.mode.lower()
    model_name = args.model_name
    validate_after_training = args.validate_after_training


    # Set data path
    data_dir = "./data"


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
    if args.track:
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project_name,
            config=config,
            name=run_name,
            save_code=True,
            # tags=[algorithm, variant, mode]
        )


    # Initialize agent depending on the algorithm
    if algorithm == "DQN":
        agent = DQN_Agent(wandb_run, env, config)
    elif algorithm == "PPO":
        agent = PPO_Agent(env, config)
    elif algorithm == "PPO":
        agent = SAC_Agent(env, config)


    # Load weights if validating/testing
    if mode in ["validate", "test"]:
        agent.load_weights(f"{model_dir}/model.pt")


    # Perform training, validation or testing
    rew_train = None
    rew_val = None
    rew_test = None

    if mode == "train":
        rew_train, rew_val = agent.train(config["num_episodes_train"], config["num_episodes_validate"], validate_after_training)
    elif mode == "validate":
        rew_val = agent.val(config["num_episodes_validate"])
    elif mode == "test":
        rew_test = agent.test(config["num_episodes_test"])


    # Log the run
    if args.save:

        # If training: create new subfolder and save weights
        if mode == "train":

            model_dir = f"./logs/Variant_{variant}/{algorithm}/{run_name}"
            os.makedirs(model_dir, exist_ok=True)

            agent.save_weights(f"{model_dir}/model.pt")

        run_info_dir = f"{model_dir}/run_info.yaml"

        # Update run_info file with relevant data
        ut.save_run_info(run_info_dir, config, mode, rew_train, rew_val, rew_test)

    # Finish wandb run
    wandb_run.finish()





